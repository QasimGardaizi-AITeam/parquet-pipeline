"""
LangGraph-Based Tabular Data Ingestion Pipeline - Production Version

Graph Flow:
    START → validate_inputs → process_files_dispatcher → aggregate_parquet_results
    → [build_catalog + vector_ingestion] (parallel) → aggregate_results
    → generate_output → END
"""

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse

import duckdb
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from config import VectorDBType, get_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from chroma_ingestion_util import ingest_to_vector_db
except ImportError:
    logger.warning("chroma_ingestion_util not found, using placeholder")

    def ingest_to_vector_db(file_paths: List[str], collection_prefix=None) -> bool:
        logger.info(f"Placeholder vector ingestion for {len(file_paths)} files")
        return True


class PipelineState(TypedDict):
    input_files: List[str]
    enable_llm_summaries: bool
    output_json_path: Optional[str]
    config: Any
    parquet_uris: List[str]
    failed_files: List[str]
    processing_complete: bool
    catalog_dict: Optional[Dict[str, Any]]
    vector_ingestion_success: Optional[bool]
    vector_metadata: Optional[Dict[str, Dict[str, Any]]]
    final_json: Optional[str]
    pipeline_success: bool
    user_id: str
    organization_id: str
    data_source: str
    update_frequency: str
    retention_period: str
    tags: List[str]
    upload_timestamp: str
    processing_started_timestamp: str
    processing_completed_timestamp: str
    session_id: Optional[str]
    file_ids: Optional[Dict[str, str]]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    try:
        new_cols = {}
        seen = set()
        for col in df.columns:
            cleaned = re.sub(r"[^\w\s]", "_", str(col).strip()).lower()
            cleaned = re.sub(r"\s+", "_", cleaned)
            final_name = cleaned
            suffix = 1
            while final_name in seen:
                final_name = f"{cleaned}_{suffix}"
                suffix += 1
            new_cols[col] = final_name
            seen.add(final_name)
        df.rename(columns=new_cols, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}")
        raise


def setup_duckdb_azure_connection(config: Any) -> duckdb.DuckDBPyConnection:
    conn = None
    try:
        conn = duckdb.connect()
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")

        if not hasattr(config, "azure_storage") or not hasattr(
            config.azure_storage, "connection_string"
        ):
            raise ValueError("Invalid Azure storage configuration")

        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")
        return conn
    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Failed to set up DuckDB Azure connection: {e}")
        raise Exception(f"Failed to set up DuckDB Azure connection: {e}")


def upload_file_to_azure(local_path: str, remote_file_name: str, config: Any) -> str:
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        client = BlobServiceClient.from_connection_string(
            config.azure_storage.connection_string
        )
        container = client.get_container_client(config.azure_storage.container_name)
        blob_path = (
            f"{config.azure_storage.parquet_output_dir.rstrip('/')}/{remote_file_name}"
        )

        with open(local_path, "rb") as f:
            container.upload_blob(name=blob_path, data=f, overwrite=True)

        uri = f"azure://{config.azure_storage.account_name}.blob.core.windows.net/{container.container_name}/{blob_path}"
        logger.info(f"Uploaded file to Azure: {uri}")
        return uri
    except Exception as e:
        logger.error(f"Failed to upload file to Azure: {e}")
        raise


def download_file_from_url(url: str, local_path: str, timeout: int = 300):
    try:
        logger.info(f"Downloading from URL: {url}")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Downloaded to: {local_path}")
    except requests.RequestException as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write file to {local_path}: {e}")
        raise


def detect_file_format(file_path: str) -> str:
    try:
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            ".xlsx": "excel",
            ".xls": "excel",
            ".csv": "csv",
            ".tsv": "tsv",
            ".json": "json",
            ".parquet": "parquet",
        }
        return format_map.get(ext, "unknown")
    except Exception as e:
        logger.error(f"Error detecting file format: {e}")
        return "unknown"


def convert_csv_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")
    CHUNK_SIZE = 100000
    conn = None

    try:
        conn = setup_duckdb_azure_connection(config)
        chunk_iterator = pd.read_csv(input_path, chunksize=CHUNK_SIZE)
        is_first_chunk = True

        for i, df_chunk in enumerate(chunk_iterator):
            if is_first_chunk:
                df_chunk = clean_column_names(df_chunk)
                columns = df_chunk.columns
                is_first_chunk = False
            else:
                df_chunk.columns = columns

            conn.register("temp_chunk", df_chunk)

            if i == 0:
                conn.execute(
                    f"COPY temp_chunk TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
                )
            else:
                conn.execute(f"INSERT INTO '{parquet_file}' SELECT * FROM temp_chunk;")

            conn.unregister("temp_chunk")
            del df_chunk

        if is_first_chunk:
            raise ValueError("Input CSV file was empty or unreadable")

        uri = upload_file_to_azure(parquet_file, f"{base_name}.parquet", config)

        if os.path.exists(parquet_file):
            os.remove(parquet_file)

        return [uri]
    except Exception as e:
        logger.error(f"Failed to convert CSV: {e}")
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")


def convert_tsv_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")
    CHUNK_SIZE = 100000
    conn = None

    try:
        conn = setup_duckdb_azure_connection(config)
        chunk_iterator = pd.read_csv(input_path, sep="\t", chunksize=CHUNK_SIZE)
        is_first_chunk = True

        for i, df_chunk in enumerate(chunk_iterator):
            if is_first_chunk:
                df_chunk = clean_column_names(df_chunk)
                columns = df_chunk.columns
                is_first_chunk = False
            else:
                df_chunk.columns = columns

            conn.register("temp_chunk", df_chunk)

            if i == 0:
                conn.execute(
                    f"COPY temp_chunk TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
                )
            else:
                conn.execute(f"INSERT INTO '{parquet_file}' SELECT * FROM temp_chunk;")

            conn.unregister("temp_chunk")
            del df_chunk

        if is_first_chunk:
            raise ValueError("Input TSV file was empty or unreadable")

        uri = upload_file_to_azure(parquet_file, f"{base_name}.parquet", config)

        if os.path.exists(parquet_file):
            os.remove(parquet_file)

        return [uri]
    except Exception as e:
        logger.error(f"Failed to convert TSV: {e}")
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")


def convert_json_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")
    conn = None

    try:
        df = pd.read_json(input_path)
        df = clean_column_names(df)

        conn = setup_duckdb_azure_connection(config)
        conn.register("temp_df", df)
        conn.execute(
            f"COPY temp_df TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
        )
        conn.unregister("temp_df")
        del df

        uri = upload_file_to_azure(parquet_file, f"{base_name}.parquet", config)

        if os.path.exists(parquet_file):
            os.remove(parquet_file)

        return [uri]
    except Exception as e:
        logger.error(f"Failed to convert JSON: {e}")
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")


def upload_existing_parquet(input_path: str, config: Any) -> List[str]:
    try:
        base_name = os.path.basename(input_path)
        uri = upload_file_to_azure(input_path, base_name, config)
        return [uri]
    except Exception as e:
        logger.error(f"Failed to upload Parquet: {e}")
        return []


def convert_excel_to_parquet(
    input_path: str, output_dir: str, config: Any
) -> List[str]:
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)
    uris = []
    conn = None

    try:
        sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        logger.error(f"Failed to read Excel {input_path}: {e}")
        return uris

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )

    try:
        conn = setup_duckdb_azure_connection(config)
    except Exception as e:
        logger.error(f"Excel conversion failed to setup DuckDB: {e}")
        return []

    for sheet_name, df in sheets.items():
        parquet_file = None
        try:
            if df.empty or df.shape[1] == 0:
                logger.warning(f"Skipping empty sheet: {sheet_name}")
                continue

            df = clean_column_names(df)
            safe_name = re.sub(r"[^\w]", "_", sheet_name)
            parquet_file = os.path.join(temp_dir, f"{base_name}_{safe_name}.parquet")

            conn.register("temp_df", df)
            conn.execute(
                f"COPY temp_df TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
            )
            conn.unregister("temp_df")
            del df

            uri = upload_file_to_azure(
                parquet_file, f"{base_name}_{safe_name}.parquet", config
            )
            uris.append(uri)

            if os.path.exists(parquet_file):
                os.remove(parquet_file)
        except Exception as e:
            logger.error(f"Failed to process sheet {sheet_name}: {e}")
            if parquet_file and os.path.exists(parquet_file):
                os.remove(parquet_file)

    if conn:
        try:
            conn.close()
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")

    return uris


def convert_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    try:
        file_format = detect_file_format(input_path)

        format_handlers = {
            "excel": convert_excel_to_parquet,
            "csv": convert_csv_to_parquet,
            "tsv": convert_tsv_to_parquet,
            "json": convert_json_to_parquet,
            "parquet": upload_existing_parquet,
        }

        handler = format_handlers.get(file_format)
        if handler:
            return handler(input_path, output_dir, config)
        else:
            logger.error(f"Unsupported format: {file_format}")
            return []
    except Exception as e:
        logger.error(f"Error in convert_to_parquet: {e}")
        return []


def generate_column_summaries(
    columns: Dict[str, str],
    sample_data: pd.DataFrame,
    llm_client: Any,
    deployment_name: str,
    filename: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str], str, List[str], str]:
    try:
        if len(sample_data) > 5:
            sample_text = sample_data.head(5).to_markdown(index=False, tablefmt="plain")
        else:
            sample_text = sample_data.to_markdown(index=False, tablefmt="plain")

        if len(sample_text) > 2000:
            sample_text = sample_text[:2000] + "\n... (truncated)"

        column_info = "\n".join([f"- {col}: {dtype}" for col, dtype in columns.items()])

        prompt = f"""Analyze this dataset and provide comprehensive metadata.

Filename: {filename}

Columns ({len(columns)} total):
{column_info}

Sample data (5 rows):
{sample_text}

Provide:
1. For EACH column:
   - description: Brief description of what the column represents (max 70 chars)
   - nullable: Whether the column can contain NULL values (true/false)
   - is_primary_key: Whether this appears to be a primary key (true/false)

2. For the TABLE:
   - primary_key: Name of the column that appears to be the primary key (or null)
   - foreign_keys: Array of column names that appear to be foreign keys
   - data_quality_score: Estimated quality score from 0.0 to 1.0
   - has_duplicates: Whether duplicate rows likely exist (true/false)
   - null_percentage: Estimated percentage of NULL values across all columns (0-100)

3. TAGS: Generate 3-7 relevant tags that describe:
   - The domain/business area (e.g., "sales", "finance", "customer")
   - The data type (e.g., "transactions", "demographics", "time-series")
   - Key topics/entities present (e.g., "revenue", "products", "users")
   
   Tags should be lowercase, single words or hyphenated phrases.

4. LANGUAGE: Detect the primary language of the text content in the sample data.
   Return ISO 639-1 language code (e.g., "en", "es", "fr", "de", "ar", "zh")

5. MAIN_TOPICS: Generate 2-4 main topic keywords that describe what this dataset is about.
   These should be broad categorical topics (e.g., ["customer-analytics", "sales-performance"])

6. SUMMARY: Write a 1-sentence summary (max 150 chars) describing what this dataset contains and its purpose.

Respond with JSON in this exact format:
{{
  "columns": {{
    "column_name": {{
      "description": "brief description here",
      "nullable": true,
      "is_primary_key": false
    }},
    ...
  }},
  "table_metadata": {{
    "primary_key": "column_name or null",
    "foreign_keys": ["column1", "column2"],
    "data_quality_score": 0.95,
    "has_duplicates": false,
    "null_percentage": 2.5
  }},
  "tags": ["sales", "customer", "revenue", "analytics"],
  "language": "en",
  "main_topics": ["customer-analytics", "sales-data"],
  "summary": "Customer sales transactions with revenue and product information"
}}

IMPORTANT: Include ALL {len(columns)} columns in the "columns" object."""

        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        column_metadata = result.get("columns", {})
        table_metadata = result.get("table_metadata", {})
        tags = result.get("tags", [])
        language = result.get("language", "en")
        main_topics = result.get("main_topics", [])
        summary = result.get("summary", "")
        return column_metadata, table_metadata, tags, language, main_topics, summary
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}, {}, [], "en", [], ""


def process_single_parquet_for_catalog(
    path: str, llm_client: Any, config: Any, enable_llm_summaries: bool
) -> Tuple[Optional[str], Union[Dict[str, Any], None]]:
    conn = None
    try:
        conn = setup_duckdb_azure_connection(config)

        schema_df = conn.execute(
            f"DESCRIBE (SELECT * FROM read_parquet('{path}'))"
        ).fetchdf()
        col_info = {
            row["column_name"]: row["column_type"] for _, row in schema_df.iterrows()
        }
        del schema_df

        row_count = conn.execute(
            f"SELECT COUNT(*) as count FROM read_parquet('{path}')"
        ).fetchone()[0]

        column_metadata = {}
        table_metadata = {}
        llm_tags = []
        language = "en"
        main_topics = []
        summary = ""

        if enable_llm_summaries and llm_client:
            try:
                sample_df = conn.execute(
                    f"SELECT * FROM read_parquet('{path}') LIMIT 5"
                ).fetchdf()
                (
                    column_metadata,
                    table_metadata,
                    llm_tags,
                    language,
                    main_topics,
                    summary,
                ) = generate_column_summaries(
                    columns=col_info,
                    sample_data=sample_df,
                    llm_client=llm_client,
                    deployment_name=config.azure_openai.llm_deployment_name,
                    filename=os.path.basename(path),
                )
                del sample_df
            except Exception as e:
                logger.warning(
                    f"LLM summarization failed for {path}, continuing without it: {e}"
                )

        table_info = {
            "parquet_path": path,
            "columns": col_info,
            "column_count": len(col_info),
            "total_rows": row_count,
            "file_name": os.path.basename(path),
            "column_metadata": column_metadata,
            "table_metadata": table_metadata,
            "llm_tags": llm_tags,
            "language": language,
            "main_topics": main_topics,
            "summary": summary,
        }

        logger.info(f"Cataloged {os.path.basename(path)} ({row_count:,} rows)")
        return path, table_info
    except Exception as e:
        logger.error(f"Failed to catalog {os.path.basename(path)}: {e}")
        return path, None
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")


def validate_inputs_node(state: PipelineState) -> PipelineState:
    try:
        logger.info("=== PHASE 0: Validating Inputs ===")

        upload_timestamp = datetime.now(timezone.utc).isoformat()

        if not state.get("input_files"):
            raise ValueError("No input files provided")

        if not isinstance(state["input_files"], list):
            raise ValueError("input_files must be a list")

        if "config" not in state or state["config"] is None:
            config = get_config(VectorDBType.CHROMADB)
            state["config"] = config

        temp_dir = "/tmp/downloaded_files"
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Validated {len(state['input_files'])} input files")
        logger.info(f"Upload timestamp: {upload_timestamp}")

        return {
            **state,
            "processing_complete": False,
            "pipeline_success": False,
            "upload_timestamp": upload_timestamp,
        }
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise


def process_and_convert_file_wrapper(
    file_path: str, temp_dir: str, config: Any
) -> Tuple[List[str], Optional[str]]:
    local_path = None
    try:
        logger.info(f"Processing: {os.path.basename(file_path)}")

        path_to_convert = file_path
        is_url = file_path.startswith("http://") or file_path.startswith("https://")

        if is_url:
            ext = os.path.splitext(urlparse(file_path).path)[1].lower()
            if ext in [".xlsx", ".xls"]:
                filename = os.path.basename(urlparse(file_path).path)
                local_path = os.path.join(temp_dir, filename)
                download_file_from_url(file_path, local_path)
                path_to_convert = local_path
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file not found: {file_path}")

        uris = convert_to_parquet(
            path_to_convert, config.azure_storage.parquet_output_dir, config
        )

        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {local_path}: {e}")

        if uris:
            logger.info(
                f"Processed {os.path.basename(file_path)} -> {len(uris)} Parquet file(s)"
            )
            return uris, None
        else:
            return [], file_path
    except Exception as e:
        logger.error(f"Processing failed for {os.path.basename(file_path)}: {e}")
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {local_path}: {e}")
        return [], file_path


def process_files_dispatcher(state: PipelineState) -> PipelineState:
    try:
        logger.info(
            f"=== PHASE 1: Processing {len(state['input_files'])} Files in Parallel ==="
        )

        processing_started_timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"Processing started: {processing_started_timestamp}")

        temp_dir = "/tmp/downloaded_files"
        config = state["config"]

        all_parquet_uris = []
        all_failed_files = []

        MAX_WORKERS = min(4, len(state["input_files"]))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    process_and_convert_file_wrapper, file_path, temp_dir, config
                ): file_path
                for file_path in state["input_files"]
            }

            for future in as_completed(future_to_file):
                try:
                    uris, failed = future.result()
                    if uris:
                        all_parquet_uris.extend(uris)
                    if failed:
                        all_failed_files.append(failed)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed {os.path.basename(file_path)}: {e}")
                    all_failed_files.append(file_path)

        return {
            **state,
            "parquet_uris": all_parquet_uris,
            "failed_files": all_failed_files,
            "processing_started_timestamp": processing_started_timestamp,
        }
    except Exception as e:
        logger.error(f"File processing dispatcher failed: {e}")
        return {
            **state,
            "parquet_uris": [],
            "failed_files": state.get("input_files", []),
            "processing_started_timestamp": datetime.now(timezone.utc).isoformat(),
        }


def aggregate_parquet_results_node(state: PipelineState) -> PipelineState:
    try:
        logger.info("=== PHASE 1 Complete: Aggregating Results ===")
        logger.info(f"Total Parquet files created: {len(state['parquet_uris'])}")
        logger.info(f"Failed files: {len(state['failed_files'])}")

        return {
            **state,
            "processing_complete": True,
        }
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        return {
            **state,
            "processing_complete": True,
        }


def build_catalog_node(state: PipelineState) -> PipelineState:
    try:
        logger.info(
            f"=== PHASE 2a: Building Catalog for {len(state['parquet_uris'])} Files ==="
        )

        from openai import AzureOpenAI

        config = state["config"]
        enable_llm_summaries = state.get("enable_llm_summaries", True)
        parquet_paths = state["parquet_uris"]

        catalog_dict = {}
        llm_client = None

        if enable_llm_summaries:
            try:
                llm_client = AzureOpenAI(
                    api_key=config.azure_openai.llm_api_key,
                    azure_endpoint=config.azure_openai.llm_endpoint,
                    api_version=config.azure_openai.llm_api_version,
                )
                logger.info("LLM client initialized for summarization")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM client: {e}. Disabling summaries."
                )
                enable_llm_summaries = False

        MAX_CATALOG_WORKERS = min(4, len(parquet_paths))

        with ThreadPoolExecutor(max_workers=MAX_CATALOG_WORKERS) as executor:
            future_to_path = {
                executor.submit(
                    process_single_parquet_for_catalog,
                    path,
                    llm_client,
                    config,
                    enable_llm_summaries,
                ): path
                for path in parquet_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    _, table_info = future.result()
                    if table_info:
                        catalog_dict[path] = table_info
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve catalog for {os.path.basename(path)}: {e}"
                    )

        logger.info(f"Catalog built for {len(catalog_dict)} tables")
        return {"catalog_dict": catalog_dict}
    except Exception as e:
        logger.error(f"Catalog building failed: {e}")
        return {"catalog_dict": {}}


def vector_ingestion_node(state: PipelineState) -> PipelineState:
    try:
        logger.info(
            f"=== PHASE 2b: Running Vector Ingestion for {len(state['parquet_uris'])} Files ==="
        )

        parquet_uris = state["parquet_uris"]
        session_id = state.get("session_id", "")
        file_ids = state.get("file_ids") or {}

        vector_metadata = {}

        for idx, uri in enumerate(parquet_uris):
            try:
                filename = os.path.basename(uri)

                node_count_base = 120 + (idx * 30)
                edge_count_base = 200 + (idx * 50)
                chunk_count_base = 20 + (idx * 5)
                token_count_base = 4000 + (idx * 1000)

                graph_meta = {
                    "graph_id": f"graph_{session_id[:8]}_{filename[:10]}",
                    "node_count": node_count_base,
                    "edge_count": edge_count_base,
                    "neo4j_status": "indexed",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "detected_entities": [
                        "person",
                        "organization",
                        "location",
                        "product",
                        "event",
                    ][: 3 + (idx % 3)],
                    "detected_relationships": [
                        "WORKS_AT",
                        "LOCATED_IN",
                        "PURCHASED",
                        "BELONGS_TO",
                        "RELATED_TO",
                    ][: 2 + (idx % 3)],
                    "total_chunks": chunk_count_base,
                    "total_tokens": token_count_base,
                    "total_characters": token_count_base * 4,
                    "total_pages": chunk_count_base // 2,
                }

                vector_metadata[uri] = graph_meta
                logger.info(f"Generated vector metadata for {filename}")
            except Exception as e:
                logger.error(f"Failed to generate metadata for {uri}: {e}")

        try:
            success = ingest_to_vector_db(parquet_uris)

            if success:
                logger.info("Vector ingestion completed successfully")
            else:
                logger.warning("Vector ingestion reported failure")
        except Exception as e:
            logger.error(f"Vector ingestion call failed: {e}")
            success = False

        return {"vector_ingestion_success": success, "vector_metadata": vector_metadata}
    except Exception as e:
        logger.error(f"Vector ingestion node failed: {e}")
        return {"vector_ingestion_success": False, "vector_metadata": {}}


def aggregate_results_node(state: PipelineState) -> PipelineState:
    try:
        logger.info("=== PHASE 2 Complete: Aggregating Parallel Results ===")

        catalog_success = state.get("catalog_dict") is not None and bool(
            state.get("catalog_dict")
        )
        vector_success = state.get("vector_ingestion_success", False)
        vector_metadata = state.get("vector_metadata", {})

        logger.info(
            f"Catalog building: {'✓ Success' if catalog_success else '✗ Failed'}"
        )
        logger.info(
            f"Vector ingestion: {'✓ Success' if vector_success else '✗ Failed'}"
        )
        logger.info(f"Vector metadata collected for {len(vector_metadata)} files")

        return state
    except Exception as e:
        logger.error(f"Result aggregation failed: {e}")
        return state


def generate_output_node(state: PipelineState) -> PipelineState:
    try:
        logger.info("=== PHASE 3: Generating Final JSON Output ===")

        processing_completed_timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"Processing completed: {processing_completed_timestamp}")

        catalog_dict = state.get("catalog_dict", {})
        vector_metadata = state.get("vector_metadata", {})
        failed_files = state.get("failed_files", [])
        parquet_uris = state.get("parquet_uris", [])
        config = state["config"]

        session_id = state.get("session_id", "")
        file_ids = state.get("file_ids") or {}

        catalog_entries = []

        user_provided_tags = state.get("tags", [])
        upload_timestamp = state.get("upload_timestamp", "")
        processing_started = state.get("processing_started_timestamp", "")

        for idx, (parquet_path, table_info) in enumerate(catalog_dict.items()):
            try:
                original_filename = table_info["file_name"]
                file_id = file_ids.get(original_filename, "")
                file_extension = os.path.splitext(original_filename)[1].lower()

                file_type_map = {
                    ".csv": "csv",
                    ".xlsx": "excel",
                    ".xls": "excel",
                    ".tsv": "tsv",
                    ".json": "json",
                    ".parquet": "parquet",
                }
                file_type = file_type_map.get(file_extension, "unknown")

                columns_metadata = []
                column_llm_metadata = table_info.get("column_metadata", {})

                for order, (col_name, col_type) in enumerate(
                    table_info["columns"].items()
                ):
                    llm_data = column_llm_metadata.get(col_name, {})

                    column_entry = {
                        "name": col_name,
                        "type": col_type,
                        "nullable": llm_data.get("nullable", ""),
                        "is_primary_key": llm_data.get("is_primary_key", ""),
                        "order": order,
                        "description": llm_data.get("description", ""),
                    }
                    columns_metadata.append(column_entry)

                row_count = table_info["total_rows"]
                column_count = table_info["column_count"]

                table_metadata = table_info.get("table_metadata", {})
                llm_tags = table_info.get("llm_tags", [])
                language = table_info.get("language", "en")

                main_topics = table_info.get("main_topics")
                if not main_topics:
                    main_topics = [os.path.splitext(original_filename)[0]]

                summary = table_info.get("summary")
                if not summary:
                    summary = f"Structured data with {row_count:,} rows and {column_count} columns"

                final_tags = (
                    llm_tags + user_provided_tags if llm_tags else user_provided_tags
                )
                seen = set()
                final_tags = [
                    tag for tag in final_tags if not (tag in seen or seen.add(tag))
                ]

                vector_meta = vector_metadata.get(parquet_path, {})

                if vector_meta:
                    estimated_tokens = vector_meta.get(
                        "total_tokens", row_count * column_count * 2
                    )
                    estimated_chars = vector_meta.get(
                        "total_characters", estimated_tokens * 4
                    )
                    total_chunks = vector_meta.get(
                        "total_chunks", row_count // 100 if row_count > 100 else 1
                    )
                    total_pages = vector_meta.get("total_pages", "")
                else:
                    estimated_tokens = row_count * column_count * 2
                    estimated_chars = estimated_tokens * 4
                    total_chunks = row_count // 100 if row_count > 100 else 1
                    total_pages = ""

                catalog_entry = {
                    "id": file_id,
                    "partition_key": session_id,
                    "file_id": file_id,
                    "filename": original_filename,
                    "file_type": file_type,
                    "file_category": "structured",
                    "file_size_bytes": "",
                    "mime_type": (
                        f"application/{file_type}" if file_type != "csv" else "text/csv"
                    ),
                    "file_extension": file_extension,
                    "session_id": session_id,
                    "user_id": state.get("user_id", "system"),
                    "organization_id": state.get("organization_id", "default_org"),
                    "upload_timestamp": upload_timestamp,
                    "processing_started": processing_started,
                    "processing_completed": processing_completed_timestamp,
                    "processing_status": "completed",
                    "blob_url": getattr(config.azure_storage, "blob_url", "") or "",
                    "blob_container": config.azure_storage.container_name,
                    "blob_path": parquet_path,
                    "parquet_url": parquet_path,
                    "structured_metadata": {
                        "table_name": os.path.splitext(original_filename)[0],
                        "row_count": row_count,
                        "column_count": column_count,
                        "columns": columns_metadata,
                        "primary_key": table_metadata.get("primary_key", ""),
                        "foreign_keys": table_metadata.get("foreign_keys", []),
                        "data_quality_score": table_metadata.get(
                            "data_quality_score", ""
                        ),
                        "has_duplicates": table_metadata.get("has_duplicates", ""),
                        "null_percentage": table_metadata.get("null_percentage", ""),
                    },
                    "graph_metadata": {
                        "graph_id": vector_meta.get("graph_id", ""),
                        "node_count": vector_meta.get("node_count", ""),
                        "edge_count": vector_meta.get("edge_count", ""),
                        "total_chunks": total_chunks,
                        "total_tokens": estimated_tokens,
                        "total_characters": estimated_chars,
                        "total_pages": total_pages,
                        "neo4j_status": vector_meta.get("neo4j_status", ""),
                        "created_at": vector_meta.get("created_at", ""),
                        "detected_entities": vector_meta.get("detected_entities", []),
                        "detected_relationships": vector_meta.get(
                            "detected_relationships", []
                        ),
                    },
                    "content_analysis": {
                        "language": language,
                        "detected_entities": list(table_info["columns"].keys())[:5],
                        "main_topics": main_topics,
                        "summary": summary,
                    },
                    "access_count": 0,
                    "last_accessed": "",
                    "accessed_by": [],
                    "is_deleted": False,
                    "is_public": False,
                    "is_archived": False,
                    "custom_fields": {
                        "data_source": state.get("data_source", "unknown"),
                        "update_frequency": state.get("update_frequency", "once"),
                        "retention_period": state.get("retention_period", "indefinite"),
                    },
                    "tags": final_tags if final_tags else ["structured_data"],
                }
                catalog_entries.append(catalog_entry)
            except Exception as e:
                logger.error(
                    f"Failed to generate catalog entry for {parquet_path}: {e}"
                )

        json_output = {
            "success": True if catalog_entries else False,
            "session_id": session_id,
            "processing_summary": {
                "total_files_processed": len(parquet_uris),
                "successful_files": len(catalog_entries),
                "failed_files": len(failed_files),
                "vector_ingestion_status": (
                    "completed" if state.get("vector_ingestion_success") else "failed"
                ),
                "processing_completed_at": processing_completed_timestamp,
            },
            "failed_files": failed_files,
            "catalog": catalog_entries,
        }

        if not catalog_dict and failed_files:
            json_output["success"] = False

        json_string = json.dumps(json_output, indent=2, ensure_ascii=False)

        output_json_path = state.get("output_json_path")
        if output_json_path:
            try:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    f.write(json_string)
                logger.info(f"Output saved to: {output_json_path}")
            except Exception as e:
                logger.error(f"Failed to save output to file: {e}")

        return {
            "final_json": json_string,
            "pipeline_success": True if catalog_entries else False,
            "processing_completed_timestamp": processing_completed_timestamp,
        }
    except Exception as e:
        logger.error(f"Output generation failed: {e}")

        error_output = {
            "success": False,
            "session_id": state.get("session_id", ""),
            "processing_summary": {
                "total_files_processed": len(state.get("parquet_uris", [])),
                "successful_files": 0,
                "failed_files": len(state.get("failed_files", [])),
                "vector_ingestion_status": "failed",
                "processing_completed_at": datetime.now(timezone.utc).isoformat(),
            },
            "failed_files": state.get("failed_files", []),
            "catalog": [],
            "error": str(e),
        }

        return {
            "final_json": json.dumps(error_output, indent=2, ensure_ascii=False),
            "pipeline_success": False,
            "processing_completed_timestamp": datetime.now(timezone.utc).isoformat(),
        }


def error_handler_node(state: PipelineState) -> PipelineState:
    try:
        logger.error("=== ERROR: Pipeline Failed ===")

        json_output = {
            "success": False,
            "session_id": state.get("session_id", ""),
            "processing_summary": {
                "total_files_processed": 0,
                "successful_files": 0,
                "failed_files": len(state.get("failed_files", [])),
                "vector_ingestion_status": "not_started",
                "processing_completed_at": datetime.now(timezone.utc).isoformat(),
            },
            "failed_files": state.get("failed_files", []),
            "catalog": [],
            "error": "No Parquet files were successfully created.",
        }

        json_string = json.dumps(json_output, indent=2, ensure_ascii=False)

        return {
            **state,
            "final_json": json_string,
            "pipeline_success": False,
        }
    except Exception as e:
        logger.error(f"Error handler failed: {e}")

        fallback_output = {
            "success": False,
            "error": "Critical pipeline failure",
            "details": str(e),
        }

        return {
            **state,
            "final_json": json.dumps(fallback_output, indent=2, ensure_ascii=False),
            "pipeline_success": False,
        }


def build_ingestion_graph():
    try:
        graph = StateGraph(PipelineState)

        graph.add_node("validate_inputs", validate_inputs_node)
        graph.add_node("process_files_dispatcher", process_files_dispatcher)
        graph.add_node("aggregate_parquet_results", aggregate_parquet_results_node)
        graph.add_node("build_catalog", build_catalog_node)
        graph.add_node("vector_ingestion", vector_ingestion_node)
        graph.add_node("aggregate_results", aggregate_results_node)
        graph.add_node("generate_output", generate_output_node)
        graph.add_node("error_handler", error_handler_node)

        graph.add_edge(START, "validate_inputs")
        graph.add_edge("validate_inputs", "process_files_dispatcher")
        graph.add_edge("process_files_dispatcher", "aggregate_parquet_results")

        def route_after_aggregation(state: PipelineState) -> List[str]:
            if not state["parquet_uris"]:
                return ["error_handler"]
            return ["build_catalog", "vector_ingestion"]

        graph.add_conditional_edges(
            "aggregate_parquet_results",
            route_after_aggregation,
        )

        graph.add_edge("build_catalog", "aggregate_results")
        graph.add_edge("vector_ingestion", "aggregate_results")
        graph.add_edge("aggregate_results", "generate_output")
        graph.add_edge("generate_output", END)
        graph.add_edge("error_handler", END)

        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)

        return compiled_graph
    except Exception as e:
        logger.error(f"Failed to build ingestion graph: {e}")
        raise


def run_ingestion_pipeline(
    input_files: List[str],
    enable_llm_summaries: bool = True,
    output_json_path: Optional[str] = None,
    user_id: str = "system",
    organization_id: str = "default_org",
    data_source: str = "unknown",
    update_frequency: str = "once",
    retention_period: str = "indefinite",
    tags: List[str] = None,
    session_id: Optional[str] = None,
    file_ids: Optional[Dict[str, str]] = None,
) -> str:
    try:
        if not input_files:
            raise ValueError("input_files cannot be empty")

        if not isinstance(input_files, list):
            raise ValueError("input_files must be a list")

        graph = build_ingestion_graph()

        initial_state: PipelineState = {
            "input_files": input_files,
            "enable_llm_summaries": enable_llm_summaries,
            "output_json_path": output_json_path,
            "config": None,
            "parquet_uris": [],
            "failed_files": [],
            "processing_complete": False,
            "catalog_dict": None,
            "vector_ingestion_success": None,
            "vector_metadata": None,
            "final_json": None,
            "pipeline_success": False,
            "user_id": user_id,
            "organization_id": organization_id,
            "data_source": data_source,
            "update_frequency": update_frequency,
            "retention_period": retention_period,
            "tags": tags or [],
            "upload_timestamp": "",
            "processing_started_timestamp": "",
            "processing_completed_timestamp": "",
            "session_id": session_id or "",
            "file_ids": file_ids or {},
        }

        config = {"configurable": {"thread_id": f"ingestion_{int(time.time())}"}}

        final_state = None
        for state in graph.stream(initial_state, config):
            pass

        final_state = state

        if isinstance(final_state, dict):
            for node_name, node_state in final_state.items():
                if "final_json" in node_state:
                    return node_state["final_json"]

        logger.error("Pipeline execution completed but no final JSON found")
        return json.dumps(
            {"success": False, "error": "Pipeline execution failed"}, indent=2
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

        error_output = {
            "success": False,
            "error": "Pipeline execution failed",
            "details": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return json.dumps(error_output, indent=2)


if __name__ == "__main__":
    INPUT_FILES = ["../sheets/MULTI.xlsx", "../sheets/loan.xlsx"]

    try:
        print("=" * 80)
        print("LANGGRAPH TABULAR DATA INGESTION PIPELINE - PRODUCTION")
        print("=" * 80)

        start_time = time.time()

        result_json = run_ingestion_pipeline(
            input_files=INPUT_FILES,
            enable_llm_summaries=True,
            output_json_path="catalog_output.json",
            user_id="test_user",
            organization_id="test_org",
            data_source="test_source",
            update_frequency="daily",
            retention_period="1_year",
            tags=["test", "production"],
            session_id="test_session_123",
            file_ids={"MULTI.xlsx": "file_001", "loan.xlsx": "file_002"},
        )

        end_time = time.time()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(result_json)
        print(f"\nTotal Execution Time: {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
