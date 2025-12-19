"""
LangGraph-Based Tabular Data Ingestion Pipeline

Graph Flow:
    START → validate_inputs → process_files_dispatcher → aggregate_parquet_results
    → [build_catalog + vector_ingestion] (parallel) → aggregate_results
    → generate_output → END
"""

import json
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

try:
    from chroma_ingestion_util_tester import ingest_to_vector_db
except ImportError:

    def ingest_to_vector_db(file_paths: List[str], collection_prefix=None) -> bool:
        print(
            f"[PLACEHOLDER] Vector ingestion for {len(file_paths)} files would run here."
        )
        return True


# ============================================================================
# STATE DEFINITION
# ============================================================================


class PipelineState(TypedDict):
    """Global state shared across all nodes in the graph"""

    input_files: List[str]
    enable_llm_summaries: bool
    output_json_path: Optional[str]
    config: Any
    parquet_uris: List[str]
    failed_files: List[str]
    processing_complete: bool
    catalog_dict: Optional[Dict[str, Any]]
    vector_ingestion_success: Optional[bool]
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be lowercase and snake_case."""
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


def setup_duckdb_azure_connection(config: Any) -> duckdb.DuckDBPyConnection:
    """Initializes and returns a DuckDB connection with Azure credentials."""
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")
        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")
        return conn
    except Exception as e:
        conn.close()
        raise Exception(f"Failed to set up DuckDB Azure connection: {e}")


def upload_file_to_azure(local_path: str, remote_file_name: str, config: Any) -> str:
    """Uploads a local file to Azure Blob Storage and returns the Azure URI."""
    client = BlobServiceClient.from_connection_string(
        config.azure_storage.connection_string
    )
    container = client.get_container_client(config.azure_storage.container_name)
    blob_path = (
        f"{config.azure_storage.parquet_output_dir.rstrip('/')}/{remote_file_name}"
    )
    with open(local_path, "rb") as f:
        container.upload_blob(name=blob_path, data=f, overwrite=True)
    return f"azure://{config.azure_storage.account_name}.blob.core.windows.net/{container.container_name}/{blob_path}"


def download_file_from_url(url: str, local_path: str):
    """Download file from URL to local path"""
    print(f"[INFO] Downloading from URL: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Downloaded to: {local_path}")


def detect_file_format(file_path: str) -> str:
    """Detect file format from extension"""
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


# ============================================================================
# CONVERSION FUNCTIONS (with transient DuckDB connections)
# ============================================================================


def convert_csv_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Convert CSV to Parquet and upload."""
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")

    conn = None
    try:
        df = pd.read_csv(input_path)
        df = clean_column_names(df)

        conn = setup_duckdb_azure_connection(config)
        conn.register("temp_df", df)
        conn.execute(
            f"COPY temp_df TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
        )
        conn.unregister("temp_df")
        del df

        uri = upload_file_to_azure(parquet_file, f"{base_name}.parquet", config)
        os.remove(parquet_file)
        return [uri]
    except Exception as e:
        print(f"[ERROR] Failed to convert CSV: {e}")
        return []
    finally:
        if conn:
            conn.close()


def convert_tsv_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Convert TSV to Parquet and upload."""
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")

    conn = None
    try:
        df = pd.read_csv(input_path, sep="\t")
        df = clean_column_names(df)

        conn = setup_duckdb_azure_connection(config)
        conn.register("temp_df", df)
        conn.execute(
            f"COPY temp_df TO '{parquet_file}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
        )
        conn.unregister("temp_df")
        del df

        uri = upload_file_to_azure(parquet_file, f"{base_name}.parquet", config)
        os.remove(parquet_file)
        return [uri]
    except Exception as e:
        print(f"[ERROR] Failed to convert TSV: {e}")
        return []
    finally:
        if conn:
            conn.close()


def convert_json_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Convert JSON to Parquet and upload."""
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
        os.remove(parquet_file)
        return [uri]
    except Exception as e:
        print(f"[ERROR] Failed to convert JSON: {e}")
        return []
    finally:
        if conn:
            conn.close()


def upload_existing_parquet(input_path: str, config: Any) -> List[str]:
    """Upload existing local Parquet file to Azure."""
    base_name = os.path.basename(input_path)
    try:
        uri = upload_file_to_azure(input_path, base_name, config)
        return [uri]
    except Exception as e:
        print(f"[ERROR] Failed to upload Parquet: {e}")
        return []


def convert_excel_to_parquet(
    input_path: str, output_dir: str, config: Any
) -> List[str]:
    """Converts Excel file (all sheets) to separate Parquet files and uploads them."""
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)
    uris = []

    conn = None
    try:
        sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Failed to read Excel {input_path}: {e}")
        return uris

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )

    try:
        conn = setup_duckdb_azure_connection(config)
    except Exception as e:
        print(f"[ERROR] Excel conversion failed to setup DuckDB: {e}")
        return []

    for sheet_name, df in sheets.items():
        if df.empty or df.shape[1] == 0:
            continue

        df = clean_column_names(df)
        safe_name = re.sub(r"[^\w]", "_", sheet_name)
        parquet_file = os.path.join(temp_dir, f"{base_name}_{safe_name}.parquet")

        try:
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
            os.remove(parquet_file)
        except Exception as e:
            print(f"[ERROR] Failed sheet {sheet_name}: {e}")

    if conn:
        conn.close()
    return uris


def convert_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Routes file conversion based on format."""
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
        print(f"[ERROR] Unsupported format: {file_format}")
        return []


# ============================================================================
# CATALOG GENERATION FUNCTIONS
# ============================================================================


def generate_column_summaries(
    columns: Dict[str, str],
    sample_data: pd.DataFrame,
    llm_client: Any,
    deployment_name: str,
    filename: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[str], str, List[str], str]:
    """Calls the LLM to generate comprehensive column metadata, table-level analysis, intelligent tags, language detection, main topics, and summary."""
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

    try:
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
        print(f"[ERROR] LLM call failed: {e}")
        return {}, {}, [], "en", [], ""


def process_single_parquet_for_catalog(
    path: str, llm_client: Any, config: Any, enable_llm_summaries: bool
) -> Tuple[Optional[str], Union[Dict[str, Any], None]]:
    """Worker function to process a single Parquet URI for the catalog."""
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

        print(f"[SUCCESS] Cataloged {os.path.basename(path)} ({row_count:,} rows).")
        return path, table_info

    except Exception as e:
        print(f"[ERROR] Failed to catalog {os.path.basename(path)}: {e}")
        return path, None
    finally:
        if conn:
            conn.close()


# ============================================================================
# LANGGRAPH NODES
# ============================================================================


def validate_inputs_node(state: PipelineState) -> PipelineState:
    """Node 1: Validate inputs and initialize configuration."""
    print("\n=== PHASE 0: Validating Inputs ===")

    upload_timestamp = datetime.now(timezone.utc).isoformat()

    if not state.get("input_files"):
        raise ValueError("No input files provided")

    if "config" not in state or state["config"] is None:
        config = get_config(VectorDBType.CHROMADB)
        state["config"] = config

    temp_dir = "/tmp/downloaded_files"
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[INFO] Validated {len(state['input_files'])} input files")
    print(f"[INFO] Upload timestamp: {upload_timestamp}")

    return {
        **state,
        "processing_complete": False,
        "pipeline_success": False,
        "upload_timestamp": upload_timestamp,
    }


def process_and_convert_file_wrapper(
    file_path: str, temp_dir: str, config: Any
) -> Tuple[List[str], Optional[str]]:
    """Wrapper for file processing that returns (uris, failed_path)"""
    local_path = None
    try:
        print(f"[INFO] Processing: {os.path.basename(file_path)}")

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
            os.remove(local_path)

        if uris:
            print(
                f"[SUCCESS] Processed {os.path.basename(file_path)} -> {len(uris)} Parquet file(s)"
            )
            return uris, None
        else:
            return [], file_path

    except Exception as e:
        print(f"[ERROR] Processing failed for {os.path.basename(file_path)}: {e}")
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        return [], file_path


def process_files_dispatcher(state: PipelineState) -> PipelineState:
    """Node 2: Process all files in parallel using ThreadPoolExecutor."""
    print(
        f"\n=== PHASE 1: Processing {len(state['input_files'])} Files in Parallel ==="
    )

    processing_started_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[INFO] Processing started: {processing_started_timestamp}")

    temp_dir = "/tmp/downloaded_files"
    config = state["config"]

    all_parquet_uris = []
    all_failed_files = []

    MAX_WORKERS = 4
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
                print(f"[ERROR] Failed {os.path.basename(file_path)}: {e}")
                all_failed_files.append(file_path)

    return {
        **state,
        "parquet_uris": all_parquet_uris,
        "failed_files": all_failed_files,
        "processing_started_timestamp": processing_started_timestamp,
    }


def aggregate_parquet_results_node(state: PipelineState) -> PipelineState:
    """Node 4: Aggregate results from all parallel file processors."""
    print(f"\n=== PHASE 1 Complete: Aggregating Results ===")
    print(f"[INFO] Total Parquet files created: {len(state['parquet_uris'])}")
    print(f"[INFO] Failed files: {len(state['failed_files'])}")

    return {
        **state,
        "processing_complete": True,
    }


def check_processing_success(state: PipelineState) -> Literal["continue", "error"]:
    """Conditional edge: Check if file processing succeeded."""
    if not state["parquet_uris"]:
        print("[ERROR] No Parquet files successfully created. Terminating.")
        return "error"
    return "continue"


def build_catalog_node(state: PipelineState) -> PipelineState:
    """Node 5a: Build LLM-enhanced catalog (runs in parallel with vector ingestion)."""
    print(
        f"\n=== PHASE 2a: Building Catalog for {len(state['parquet_uris'])} Files ==="
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
            print("[INFO] LLM client initialized for summarization.")
        except Exception as e:
            print(
                f"[WARNING] Failed to initialize LLM client: {e}. Disabling summaries."
            )
            enable_llm_summaries = False

    MAX_CATALOG_WORKERS = 4
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
                print(
                    f"[ERROR] Failed to retrieve catalog for {os.path.basename(path)}: {e}"
                )

    print(f"[SUCCESS] Catalog built for {len(catalog_dict)} tables.")
    return {"catalog_dict": catalog_dict}


def vector_ingestion_node(state: PipelineState) -> PipelineState:
    """Node 5b: Vector database ingestion (Nouman's embeddings pipeline)."""
    print(
        f"\n=== PHASE 2b: Running Vector Ingestion for {len(state['parquet_uris'])} Files ==="
    )

    parquet_uris = state["parquet_uris"]

    try:
        success = ingest_to_vector_db(parquet_uris)
        if success:
            print("[SUCCESS] Vector ingestion completed successfully.")
        else:
            print("[WARNING] Vector ingestion reported failure.")
        return {"vector_ingestion_success": success}
    except Exception as e:
        print(f"[ERROR] Vector ingestion failed: {e}")
        return {"vector_ingestion_success": False}


def aggregate_results_node(state: PipelineState) -> PipelineState:
    """Node 6: Aggregate results from parallel branches (catalog + vector ingestion)."""
    print(f"\n=== PHASE 2 Complete: Aggregating Parallel Results ===")

    catalog_success = state.get("catalog_dict") is not None
    vector_success = state.get("vector_ingestion_success", False)

    print(f"[INFO] Catalog building: {'✓ Success' if catalog_success else '✗ Failed'}")
    print(f"[INFO] Vector ingestion: {'✓ Success' if vector_success else '✗ Failed'}")

    return state


def generate_output_node(state: PipelineState) -> PipelineState:
    """Node 7: Generate final JSON output in comprehensive catalog format."""
    print(f"\n=== PHASE 3: Generating Final JSON Output ===")

    processing_completed_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[INFO] Processing completed: {processing_completed_timestamp}")

    catalog_dict = state.get("catalog_dict", {})
    failed_files = state.get("failed_files", [])
    parquet_uris = state.get("parquet_uris", [])
    config = state["config"]

    # Use frontend-provided session_id or generate one
    session_id = state.get("session_id")

    # Get frontend-provided file_ids mapping (filename -> file_id)
    file_ids = state.get("file_ids") or {}

    catalog_entries = []

    user_provided_tags = state.get("tags", [])
    upload_timestamp = state.get("upload_timestamp", "PLACEHOLDER")
    processing_started = state.get("processing_started_timestamp", "PLACEHOLDER")

    for idx, (parquet_path, table_info) in enumerate(catalog_dict.items()):
        original_filename = table_info["file_name"]

        # Use frontend-provided file_id if available, otherwise generate one
        file_id = file_ids.get(original_filename) or "PLACEHOLDER"

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

        for order, (col_name, col_type) in enumerate(table_info["columns"].items()):
            llm_data = column_llm_metadata.get(col_name, {})

            column_entry = {
                "name": col_name,
                "type": col_type,
                "nullable": llm_data.get("nullable", "PLACEHOLDER"),
                "is_primary_key": llm_data.get("is_primary_key", "PLACEHOLDER"),
                "order": order,
                "description": llm_data.get("description", "PLACEHOLDER"),
            }
            columns_metadata.append(column_entry)

        row_count = table_info["total_rows"]
        column_count = table_info["column_count"]
        estimated_tokens = row_count * column_count * 2
        estimated_chars = estimated_tokens * 4

        table_metadata = table_info.get("table_metadata", {})
        llm_tags = table_info.get("llm_tags", [])
        language = table_info.get("language", "Failed")

        # Get LLM-generated main_topics and summary with fallbacks
        main_topics = table_info.get("main_topics")
        if not main_topics:
            main_topics = [os.path.splitext(original_filename)[0]]

        summary = table_info.get("summary")
        if not summary:
            summary = (
                f"Structured data with {row_count:,} rows and {column_count} columns"
            )

        final_tags = llm_tags + user_provided_tags if llm_tags else user_provided_tags
        seen = set()
        final_tags = [tag for tag in final_tags if not (tag in seen or seen.add(tag))]

        catalog_entry = {
            "id": file_id,
            "partition_key": session_id,
            "file_id": file_id,
            "filename": original_filename,
            "file_type": file_type,
            "file_category": "structured",
            "file_size_bytes": "PLACEHOLDER",
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
            "total_chunks": row_count // 100 if row_count > 100 else 1,
            "total_tokens": estimated_tokens,
            "total_characters": estimated_chars,
            "total_pages": "PLACEHOLDER",
            "blob_url": "PLACEHOLDER",
            "blob_container": config.azure_storage.container_name,
            "blob_path": parquet_path,
            "parquet_url": parquet_path,
            "structured_metadata": {
                "table_name": os.path.splitext(original_filename)[0],
                "row_count": row_count,
                "column_count": column_count,
                "columns": columns_metadata,
                "primary_key": table_metadata.get("primary_key", "PLACEHOLDER"),
                "foreign_keys": table_metadata.get("foreign_keys", []),
                "data_quality_score": table_metadata.get(
                    "data_quality_score", "PLACEHOLDER"
                ),
                "has_duplicates": table_metadata.get("has_duplicates", "PLACEHOLDER"),
                "null_percentage": table_metadata.get("null_percentage", "PLACEHOLDER"),
            },
            "graph_metadata": {
                "graph_id": "PLACEHOLDER",
                "node_count": "PLACEHOLDER",
                "edge_count": "PLACEHOLDER",
                "neo4j_status": "PLACEHOLDER",
                "created_at": "PLACEHOLDER",
            },
            "content_analysis": {
                "language": language,
                "detected_entities": list(table_info["columns"].keys())[:5],
                "main_topics": main_topics,
                "summary": summary,
            },
            "access_count": 0,
            "last_accessed": "PLACEHOLDER",
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

    json_output = {
        "success": True,
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
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
        print(f"[INFO] Output saved to: {output_json_path}")

    return {
        "final_json": json_string,
        "pipeline_success": True,
        "processing_completed_timestamp": processing_completed_timestamp,
    }


def error_handler_node(state: PipelineState) -> PipelineState:
    """Error handler node for when file processing fails completely."""
    print("\n=== ERROR: Pipeline Failed ===")

    json_output = {
        "success": False,
        "total_files": 0,
        "failed_files": state.get("failed_files", []),
        "tables": [],
        "error": "No Parquet files were successfully created.",
    }

    json_string = json.dumps(json_output, indent=2, ensure_ascii=False)

    return {
        **state,
        "final_json": json_string,
        "pipeline_success": False,
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def build_ingestion_graph():
    """Constructs the LangGraph state machine for the ingestion pipeline."""
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================


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
    """
    Main entry point for running the LangGraph-based ingestion pipeline.

    Args:
        input_files: List of file paths or URLs to ingest
        enable_llm_summaries: Whether to generate LLM summaries for columns
        output_json_path: Optional path to save output JSON
        user_id: User identifier for tracking
        organization_id: Organization identifier
        data_source: Source of the data (e.g., "CRM", "ERP")
        update_frequency: How often data is updated (e.g., "daily", "weekly")
        retention_period: How long to retain data (e.g., "7_years", "indefinite")
        tags: List of tags for categorization
        session_id: Optional session ID from frontend (auto-generated if not provided)
        file_ids: Optional dict mapping filename to file_id from frontend

    Returns:
        JSON string with pipeline results
    """
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
        "session_id": session_id,
        "file_ids": file_ids,
    }

    config = {"configurable": {"thread_id": "ingestion_run_1"}}

    final_state = None
    for state in graph.stream(initial_state, config):
        pass

    final_state = state

    if isinstance(final_state, dict):
        for node_name, node_state in final_state.items():
            if "final_json" in node_state:
                return node_state["final_json"]

    return json.dumps({"error": "Pipeline execution failed"}, indent=2)


if __name__ == "__main__":
    INPUT_FILES = [
        "../sheets/MULTI.xlsx",
        "../sheets/loan.xlsx",
        "https://gist.githubusercontent.com/dsternlicht/74020ebfdd91a686d71e785a79b318d4/raw/d3104389ba98a8605f8e641871b9ce71eff73f7e/chartsninja-data-1.csv",
    ]

    try:
        print("=" * 80)
        print("LANGGRAPH TABULAR DATA INGESTION PIPELINE")
        print("=" * 80)

        start_time = time.time()

        result_json = run_ingestion_pipeline(
            input_files=INPUT_FILES,
            enable_llm_summaries=True,
            output_json_path="catalog_output.json",
            user_id="PLACEHOLDER",
            organization_id="PLACEHOLDER",
            data_source="PLACEHOLDER",
            update_frequency="PLACEHOLDER",
            retention_period="PLACEHOLDER",
            tags=[],
            session_id="PLACEHOLDER",
            file_ids={
                "MULTI.xlsx": "PLACEHOLDER",
                "loan.xlsx": "PLACEHOLDER",
                "chartsninja-data-1.csv": "PLACEHOLDER",
            },
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
        print(f"\n[FATAL] Unhandled exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
