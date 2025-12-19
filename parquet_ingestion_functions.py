"""
Tabular Data Ingestion Pipeline with LLM-Enhanced Catalog Generation

DESIGN GOAL: MAXIMUM PARALLELISM ACROSS ALL LAYERS
This pipeline is optimized for concurrent execution across four layers:

1. L4 (Phase 1): Concurrent conversion and upload of all input files.
2. L3 (Phase 2): Concurrent cataloging of *each* Parquet file AND concurrent vector
   ingestion of *all* Parquet files.
3. L2 (Internal Vector Ingestion): Concurrent processing of all Parquet URIs within the
   main Vector Ingestion job.
4. L1 (Internal Vector Embedding): Concurrent embedding of text batches for maximum
   Azure OpenAI throughput.

Processing Flow:
1. [Parallel] Phase 1: Download (if needed), Convert to Parquet, and Upload to Azure Blob.
2. [Concurrent] Phase 2: Build LLM-enhanced Catalog AND run Vector Ingestion concurrently.
   * Both jobs now utilize internal parallelism to process files simultaneously.
3. [Sequential] Phase 3: Collect results and generate final JSON output.
"""

import atexit
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import duckdb
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient

from config import Config, VectorDBType, get_config

# --- Core Dependency Import: Vector Ingestion ---
try:
    from chroma_ingestion_util import ingest_to_vector_db
except ImportError:
    # Placeholder for developer who is working on it
    def ingest_to_vector_db(file_paths: List[str], collection_prefix=None) -> bool:
        print(
            f"[PLACEHOLDER] Vector ingestion for {len(file_paths)} files would run here."
        )
        return True


# Global/Persistent Connection is still used for sequential operations, but avoided in parallel workers.
PERSISTENT_DUCKDB_CONN: Union[duckdb.DuckDBPyConnection, None] = None
CONN_LOCK = threading.Lock()


def close_persistent_duckdb_connection():
    global PERSISTENT_DUCKDB_CONN
    if PERSISTENT_DUCKDB_CONN:
        PERSISTENT_DUCKDB_CONN.close()
        PERSISTENT_DUCKDB_CONN = None


atexit.register(close_persistent_duckdb_connection)


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
    """
    Initializes and returns a DuckDB connection with Azure credentials.
    This function can be called to get a local, transient connection for workers,
    or used to set up the global persistent connection.
    """
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


def get_global_duckdb_connection(config: Any) -> duckdb.DuckDBPyConnection:
    """Gets or sets the persistent global connection (used primarily for setup/sequential)."""
    global PERSISTENT_DUCKDB_CONN
    if PERSISTENT_DUCKDB_CONN:
        return PERSISTENT_DUCKDB_CONN

    with CONN_LOCK:
        if PERSISTENT_DUCKDB_CONN:
            return PERSISTENT_DUCKDB_CONN
        PERSISTENT_DUCKDB_CONN = setup_duckdb_azure_connection(config)
        return PERSISTENT_DUCKDB_CONN


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
    if ext in [".xlsx", ".xls"]:
        return "excel"
    elif ext == ".csv":
        return "csv"
    elif ext == ".tsv":
        return "tsv"
    elif ext == ".json":
        return "json"
    elif ext == ".parquet":
        return "parquet"
    else:
        return "unknown"


# --- Conversion Functions (Accepts URL or Local Path) ---


def convert_csv_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Convert CSV (local or remote URL) to Parquet and upload."""
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )
    parquet_file = os.path.join(temp_dir, f"{base_name}.parquet")

    conn = None
    try:
        # pandas.read_csv handles remote HTTP/S paths directly
        df = pd.read_csv(input_path)
        df = clean_column_names(df)

        # CRITICAL FIX: Use local, transient connection
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
    """Convert TSV (local or remote URL) to Parquet and upload."""
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

        # CRITICAL FIX: Use local, transient connection
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
    """Convert JSON (local or remote URL) to Parquet and upload."""
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

        # CRITICAL FIX: Use local, transient connection
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


def convert_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """Routes file conversion based on format."""
    file_format = detect_file_format(input_path)

    if file_format == "excel":
        return convert_excel_to_parquet(input_path, output_dir, config)
    elif file_format == "csv":
        return convert_csv_to_parquet(input_path, output_dir, config)
    elif file_format == "tsv":
        return convert_tsv_to_parquet(input_path, output_dir, config)
    elif file_format == "json":
        return convert_json_to_parquet(input_path, output_dir, config)
    elif file_format == "parquet":
        return upload_existing_parquet(input_path, config)
    else:
        print(f"[ERROR] Unsupported format: {file_format}")
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
        # pandas.read_excel requires a local file or buffer, hence the earlier download step for URLs
        sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Failed to read Excel {input_path}: {e}")
        return uris

    base_name = re.sub(
        r"[^\w]", "_", os.path.splitext(os.path.basename(input_path.split("/")[-1]))[0]
    )

    # CRITICAL FIX: Use local, transient connection for the multi-sheet process
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


# --- PHASE 1 Parallel Worker (L4 Parallelism) ---
def process_and_convert_file(
    input_file_path: str, temp_dir: str, config: Any
) -> Tuple[List[str], Optional[str]]:
    """
    Worker function executed in parallel for each input file.
    """
    local_path = None
    original_path = input_file_path

    # NOTE: This worker is clean. It calls the conversion functions which now handle
    # their own transient DuckDB connections internally.
    try:
        print(
            f"[INFO] Starting parallel processing for: {os.path.basename(original_path)}"
        )

        path_to_convert = original_path
        is_url = original_path.startswith("http://") or original_path.startswith(
            "https://"
        )

        if is_url:
            ext = os.path.splitext(urlparse(input_file_path).path)[1].lower()

            if ext in [".xlsx", ".xls"]:
                filename = os.path.basename(urlparse(input_file_path).path)
                local_path = os.path.join(temp_dir, filename)
                download_file_from_url(input_file_path, local_path)
                path_to_convert = local_path
        else:
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"Local file not found: {input_file_path}")

        uris = convert_to_parquet(
            path_to_convert, config.azure_storage.parquet_output_dir, config
        )

        if local_path and os.path.exists(local_path):
            os.remove(local_path)

        return uris, None

    except Exception as e:
        print(f"[ERROR] Full processing failed for {original_path}: {e}")
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        return [], original_path


def generate_column_summaries(
    columns: Dict[str, str],
    sample_data: pd.DataFrame,
    llm_client: Any,
    deployment_name: str,
) -> Dict[str, str]:
    """Calls the LLM to generate descriptive summaries for all columns based on sample data."""
    if len(sample_data) > 5:
        sample_text = sample_data.head(5).to_markdown(index=False, tablefmt="plain")
    else:
        sample_text = sample_data.to_markdown(index=False, tablefmt="plain")

    if len(sample_text) > 2000:
        sample_text = sample_text[:2000] + "\n... (truncated)"

    column_info = "\n".join([f"- {col}: {dtype}" for col, dtype in columns.items()])

    prompt = f"""Analyze ALL columns and provide ONE brief description per column (max 70 chars).
Describe what EACH column represents, NOT min/max/analytical values.

You MUST provide a summary for EVERY column listed below.

Columns ({len(columns)} total):
{column_info}

Sample data (5 rows):
{sample_text}

Respond with JSON containing ALL {len(columns)} columns:
{{"column": "brief description", ...}}

IMPORTANT: Include ALL {len(columns)} columns in your response."""

    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        summaries = json.loads(response.choices[0].message.content.strip())
        return summaries

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return {}


# --- PHASE 2 Parallel Worker (L3 Parallelism) ---


def process_single_parquet_for_catalog(
    path: str, llm_client: Any, config: Any, enable_llm_summaries: bool
) -> Tuple[Optional[str], Union[Dict[str, Any], None]]:
    """
    Worker function to process a single Parquet URI for the catalog.
    CRITICAL FIX: Creates and closes its own DuckDB connection.
    Returns: (parquet_path, table_catalog_dict)
    """
    conn = None
    try:
        # CRITICAL FIX: Use local, transient connection
        conn = setup_duckdb_azure_connection(config)

        # Get schema
        schema_df = conn.execute(
            f"DESCRIBE (SELECT * FROM read_parquet('{path}'))"
        ).fetchdf()
        col_info = {
            row["column_name"]: row["column_type"] for _, row in schema_df.iterrows()
        }
        del schema_df

        # Get row count
        row_count = conn.execute(
            f"SELECT COUNT(*) as count FROM read_parquet('{path}')"
        ).fetchone()[0]

        column_summaries = {}
        if enable_llm_summaries and llm_client:
            # LLM summarization logic
            sample_df = conn.execute(
                f"SELECT * FROM read_parquet('{path}') LIMIT 5"
            ).fetchdf()

            column_summaries = generate_column_summaries(
                columns=col_info,
                sample_data=sample_df,
                llm_client=llm_client,
                deployment_name=config.azure_openai.llm_deployment_name,
            )
            del sample_df

        # Store in catalog structure
        table_info = {
            "parquet_path": path,
            "columns": col_info,
            "column_count": len(col_info),
            "total_rows": row_count,
            "file_name": os.path.basename(path),
        }

        if column_summaries:
            table_info["column_summaries"] = column_summaries

        print(
            f"[SUCCESS] Cataloged {os.path.basename(path)} ({row_count:,} rows) in parallel."
        )
        return path, table_info

    except Exception as e:
        print(f"[ERROR] Failed to catalog {os.path.basename(path)} in parallel: {e}")
        return path, None
    finally:
        if conn:
            conn.close()


def build_global_catalog(
    parquet_paths: List[str], config: Any, enable_llm_summaries: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Generates a catalog for all Parquet files concurrently using a ThreadPoolExecutor.
    """
    from openai import AzureOpenAI

    catalog_dict = {}

    llm_client = None
    if enable_llm_summaries:
        try:
            llm_client = AzureOpenAI(
                api_key=config.azure_openai.llm_api_key,
                azure_endpoint=config.azure_openai.llm_endpoint,
                api_version=config.azure_openai.llm_api_version,
            )
            print("[INFO] LLM client initialized successfully for summarization.")
        except Exception as e:
            print(
                f"[WARNING] Failed to initialize LLM client: {e}. Disabling LLM summaries."
            )
            enable_llm_summaries = False

    # Note: DuckDB connections are now managed inside the worker

    # Internal worker pool for cataloging files (L3 Parallelism)
    MAX_CATALOG_WORKERS = 4

    print(
        f"\n[INFO] Starting concurrent cataloging for {len(parquet_paths)} URIs using {MAX_CATALOG_WORKERS} workers."
    )

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
                    f"[ERROR] Failed to retrieve result for {os.path.basename(path)}: {e}"
                )

    print(
        f"\n[SUCCESS] Concurrent cataloging complete. Cataloged {len(catalog_dict)} tables."
    )
    return catalog_dict


def run_ingestion(
    input_files: List[str],
    enable_llm_summaries: bool = True,
    output_json_path: str = None,
) -> str:
    """The main orchestration function for the ingestion pipeline."""
    config = get_config(VectorDBType.CHROMADB)

    try:
        # Get the global connection once, mainly to register atexit cleanup
        get_global_duckdb_connection(config)
        atexit.register(close_persistent_duckdb_connection)
    except Exception as e:
        print(f"[FATAL] Failed to setup DuckDB: {e}")
        sys.exit(1)

    temp_dir = "/tmp/downloaded_files"
    os.makedirs(temp_dir, exist_ok=True)

    # --- PHASE 1: Full File Processing (Download, Convert, Upload) in Parallel ---
    print(
        f"\n--- PHASE 1: Processing {len(input_files)} Files (Download, Convert, Upload) in Parallel ---"
    )

    all_parquet_uris = []
    failed_files = []

    # MAX_CONVERSION_WORKERS drives L4 parallelism
    MAX_CONVERSION_WORKERS = 4
    with ThreadPoolExecutor(max_workers=MAX_CONVERSION_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                process_and_convert_file,
                file_path,
                temp_dir,
                config,
            ): file_path
            for file_path in input_files
        }

        for future in as_completed(future_to_file):
            original_path = future_to_file[future]
            try:
                uris, failed_path = future.result()
                if uris:
                    all_parquet_uris.extend(uris)
                    print(
                        f"[SUCCESS] Processed {os.path.basename(original_path)} -> {len(uris)} Parquet URI(s)"
                    )
                elif failed_path:
                    failed_files.append(failed_path)
            except Exception as e:
                print(
                    f"[ERROR] Unexpected failure during processing of {os.path.basename(original_path)}: {e}"
                )
                failed_files.append(original_path)

    if not all_parquet_uris:
        print("[ERROR] No Parquet files successfully created.")
        if not failed_files:
            sys.exit(1)

    print(f"[SUCCESS] Converted and uploaded {len(all_parquet_uris)} Parquet files.")

    # --- PHASE 2: Parallel Catalog Building & Vector Ingestion ---
    print(f"\n--- PHASE 2: Starting Parallel Catalog and Vector Ingestion ---")

    # MAX_POST_PROCESSING_WORKERS drives L3 parallelism (between the two major jobs)
    MAX_POST_PROCESSING_WORKERS = 5
    catalog_dict = {}

    with ThreadPoolExecutor(max_workers=MAX_POST_PROCESSING_WORKERS) as executor:

        # 1. Submit Catalog Building (LLM-Bound job runs concurrently, now internally parallel)
        catalog_future = executor.submit(
            build_global_catalog, all_parquet_uris, config, enable_llm_summaries
        )

        # 2. Submit Vector Ingestion (I/O-Bound job runs concurrently, handles L2/L1 parallelism internally)
        vector_ingestion_future = executor.submit(ingest_to_vector_db, all_parquet_uris)

        # Wait for Catalog to finish
        try:
            catalog_dict = catalog_future.result()
            print(f"[SUCCESS] Global catalog built for {len(catalog_dict)} tables.")
        except Exception as e:
            print(f"[FATAL ERROR] Catalog building failed: {e}")
            catalog_dict = {}

        # Wait for the single Vector Ingestion task to finish (which wrapped all file ingestions)
        vector_ingestion_success = False
        try:
            vector_ingestion_success = vector_ingestion_future.result()
        except Exception as e:
            print(f"[ERROR] Vector ingestion job failed: {e}")
            vector_ingestion_success = False

        if vector_ingestion_success:
            print("[SUCCESS] All vector ingestion tasks completed.")
        else:
            print("[WARNING] One or more vector ingestion tasks reported failure.")

    # --- PHASE 3: Final JSON Output Generation ---
    print(f"\n--- PHASE 3: Generating Final JSON Output ---")

    json_output = {
        "success": True,
        "total_files": len(all_parquet_uris),
        "failed_files": failed_files,
        "tables": [],
    }

    # Structure final output from catalog_dict
    for table_info in catalog_dict.values():
        table_block = {
            "file_path": table_info["parquet_path"],
            "file_name": table_info["file_name"],
            "column_count": table_info["column_count"],
            "total_rows": table_info["total_rows"],
            "columns": [],
        }

        for col_name, col_type in table_info["columns"].items():
            column_entry = {"name": col_name, "type": col_type}
            if (
                "column_summaries" in table_info
                and col_name in table_info["column_summaries"]
            ):
                column_entry["summary"] = table_info["column_summaries"][col_name]
            table_block["columns"].append(column_entry)

        json_output["tables"].append(table_block)

    if not catalog_dict and failed_files:
        json_output["success"] = False

    json_string = json.dumps(json_output, indent=2, ensure_ascii=False)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    return json_string


if __name__ == "__main__":
    # Ensure this list is used consistently across runs
    INPUT_FILES = [
        "../sheets/MULTI.xlsx",
        "../sheets/loan.xlsx",
        "https://gist.githubusercontent.com/dsternlicht/74020ebfdd91a686d71e785a79b318d4/raw/d3104389ba98a8605f8e641871b9ce71eff73f7e/chartsninja-data-1.csv",
    ]
    try:
        result_json = run_ingestion(
            input_files=INPUT_FILES,
            enable_llm_summaries=True,
            output_json_path="catalog_output.json",
        )
        print(result_json)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[FATAL] An unhandled exception occurred: {e}")
        sys.exit(1)
