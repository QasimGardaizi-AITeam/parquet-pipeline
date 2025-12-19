"""
Excel to Parquet Ingestion Pipeline with LLM-Enhanced Catalog Generation

Purpose:
- Converts Excel files to compressed Parquet format (ZSTD, 100K row chunks)
- Uploads Parquet files to Azure Blob Storage
- Generates LLM-enhanced catalog with column descriptions
- Returns JSON catalog for downstream query processing

Output Format:
JSON with structure:
{
  "success": bool,
  "total_files": int,
  "failed_files": [],
  "tables": [
    {
      "file_path": str (azure:// URI),
      "file_name": str,
      "column_count": int,
      "columns": [
        {"name": str, "type": str, "summary": str},
        ...
      ]
    }
  ]
}
"""

import atexit
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import duckdb
import pandas as pd
from azure.storage.blob import BlobServiceClient

from config import Config, VectorDBType, get_config

try:
    from chroma_ingestion_util import ingest_to_chroma_db as ingest_to_vector_db
except ImportError:

    def ingest_to_vector_db(file_path, sheet_name=None, collection_prefix=None):
        return True


PERSISTENT_DUCKDB_CONN = None
CONN_LOCK = threading.Lock()


def close_persistent_duckdb_connection():
    global PERSISTENT_DUCKDB_CONN
    if PERSISTENT_DUCKDB_CONN:
        PERSISTENT_DUCKDB_CONN.close()
        PERSISTENT_DUCKDB_CONN = None


atexit.register(close_persistent_duckdb_connection)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
    global PERSISTENT_DUCKDB_CONN
    if PERSISTENT_DUCKDB_CONN:
        return PERSISTENT_DUCKDB_CONN

    with CONN_LOCK:
        if PERSISTENT_DUCKDB_CONN:
            return PERSISTENT_DUCKDB_CONN
        conn = duckdb.connect()
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")
        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")
        PERSISTENT_DUCKDB_CONN = conn
        return conn


def upload_file_to_azure(local_path: str, remote_file_name: str, config: Any) -> str:
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


def convert_excel_to_parquet(
    input_path: str, output_dir: str, config: Any
) -> List[str]:
    temp_dir = "/tmp/parquet_cache"
    os.makedirs(temp_dir, exist_ok=True)
    uris = []

    try:
        sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Failed to read Excel {input_path}: {e}")
        return uris

    base_name = re.sub(r"[^\w]", "_", os.path.splitext(os.path.basename(input_path))[0])

    conn = duckdb.connect()
    conn.execute("INSTALL azure;")
    conn.execute("LOAD azure;")
    escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
    conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")

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

    conn.close()
    return uris


def build_global_catalog(
    parquet_paths: List[str], config: Any, enable_llm_summaries: bool = True
) -> Dict[str, Dict[str, Any]]:
    from openai import AzureOpenAI

    conn = setup_duckdb_azure_connection(config)

    llm_client = None
    if enable_llm_summaries:
        try:
            llm_client = AzureOpenAI(
                api_key=config.azure_openai.llm_api_key,
                azure_endpoint=config.azure_openai.llm_endpoint,
                api_version=config.azure_openai.llm_api_version,
            )
            print("[INFO] LLM client initialized successfully")
        except Exception as e:
            print(f"[WARNING] Failed to initialize LLM client: {e}")
            enable_llm_summaries = False

    catalog_dict = {}

    for idx, path in enumerate(parquet_paths, 1):
        print(
            f"\n[INFO] Processing file {idx}/{len(parquet_paths)}: {os.path.basename(path)}"
        )

        try:
            schema_df = conn.execute(
                f"DESCRIBE (SELECT * FROM read_parquet('{path}'))"
            ).fetchdf()
            col_info = {
                row["column_name"]: row["column_type"]
                for _, row in schema_df.iterrows()
            }
            del schema_df

            print(f"[INFO] Found {len(col_info)} columns")

            column_summaries = {}
            if enable_llm_summaries and llm_client:
                try:
                    print(
                        f"[INFO] Generating summaries for all {len(col_info)} columns..."
                    )

                    sample_df = conn.execute(
                        f"SELECT * FROM read_parquet('{path}') LIMIT 5"
                    ).fetchdf()

                    print(f"[INFO] Sample data shape: {sample_df.shape}")

                    column_summaries = generate_column_summaries(
                        columns=col_info,
                        sample_data=sample_df,
                        llm_client=llm_client,
                        deployment_name=config.azure_openai.llm_deployment_name,
                    )

                    print(f"[SUCCESS] Generated {len(column_summaries)} summaries")

                    if len(column_summaries) < len(col_info):
                        print(
                            f"[WARNING] Only {len(column_summaries)}/{len(col_info)} columns got summaries"
                        )

                    del sample_df

                except Exception as e:
                    print(f"[ERROR] Failed to generate summaries: {e}")
                    import traceback

                    traceback.print_exc()

            catalog_dict[path] = {
                "parquet_path": path,
                "columns": col_info,
                "column_count": len(col_info),
                "file_name": os.path.basename(path),
            }

            if column_summaries:
                catalog_dict[path]["column_summaries"] = column_summaries

        except Exception as e:
            print(f"[ERROR] Failed to catalog {path}: {e}")

    return catalog_dict


def generate_column_summaries(
    columns: Dict[str, str],
    sample_data: pd.DataFrame,
    llm_client: Any,
    deployment_name: str,
) -> Dict[str, str]:

    print(f"[INFO] Preparing data for {len(columns)} columns")

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

Example format:
{{"id": "Unique customer identifiers", "name": "Full customer names", "age": "Customer ages in years"}}

IMPORTANT: Include ALL {len(columns)} columns in your response."""

    print(f"[INFO] Sending request to LLM for {len(columns)} columns")
    print(f"[INFO] Prompt length: {len(prompt)} chars")

    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        summaries = json.loads(response.choices[0].message.content.strip())

        print(f"[INFO] LLM returned {len(summaries)} summaries")

        if len(summaries) < len(columns):
            missing = set(columns.keys()) - set(summaries.keys())
            print(f"[WARNING] Missing summaries for: {list(missing)[:5]}...")

        return summaries

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        import traceback

        traceback.print_exc()
        return {}


def get_parquet_context(
    parquet_paths: List[str],
    use_union_by_name: bool,
    all_parquet_glob_pattern: str,
    config: Any,
) -> Tuple[str, pd.DataFrame]:
    conn = setup_duckdb_azure_connection(config)
    schema_lines = []
    sample_dfs = []

    for path in parquet_paths:
        try:
            df = conn.execute(
                f"SELECT * FROM read_parquet('{path}') LIMIT 10"
            ).fetchdf()
            df.insert(0, "__TABLE__", os.path.splitext(os.path.basename(path))[0])
            sample_dfs.append(df)

            schema_df = conn.execute(
                f"DESCRIBE (SELECT * FROM read_parquet('{path}'))"
            ).fetchdf()
            schema_lines.append(f"\nTable: {os.path.basename(path)}")
            for _, row in schema_df.iterrows():
                schema_lines.append(f"  {row['column_name']} ({row['column_type']})")
        except Exception:
            pass

    combined_schema = "\n".join(schema_lines)
    combined_sample = (
        pd.concat(sample_dfs, ignore_index=True, sort=False)
        if sample_dfs
        else pd.DataFrame()
    )
    return combined_schema, combined_sample


def run_ingestion(
    input_files: List[str],
    enable_llm_summaries: bool = True,
    output_json_path: str = None,
) -> str:
    config = get_config(VectorDBType.CHROMADB)

    try:
        setup_duckdb_azure_connection(config)
        atexit.register(close_persistent_duckdb_connection)
    except Exception as e:
        print(f"[FATAL] Failed to setup DuckDB: {e}")
        sys.exit(1)

    existing_files = [path for path in input_files if os.path.exists(path)]

    if not existing_files:
        print("[ERROR] No valid input files found")
        sys.exit(1)

    all_parquet_uris = []
    failed_files = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                convert_excel_to_parquet,
                excel_path,
                config.azure_storage.parquet_output_dir,
                config,
            ): excel_path
            for excel_path in existing_files
        }

        for future in as_completed(future_to_file):
            excel_path = future_to_file[future]
            try:
                uris = future.result()
                if uris:
                    all_parquet_uris.extend(uris)
                else:
                    failed_files.append(excel_path)
            except Exception:
                failed_files.append(excel_path)

    if not all_parquet_uris:
        print("[ERROR] No Parquet files created")
        sys.exit(1)

    print(f"\n[INFO] Building catalog for {len(all_parquet_uris)} Parquet files")
    catalog_dict = build_global_catalog(all_parquet_uris, config, enable_llm_summaries)

    json_output = {
        "success": True,
        "total_files": len(all_parquet_uris),
        "failed_files": failed_files,
        "tables": [],
    }

    for table_info in catalog_dict.values():
        table_block = {
            "file_path": table_info["parquet_path"],
            "file_name": table_info["file_name"],
            "column_count": table_info["column_count"],
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

    json_string = json.dumps(json_output, indent=2, ensure_ascii=False)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    return json_string


if __name__ == "__main__":
    try:
        result_json = run_ingestion(
            input_files=[
                "../sheets/MULTI.xlsx",
                "../sheets/loan.xlsx",
            ],
            enable_llm_summaries=True,
            output_json_path="catalog_output.json",
        )
        print(result_json)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
