"""
Tabular Data Converter with Azure Storage
Converts CSV/Excel to Parquet, uploads to Azure, and returns Azure URIs
"""

import gc
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient


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


def download_blob(url: str, local_path: str, timeout: int = 300) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def upload_to_azure(local_path: str, remote_name: str, config: Any) -> str:
    client = BlobServiceClient.from_connection_string(
        config.azure_storage.connection_string
    )
    container = client.get_container_client(config.azure_storage.container_name)
    blob_path = f"{config.azure_storage.parquet_output_dir.rstrip('/')}/{remote_name}"

    with open(local_path, "rb") as f:
        container.upload_blob(name=blob_path, data=f, overwrite=True)

    return f"azure://{config.azure_storage.account_name}.blob.core.windows.net/{container.container_name}/{blob_path}"


def detect_format(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {".xlsx": "excel", ".xls": "excel", ".csv": "csv", ".tsv": "tsv"}
    return format_map.get(ext, "unknown")


def generate_llm_summary(
    columns: Dict[str, str],
    sample_data: pd.DataFrame,
    llm_client: Any,
    deployment_name: str,
    filename: str,
) -> Dict[str, Any]:
    try:
        sample_text = sample_data.head(5).to_markdown(index=False, tablefmt="plain")
        if len(sample_text) > 2000:
            sample_text = sample_text[:2000] + "\n... (truncated)"

        column_info = "\n".join([f"- {col}: {dtype}" for col, dtype in columns.items()])

        prompt = f"""Analyze this dataset and provide comprehensive metadata.

Filename: {filename}
Columns ({len(columns)} total): {column_info}
Sample data (5 rows): {sample_text}

Provide:
1. For EACH column: description (max 70 chars), nullable (true/false), is_primary_key (true/false)
2. For TABLE: primary_key, foreign_keys, data_quality_score (0.0-1.0), has_duplicates, null_percentage
3. TAGS: 3-7 lowercase tags describing domain/type/topics
4. LANGUAGE: ISO 639-1 code (en, es, fr, etc)
5. MAIN_TOPICS: 2-4 broad categorical topics
6. SUMMARY: 1-sentence summary (max 150 chars)

JSON format:
{{
  "columns": {{"col_name": {{"description": "...", "nullable": true, "is_primary_key": false}}}},
  "table_metadata": {{"primary_key": "...", "foreign_keys": [], "data_quality_score": 0.95, "has_duplicates": false, "null_percentage": 2.5}},
  "tags": ["tag1", "tag2"],
  "language": "en",
  "main_topics": ["topic1", "topic2"],
  "summary": "..."
}}

Include ALL {len(columns)} columns."""

        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3500,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())

        return {
            "column_metadata": result.get("columns", {}),
            "table_metadata": result.get("table_metadata", {}),
            "tags": result.get("tags", []),
            "language": result.get("language", "en"),
            "main_topics": result.get("main_topics", []),
            "summary": result.get("summary", ""),
        }
    except Exception as e:
        return {
            "column_metadata": {},
            "table_metadata": {},
            "tags": [],
            "language": "en",
            "main_topics": [],
            "summary": "",
            "error": str(e),
        }


def extract_schema_with_summary(
    parquet_uri: str,
    config: Any,
    llm_client: Optional[Any] = None,
    deployment_name: Optional[str] = None,
) -> Dict:
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")
        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")

        schema_df = conn.execute(
            f"DESCRIBE (SELECT * FROM read_parquet('{parquet_uri}'))"
        ).fetchdf()
        col_info = {
            row["column_name"]: row["column_type"] for _, row in schema_df.iterrows()
        }

        columns = [
            {
                "name": row["column_name"],
                "type": row["column_type"],
                "nullable": row.get("null", "YES") == "YES",
            }
            for _, row in schema_df.iterrows()
        ]
        row_count = conn.execute(
            f"SELECT COUNT(*) as count FROM read_parquet('{parquet_uri}')"
        ).fetchone()[0]

        schema_info = {
            "columns": columns,
            "row_count": row_count,
            "column_count": len(columns),
        }

        if llm_client and deployment_name:
            try:
                sample_df = conn.execute(
                    f"SELECT * FROM read_parquet('{parquet_uri}') LIMIT 5"
                ).fetchdf()
                llm_metadata = generate_llm_summary(
                    col_info,
                    sample_df,
                    llm_client,
                    deployment_name,
                    os.path.basename(parquet_uri),
                )
                schema_info.update(llm_metadata)

                del sample_df
                gc.collect()
            except Exception as e:
                schema_info["llm_error"] = str(e)

        del schema_df
        gc.collect()
        return schema_info
    finally:
        conn.close()


def convert_csv_to_parquet(
    input_path: str, output_path: str, chunk_size: int = 100000
) -> None:
    conn = duckdb.connect()
    try:
        chunk_iterator = pd.read_csv(input_path, chunksize=chunk_size)
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
                    f"COPY temp_chunk TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
                )
            else:
                conn.execute(f"INSERT INTO '{output_path}' SELECT * FROM temp_chunk;")
            conn.unregister("temp_chunk")

            del df_chunk
            gc.collect()

        if is_first_chunk:
            raise ValueError("CSV file is empty")
    finally:
        conn.close()


def convert_tsv_to_parquet(
    input_path: str, output_path: str, chunk_size: int = 100000
) -> None:
    conn = duckdb.connect()
    try:
        chunk_iterator = pd.read_csv(input_path, sep="\t", chunksize=chunk_size)
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
                    f"COPY temp_chunk TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
                )
            else:
                conn.execute(f"INSERT INTO '{output_path}' SELECT * FROM temp_chunk;")
            conn.unregister("temp_chunk")

            del df_chunk
            gc.collect()

        if is_first_chunk:
            raise ValueError("TSV file is empty")
    finally:
        conn.close()


def convert_excel_to_parquet(
    input_path: str, output_dir: str, sheet_name: Optional[str] = None
) -> List[str]:
    conn = duckdb.connect()
    output_paths = []

    try:
        sheets = pd.read_excel(input_path, sheet_name=sheet_name, engine="openpyxl")
        if not isinstance(sheets, dict):
            sheets = {"Sheet1": sheets}

        base_name = re.sub(
            r"[^\w]", "_", os.path.splitext(os.path.basename(input_path))[0]
        )

        for sheet, df in sheets.items():
            if df.empty or df.shape[1] == 0:
                continue

            df = clean_column_names(df)
            safe_sheet = re.sub(r"[^\w]", "_", str(sheet))
            output_path = os.path.join(
                output_dir,
                (
                    f"{base_name}_{safe_sheet}.parquet"
                    if len(sheets) > 1
                    else f"{base_name}.parquet"
                ),
            )

            conn.register("temp_df", df)
            conn.execute(
                f"COPY temp_df TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);"
            )
            conn.unregister("temp_df")

            del df
            gc.collect()

            output_paths.append(output_path)

        del sheets
        gc.collect()
        return output_paths
    finally:
        conn.close()


def process_single_file(
    input_file: Union[str, bytes],
    config: Any,
    is_url: bool,
    llm_client: Optional[Any],
    deployment_name: Optional[str],
) -> Tuple[List[str], List[Dict]]:
    temp_dir = tempfile.mkdtemp()
    temp_file = None
    azure_uris = []
    schema_info = []

    try:
        if is_url:
            ext = os.path.splitext(input_file.split("?")[0])[1].lower()
            temp_file = os.path.join(temp_dir, f"download{ext}")
            download_blob(input_file, temp_file)
            file_path = temp_file
        elif isinstance(input_file, bytes):
            temp_file = os.path.join(temp_dir, "upload.bin")
            with open(temp_file, "wb") as f:
                f.write(input_file)
            file_path = temp_file
        else:
            file_path = input_file

        file_format = detect_format(file_path)
        local_parquet_paths = []

        if file_format == "excel":
            local_parquet_paths = convert_excel_to_parquet(file_path, temp_dir)
        elif file_format == "csv":
            base_name = re.sub(
                r"[^\w]", "_", os.path.splitext(os.path.basename(file_path))[0]
            )
            output_path = os.path.join(temp_dir, f"{base_name}.parquet")
            convert_csv_to_parquet(file_path, output_path)
            local_parquet_paths = [output_path]
        elif file_format == "tsv":
            base_name = re.sub(
                r"[^\w]", "_", os.path.splitext(os.path.basename(file_path))[0]
            )
            output_path = os.path.join(temp_dir, f"{base_name}.parquet")
            convert_tsv_to_parquet(file_path, output_path)
            local_parquet_paths = [output_path]
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        for local_path in local_parquet_paths:
            remote_name = os.path.basename(local_path)
            azure_uri = upload_to_azure(local_path, remote_name, config)
            azure_uris.append(azure_uri)

            schema = extract_schema_with_summary(
                azure_uri, config, llm_client, deployment_name
            )
            schema_info.append(
                {"azure_uri": azure_uri, "file_name": remote_name, **schema}
            )

        return azure_uris, schema_info
    finally:
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def convert_to_parquet(
    input_files: List[Union[str, bytes]],
    config: Any,
    llm_client: Optional[Any] = None,
    deployment_name: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    all_azure_uris = []
    all_schema_info = []

    for input_file in input_files:
        is_url = isinstance(input_file, str) and (
            input_file.startswith("http://") or input_file.startswith("https://")
        )

        azure_uris, schema_info = process_single_file(
            input_file, config, is_url, llm_client, deployment_name
        )

        all_azure_uris.extend(azure_uris)
        all_schema_info.extend(schema_info)

    return all_azure_uris, all_schema_info


if __name__ == "__main__":
    from openai import AzureOpenAI

    from config import VectorDBType, get_config

    config = get_config(VectorDBType.CHROMADB)

    start = time.time()
    llm_client = AzureOpenAI(
        api_key=config.azure_openai.llm_api_key,
        azure_endpoint=config.azure_openai.llm_endpoint,
        api_version=config.azure_openai.llm_api_version,
    )

    azure_uris, schemas = convert_to_parquet(
        [
            "./loan.xlsx",
            "https://gist.githubusercontent.com/dsternlicht/74020ebfdd91a686d71e785a79b318d4/raw/d3104389ba98a8605f8e641871b9ce71eff73f7e/chartsninja-data-1.csv",
        ],
        config=config,
        llm_client=llm_client,
        deployment_name=config.azure_openai.llm_deployment_name,
    )
    end = time.time()

    print("Azure URIs:", azure_uris)
    print("Schemas:", json.dumps(schemas, indent=2))
    print(f"Time:  {end - start}")
