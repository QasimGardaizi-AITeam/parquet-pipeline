import os
import pandas as pd
import duckdb
from typing import List, Tuple, Any
import glob
import sys 
from azure.storage.blob import BlobServiceClient 

# Placeholder/Fallback for vector_ingestion_util
try:
    from vector_ingestion_util import ingest_to_vector_db 
except ImportError:
    def ingest_to_vector_db(file_path, sheet_name=None):
        print(f"[WARNING] Using internal fallback ingestion for {file_path}.")
        return True

# --- FIXED AZURE CONNECTION SETUP ---

def setup_duckdb_azure_connection(conn: duckdb.DuckDBPyConnection, config: Any):
    """Initializes DuckDB extensions and sets Azure Blob Storage credentials."""
    try:
        # 1. Install and Load Azure Extension
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")
        
        # 2. Set Credentials using Connection String (CORRECT METHOD)
        connection_string = config.AZURE_STORAGE_CONNECTION_STRING
        
        if connection_string:
            print(f"[INFO] Configuring DuckDB Azure connection for storage: {config.AZURE_STORAGE_ACCOUNT_NAME}")
            
            # CORRECT: Use the azure_storage_connection_string parameter
            # Escape single quotes in the connection string by doubling them
            escaped_conn_str = connection_string.replace("'", "''")
            conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")
            
            print("[SUCCESS] DuckDB Azure authentication configured successfully.")
        else:
            print("[CRITICAL ERROR] DuckDB Azure connection failed: Missing AZURE_STORAGE_CONNECTION_STRING in config.")
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required")
             
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print(f"[CRITICAL ERROR] Failed to set up DuckDB Azure connection: {e}")
        raise  # Re-raise to stop execution if auth fails
        
def get_blob_uri(file_name: str, config: Any) -> str:
    """Constructs the full Azure Blob Storage URI for a given file name."""
    account = config.AZURE_STORAGE_ACCOUNT_NAME
    container = config.AZURE_STORAGE_CONTAINER_NAME
    return f"azure://{account}.blob.core.windows.net/{container}/{config.PARQUET_OUTPUT_DIR}{file_name}"

def upload_file_to_azure(local_path: str, remote_file_path: str, config: Any) -> str:
    """
    Uploads a local file to Azure Blob Storage using the SDK.
    Returns the full Azure URI on success.
    """
    try:
        connection_string = config.AZURE_STORAGE_CONNECTION_STRING
        container_name = config.AZURE_STORAGE_CONTAINER_NAME

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        blob_path = f"{config.PARQUET_OUTPUT_DIR}{remote_file_path}"
        
        print(f"[INFO] Uploading {os.path.basename(local_path)} to Azure Blob: {blob_path}")

        with open(local_path, "rb") as data:
            container_client.upload_blob(name=blob_path, data=data, overwrite=True)
            
        return get_blob_uri(remote_file_path, config)

    except Exception as e:
        raise Exception(f"Azure Upload Failed for {local_path}: {e}")

# -----------------------------------------------
# 1. CORE DUCKDB UTILITIES (MODIFIED FOR AZURE)
# -----------------------------------------------

def get_parquet_context(parquet_paths: List[str], use_union_by_name: bool = False, all_parquet_glob_pattern: str = None, config: Any = None) -> Tuple[str, pd.DataFrame]:
    """
    Dynamically gets the combined schema and top 5 rows from a list of Parquet URIs/Paths.
    """
    try:
        if not parquet_paths:
            return "TABLE SCHEMA: No Parquet files specified.", pd.DataFrame()
        
        conn = duckdb.connect()
        
        # CRITICAL: Setup Azure connection before any DuckDB storage operation
        if config:
            setup_duckdb_azure_connection(conn, config)

        schema_string = ""
        df_sample = pd.DataFrame()
        
        is_global_union = len(parquet_paths) == 1 and parquet_paths[0] == all_parquet_glob_pattern

        if use_union_by_name:
            # --- UNION MODE ---
            if is_global_union:
                query = f"SELECT * FROM read_parquet('{all_parquet_glob_pattern}', union_by_name=true)"
            else:
                paths_str = ', '.join(f"'{p}'" for p in parquet_paths)
                query = f"SELECT * FROM read_parquet([{paths_str}], union_by_name=true)"

            schema_df = conn.execute(f"DESCRIBE {query}").fetchdf()
            
            schema_lines = [f"Column: {row['column_name']} ({row['column_type']})" 
                            for index, row in schema_df.iterrows()]
            schema_string = f"TABLE SCHEMA (UNIONED):\n" + "\n".join(schema_lines)
            
            df_sample = conn.execute(f"{query} LIMIT 5").fetchdf()
            
        else:
            # --- JOIN MODE ---
            all_schema_lines = ["TABLE SCHEMAS (JOIN MODE)"]
            
            for i, p_path in enumerate(parquet_paths):
                logical_name = os.path.splitext(os.path.basename(p_path.split('/')[-1]))[0] 
                alias = f"T{i+1}"
                
                describe_query = f"DESCRIBE (SELECT * FROM read_parquet('{p_path}'))" 
                schema_df = conn.execute(describe_query).fetchdf()
                
                all_schema_lines.append(f"\n--- LOGICAL TABLE: {logical_name} (Alias: {alias}) ---")
                
                col_lines = [f"Column: {row['column_name']} ({row['column_type']})" 
                             for index, row in schema_df.iterrows()]
                all_schema_lines.extend(col_lines)

            schema_string = "\n".join(all_schema_lines)
            
            if parquet_paths:
                df_sample = conn.execute(f"SELECT * FROM read_parquet('{parquet_paths[0]}') LIMIT 5").fetchdf()
                 
        conn.close()
        return schema_string, df_sample
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to retrieve Parquet context from Azure: {e}")
        return "TABLE SCHEMA: Failed to load dynamic schema.", pd.DataFrame()


def execute_duckdb_query(query: str, config: Any) -> pd.DataFrame:
    """Executes a SQL query against the Azure Parquet files using DuckDB."""
    try:
        conn = duckdb.connect()
        # CRITICAL: Setup Azure connection
        setup_duckdb_azure_connection(conn, config)
        
        result_df = conn.execute(query).fetchdf()
        conn.close()
        return result_df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


def convert_excel_to_parquet(input_path: str, output_dir: str, config: Any) -> List[str]:
    """
    Reads Excel, converts each sheet to a Parquet file locally, uploads to Azure,
    and returns the list of Azure URIs.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    azure_uris = []
    
    print(f"[INFO] Scanning '{os.path.basename(input_path)}' for sheets...")
    
    try:
        excel_sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"[ERROR] Failed to read Excel file {input_path}: {e}")
        return []

    # Ensure a local directory exists for temporary files
    LOCAL_TEMP_DIR = "/tmp/parquet_cache"
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    
    conn = duckdb.connect()

    for sheet_name, df in excel_sheets.items():
        safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '_')).rstrip()
        remote_file_name = f"{base_name}_{safe_sheet_name}.parquet"
        
        # 1. Write the Parquet file to a local temporary path
        local_temp_path = os.path.join(LOCAL_TEMP_DIR, remote_file_name)
        
        print(f"[INFO] Converting sheet '{sheet_name}' to Parquet and writing locally to: {local_temp_path}")
        
        try:
            # Use DuckDB to write the DataFrame to the local path
            conn.register('temp_df', df)
            conn.execute(f"COPY temp_df TO '{local_temp_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
            conn.unregister('temp_df')

            # 2. Upload the local file to Azure Blob Storage using the SDK
            azure_uri = upload_file_to_azure(local_temp_path, remote_file_name, config)
            
            # 3. Trigger ingestion to vector DB, passing the Azure URI
            ingest_to_vector_db(azure_uri, sheet_name=sheet_name) 
            
            azure_uris.append(azure_uri)
            
            # 4. Clean up local temp file
            os.remove(local_temp_path)
            del df
            
        except Exception as e:
            print(f"[ERROR] Failed to convert/upload sheet '{sheet_name}': {e}")
            
    conn.close()
    print(f"[SUCCESS] {len(azure_uris)} Parquet files created and ingested from {os.path.basename(input_path)}.")
    return azure_uris


def build_global_catalog(parquet_files: List[str], config: Any) -> str:
    """Generates a simplified, LLM-readable catalog of all available logical tables."""
    catalog = []
    
    conn = duckdb.connect()
    # CRITICAL: Setup Azure connection
    setup_duckdb_azure_connection(conn, config)

    for p_path in parquet_files:
        logical_name = os.path.splitext(os.path.basename(p_path.split('/')[-1]))[0]
        try:
            schema_df = conn.execute(f"DESCRIBE (SELECT * FROM read_parquet('{p_path}'))").fetchdf()
            columns = ', '.join(schema_df['column_name'].tolist())
            catalog.append(f"Logical Table: {logical_name} (Columns: {columns})")
        except Exception as e:
            catalog.append(f"Logical Table: {logical_name} (ERROR: Failed to read schema: {e})")

    conn.close()
    return "\n".join(catalog)