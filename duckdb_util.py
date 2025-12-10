import os
import pandas as pd
import duckdb
from typing import List, Tuple, Dict, Any
import glob

# NOTE: The dependency on 'ingest_to_vector_db' from vector_ingestion_util.py is preserved
# The placeholder/fallback for ingestion must be handled in main_pipeline.py and imported here.
try:
    from vector_ingestion_util import ingest_to_vector_db 
except ImportError:
    # Fallback to avoid breaking duckdb_util if vector_ingestion_util isn't in place yet
    def ingest_to_vector_db(file_path, sheet_name=None):
        print(f"[WARNING] Using internal fallback ingestion for {file_path}. Please install vector_ingestion_util.")
        return True

# NOTE: Config.ALL_PARQUET_GLOB_PATTERN must be passed into get_parquet_context 
# if it is needed by the UNION mode logic. We will restructure the function calls slightly 
# in the main file to pass this parameter.

def get_parquet_context(parquet_paths: List[str], use_union_by_name: bool = False, all_parquet_glob_pattern: str = None) -> Tuple[str, pd.DataFrame]:
    """
    Dynamically gets the combined schema and top 5 rows from a list of Parquet files.
    Handles JOIN (separate schemas) vs. UNION (combined schema).
    """
    try:
        if not parquet_paths:
             return "TABLE SCHEMA: No Parquet files specified.", pd.DataFrame()
        
        conn = duckdb.connect()
        schema_string = ""
        df_sample = pd.DataFrame()
        
        # Check if the path list contains the glob pattern itself
        is_global_union = len(parquet_paths) == 1 and parquet_paths[0] == all_parquet_glob_pattern

        if use_union_by_name:
            # --- UNION MODE ---
            paths_str = ', '.join(f"'{p}'" for p in parquet_paths)
            union_flag = ", union_by_name=true"
            # If it's a global union, the string is already the glob pattern
            if is_global_union:
                 query = f"SELECT * FROM read_parquet('{all_parquet_glob_pattern}', union_by_name=true)"
            else:
                 query = f"SELECT * FROM read_parquet([{paths_str}]{union_flag})"

            schema_df = conn.execute(f"DESCRIBE {query}").fetchdf()
            
            schema_lines = [f"Column: {row['column_name']} ({row['column_type']})" 
                            for index, row in schema_df.iterrows()]
            schema_string = f"TABLE SCHEMA (UNIONED):\n" + "\n".join(schema_lines)
            
            df_sample = conn.execute(f"{query} LIMIT 5").fetchdf()
            
        else:
            # --- JOIN MODE ---
            all_schema_lines = ["TABLE SCHEMAS (JOIN MODE)"]
            
            for i, p_path in enumerate(parquet_paths):
                logical_name = os.path.splitext(os.path.basename(p_path))[0]
                alias = f"T{i+1}"
                
                # Correct DuckDB Syntax
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
         print(f"[CRITICAL ERROR] Failed to retrieve Parquet context: {e}")
         return "TABLE SCHEMA: Failed to load dynamic schema.", pd.DataFrame()


def execute_duckdb_query(query: str) -> pd.DataFrame:
    """Executes a SQL query against the local Parquet files using DuckDB."""
    try:
        conn = duckdb.connect()
        result_df = conn.execute(query).fetchdf()
        conn.close()
        return result_df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})


def convert_excel_to_parquet(input_path: str, output_dir: str) -> List[str]:
    """Reads Excel, converts each sheet to a separate Parquet file, and triggers ingestion."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    parquet_paths = []
    
    print(f"[INFO] Scanning '{os.path.basename(input_path)}' for sheets...")
    
    try:
        excel_sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"[ERROR] Failed to read Excel file {input_path}: {e}")
        return []
        
    for sheet_name, df in excel_sheets.items():
        safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = f"{base_name}_{safe_sheet_name}.parquet"
        output_path = os.path.join(output_dir, file_name)
        
        print(f"[INFO] Converting sheet '{sheet_name}' to Parquet: {output_path}")
        
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        parquet_paths.append(output_path)
        
        # Trigger ingestion to vector DB
        ingest_to_vector_db(output_path, sheet_name=sheet_name) 
        
        del df
        
    print(f"[SUCCESS] {len(parquet_paths)} Parquet files created and ingested from {os.path.basename(input_path)}.")
    return parquet_paths


def build_global_catalog(parquet_files: List[str]) -> str:
    """Generates a simplified, LLM-readable catalog of all available logical tables."""
    catalog = []
    
    conn = duckdb.connect()
    for p_path in parquet_files:
        logical_name = os.path.splitext(os.path.basename(p_path))[0]
        try:
            # Correct DuckDB Syntax
            schema_df = conn.execute(f"DESCRIBE (SELECT * FROM read_parquet('{p_path}'))").fetchdf()
            columns = ', '.join(schema_df['column_name'].tolist())
            catalog.append(f"Logical Table: {logical_name} (Columns: {columns})")
        except Exception as e:
            catalog.append(f"Logical Table: {logical_name} (ERROR: Failed to read schema: {e})")

    conn.close()
    return "\n".join(catalog)