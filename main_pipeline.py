import pandas as pd
import os
import sys
import duckdb
import json
from openai import AzureOpenAI
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal, Tuple, List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed 
import glob 

# Placeholder/Fallback logic for decomposition_util
try:
    from decomposition_util import decompose_multi_intent_query 
except ImportError:
    def decompose_multi_intent_query(llm_client, user_question, deployment_name):
        print("[WARNING] Using fallback decomposition. Create decomposition_util.py for full functionality.")
        return [user_question] 

# Placeholder/Fallback for multi_file_util
try:
    from multi_file_util import identify_required_tables 
except ImportError:
    # Fallback function for file identification
    def identify_required_tables(llm_client, user_question, deployment_name, catalog_schema) -> Tuple[List[str], str]:
        # Default to querying all available Parquet files
        print("[WARNING] Using fallback file identification. Defaulting to all files ('*').")
        return ["*"], ""


# Placeholder/Fallback for ingestion and retrieval (assuming they are file-path aware)
try:
    # Assuming the user has updated vector_ingestion_util with the 180s timeout fix
    from retrival_util import get_semantic_context_for_query 
except ImportError:
    def get_semantic_context_for_query(query, file_name, limit):
        print("[WARNING] Using fallback semantic context.")
        return ""
try:
    from vector_ingestion_util import ingest_to_vector_db 
except ImportError:
    def ingest_to_vector_db(file_path, sheet_name=None):
        print(f"[WARNING] Using fallback ingestion for {file_path}.")
        return True # Assume success


load_dotenv()

class Config:
    """Centralized configuration constants."""
    LLM_DEPLOYMENT_NAME = os.getenv("azureOpenAIApiDeploymentName") 
    LLM_API_KEY = os.getenv("azureOpenAIApiKey")
    LLM_ENDPOINT = os.getenv("azureOpenAIEndpoint")
    LLM_API_VERSION = os.getenv("azureOpenAIApiVersion", "2024-08-01-preview")
    LLM_MODEL_NAME = "gpt-4o"

    # --- UPDATED FILE PATHS FOR MULTI-FILE/TAB SUPPORT ---
    INPUT_FILE_PATHS: List[str] = [
        '../sheets/file1.xlsx', 
        '../sheets/file2.xlsx',
        # Add any other specific file paths here
    ]
    PARQUET_OUTPUT_DIR = '../data_parquet/' # Directory to store all Parquet files (one per sheet/file)
    ALL_PARQUET_GLOB_PATTERN = os.path.join(PARQUET_OUTPUT_DIR, '*.parquet') 
    
    @staticmethod
    def validate_env():
        """Checks for essential Azure OpenAI environment variables."""
        if not all([Config.LLM_API_KEY, Config.LLM_ENDPOINT, Config.LLM_DEPLOYMENT_NAME]):
            print("\n[FATAL] Missing one or more critical Azure OpenAI environment variables.")
            sys.exit(1)

# 1. CLIENT SETUP AND UTILITIES
def setup_llm_client():
    """Configures the native AzureOpenAI client."""
    try:
        llm_client = AzureOpenAI(
            api_key=Config.LLM_API_KEY,
            azure_endpoint=Config.LLM_ENDPOINT,
            api_version=Config.LLM_API_VERSION
        )
        print(f"[INFO] AzureOpenAI Client configured successfully for deployment: {Config.LLM_DEPLOYMENT_NAME}")
        return llm_client
    except Exception as e:
        print(f"[ERROR] Configuring AzureOpenAI Client: {e}")
        return None

def get_parquet_context(parquet_paths: List[str], use_union_by_name: bool = False) -> Tuple[str, pd.DataFrame]:
    """
    Dynamically gets the combined schema and top 5 rows from a list of Parquet files.
    The union_by_name flag is now explicit and controlled by the orchestrator.
    """
    try:
        if not parquet_paths:
             return "TABLE SCHEMA: No Parquet files specified.", pd.DataFrame()
        
        # Format the list of paths for DuckDB's read_parquet
        paths_str = ', '.join(f"'{p}'" for p in parquet_paths)
        union_flag = ", union_by_name=true" if use_union_by_name else ""
        
        conn = duckdb.connect()
        
        # Build the initial query string
        query = f"SELECT * FROM read_parquet([{paths_str}]{union_flag})"
        
        # 1. Get Schema
        schema_df = conn.execute(f"DESCRIBE {query}").fetchdf()
        
        schema_lines = [f"Column: {row['column_name']} ({row['column_type']})" 
                        for index, row in schema_df.iterrows()]
        schema_header = "TABLE SCHEMA (UNIONED)" if use_union_by_name else "TABLE SCHEMA"
        schema_string = f"{schema_header}:\n" + "\n".join(schema_lines)
        
        # 2. Get Sample Data
        df_sample = conn.execute(f"{query} LIMIT 5").fetchdf()
        conn.close()
        
        return schema_string, df_sample
        
    except Exception as e:
         print(f"[CRITICAL ERROR] Failed to retrieve Parquet context: {e}")
         return "TABLE SCHEMA: Failed to load dynamic schema.", pd.DataFrame()


# 2. DATA PROCESSING AND DUCKDB EXECUTION
def convert_excel_to_parquet(input_path: str, output_dir: str) -> List[str]:
    """Reads Excel, converts each sheet to a separate Parquet file, and returns file paths."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    parquet_paths = []
    
    print(f"[INFO] Scanning '{os.path.basename(input_path)}' for sheets...")
    
    # Read all sheets from the Excel file
    excel_sheets = pd.read_excel(input_path, sheet_name=None, engine='openpyxl')
    
    for sheet_name, df in excel_sheets.items():
        # Clean sheet name for file system (remove spaces/special chars)
        safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '_')).rstrip()
        file_name = f"{base_name}_{safe_sheet_name}.parquet"
        output_path = os.path.join(output_dir, file_name)
        
        print(f"[INFO] Converting sheet '{sheet_name}' to Parquet: {output_path}")
        
        # Save to Parquet
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        parquet_paths.append(output_path)
        
        # Ingest to vector DB for semantic RAG on this specific sheet
        ingest_to_vector_db(output_path, sheet_name=sheet_name) 
        
        # CRITICAL: Delete the DataFrame object to free up RAM
        del df
        
    print(f"[SUCCESS] {len(parquet_paths)} Parquet files created and ingested.")
    return parquet_paths

def build_global_catalog(parquet_files: List[str]) -> str:
    """Generates a simplified, LLM-readable catalog of all available logical tables."""
    catalog = []
    
    conn = duckdb.connect()
    for p_path in parquet_files:
        logical_name = os.path.splitext(os.path.basename(p_path))[0]
        try:
            schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{p_path}')").fetchdf()
            columns = ', '.join(schema_df['column_name'].tolist())
            catalog.append(f"Logical Table: {logical_name} (Columns: {columns})")
        except Exception as e:
            catalog.append(f"Logical Table: {logical_name} (ERROR: Failed to read schema)")

    conn.close()
    return "\n".join(catalog)


def execute_duckdb_query(query: str) -> pd.DataFrame:
    """Executes a SQL query against the local Parquet files using DuckDB."""
    try:
        conn = duckdb.connect()
        result_df = conn.execute(query).fetchdf()
        conn.close()
        return result_df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

# 3. LLM ROUTER & CORE RAG FUNCTION (process_single_query is updated below)

def route_query_intent(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame) -> Literal['SEMANTIC_SEARCH', 'SQL_QUERY']:
    # Placeholder for the actual router logic
    return 'SQL_QUERY' # Default to SQL for simplicity of the change

def process_single_query(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame, 
                         parquet_files: List[str], excel_path: str, join_key: str) -> Tuple[str, pd.DataFrame, float, str]:
    """
    CORE WORKER: Processes a single, atomic query through the Hybrid RAG pipeline.
    CRITICAL UPDATE: Logic to build TABLE_DEFINITION for JOIN vs. UNION.
    """
    
    start_time = time.time()
    
    # 1. Routing
    intent = route_query_intent(llm_client, user_question, schema, df_sample)
    
    AUGMENTATION_HINT = ""
    semantic_lookup_duration = 0

    # 2. RAG Lookup (if needed)
    if intent == 'SEMANTIC_SEARCH':
        lookup_start = time.time()
        # IMPORTANT: Pass the path of the *primary* Excel file for RAG to know which vector index to query
        semantic_context = get_semantic_context_for_query(user_question, file_name=excel_path, limit=7) 
        semantic_lookup_duration = time.time() - lookup_start
        
        if semantic_context:
            AUGMENTATION_HINT = (
                "\n*** CONTEXTUAL SAMPLE DATA HINT (Retrieved via Semantic Search): ***\n"
                f"{semantic_context}\n"
                "*** END HINT ***\n"
            )
            
    # 3. Dynamic Table Source Definition (FIXED LOGIC)
    
    # Check if we should use the multi-file JOIN definition
    if len(parquet_files) > 1 and join_key:
        # Scenario: Multiple tables need a JOIN
        table_aliases = []
        parquet_reads = []
        for i, p_path in enumerate(parquet_files):
            alias = f"T{i+1}"
            logical_name = os.path.splitext(os.path.basename(p_path))[0]
            table_aliases.append(f"{logical_name} (Alias: {alias})")
            parquet_reads.append(f"read_parquet('{p_path}') AS {alias}")
        
        TABLE_DEFINITION = (
            f"The query MUST JOIN the following logical tables: {', '.join(table_aliases)}. "
            f"The primary join column is: **{join_key}**. "
            f"You must use the `read_parquet()` function with aliases in your FROM/JOIN clauses, for example: "
            + ", ".join(parquet_reads)
        )
    else:
        # Scenario: Single file, or multiple files using UNION_BY_NAME (e.g., when tables_required = ['*'])
        paths_str = ', '.join(f"'{p}'" for p in parquet_files)
        
        if parquet_files == [Config.ALL_PARQUET_GLOB_PATTERN]:
             # General query across all files
             TABLE_DEFINITION = f"The data source is ALWAYS loaded as a single unified table using: 'read_parquet('{Config.ALL_PARQUET_GLOB_PATTERN}', union_by_name=true)'"
        else:
            # Single file or explicit multi-file without join key
            TABLE_DEFINITION = f"The source data is the set of files: {paths_str}. They are loaded as one unified table using: 'read_parquet([{paths_str}], union_by_name=true)'"


    # 4. SQL Generation System Prompt
    SYSTEM_PROMPT = f"""
        You are an expert SQL Generator optimized for DuckDB. Your task is to translate a user's question into a single, valid, executable SQL query.

        --- DYNAMIC DATABASE CONTEXT (Source Data) ---
        {TABLE_DEFINITION}

        --- DYNAMIC DATABASE SCHEMA ---
        {schema}
        --- END OF SCHEMA ---

        -- Sample Data (Top 5 rows for general structure) --
        {df_sample.to_markdown(index=False)}
        -- End of Sample Data --

        {AUGMENTATION_HINT} 
        
        **CRITICAL RULES FOR SQL GENERATION:**
        1. **STRICTLY USE ONLY THE COLUMN NAMES** found in the "DYNAMIC DATABASE SCHEMA".
        2. **STRICTLY FOLLOW THE TABLE DEFINITION** in your FROM/JOIN clauses. If table aliases (T1, T2) are provided, you MUST use them and their corresponding join key.
        3. **SEMANTIC FILTERING (RAG):** If the **CONTEXTUAL SAMPLE DATA HINT** is present, you **MUST** use the exact, canonical values from the hint to form a precise `WHERE...IN (...)` clause. 
        4. **OUTPUT REQUIREMENT**: Return ONLY the raw SQL query.
        """

    # 5. Call LLM for SQL
    response = llm_client.chat.completions.create(
        model=Config.LLM_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    sql_query = response.choices[0].message.content.strip()
    
    # 6. Cleanup and Execution
    if sql_query.lower().startswith("```sql"):
        sql_query = sql_query[len("```sql"):].strip()
    if sql_query.endswith("```"):
        sql_query = sql_query[:-len("```")].strip()
        
    result_df = execute_duckdb_query(sql_query)

    total_duration = time.time() - start_time
    
    return user_question, result_df, total_duration, intent


def generate_and_execute_query(llm_client: AzureOpenAI, user_question: str, all_parquet_files: List[str], global_catalog: str, excel_path: str) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the new three-stage, multi-file query flow.
    """
    if not llm_client: 
        return {"ORCHESTRATION_FAILURE": pd.DataFrame({'Error': ["FATAL: LLM client not initialized."]}) }

    print("\n" + "="*80)
    print(f"       *** ORCHESTRATION START: '{user_question}' ***")
    print("="*80)
    
    overall_start_time = time.time()
    
    # 1. FILE/TABLE IDENTIFICATION (New LLM Step)
    required_tables_names, join_key = identify_required_tables(llm_client, user_question, Config.LLM_DEPLOYMENT_NAME, global_catalog)
    
    # Map the logical names back to the actual file paths
    # Filter the master list of all_parquet_files to find the ones matching the logical names
    
    target_parquet_files = []
    use_union_by_name = False
    
    if required_tables_names == ["*"]:
        # Querying all Parquet files using the glob pattern
        target_parquet_files = [Config.ALL_PARQUET_GLOB_PATTERN]
        use_union_by_name = True
        print(f"-> STEP 1: Identified ALL available files/tables via glob: {target_parquet_files[0]} (Mode: UNION_BY_NAME)")
    else:
        # A simple, direct mapping assumption for this example:
        for name in required_tables_names:
            # Find the actual path in the list of all created files
            matching_paths = [p for p in all_parquet_files if os.path.splitext(os.path.basename(p))[0] == name]
            if matching_paths:
                target_parquet_files.append(matching_paths[0])
        
        mode = "JOIN" if len(target_parquet_files) > 1 and join_key else "UNION_BY_NAME (Fallback)"
        use_union_by_name = True if mode != "JOIN" else False
        
        print(f"-> STEP 1: Identified specific logical tables: {required_tables_names} -> {target_parquet_files} (Join Key: {join_key if join_key else 'N/A'} | Mode: {mode})")


    if not target_parquet_files:
        print("[CRITICAL] File identification resulted in no usable paths. Aborting.")
        return {"ORCHESTRATION_FAILURE": pd.DataFrame({'Error': ["No relevant files could be identified."]}) }
        
    # 2. Get SCHEMA and SAMPLE DATA for the identified tables
    # Note: Only use UNION_BY_NAME when NOT using JOIN mode (i.e., when loading all files or for simple unions)
    parquet_schema, df_sample = get_parquet_context(
        target_parquet_files,
        use_union_by_name=use_union_by_name
    )
    
    # 3. DECOMPOSITION
    sub_queries = decompose_multi_intent_query(llm_client, user_question, Config.LLM_DEPLOYMENT_NAME)
    
    if len(sub_queries) == 1 and sub_queries[0] == user_question:
        print("-> STEP 3: Single-Topic Flow Detected.")
    else:
        print(f"-> STEP 3: Multi-Topic Flow Detected. Sub-queries: {sub_queries}")
        print("-> Executing sub-queries concurrently...")
        

    final_combined_results: Dict[str, pd.DataFrame] = {}
    
    # 4. PARALLEL EXECUTION 
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor: 
        
        future_to_query = {
            executor.submit(
                # Pass the join_key to process_single_query
                process_single_query, 
                llm_client, sub_query, parquet_schema, df_sample, target_parquet_files, excel_path, join_key
            ): sub_query for sub_query in sub_queries
        }

        for future in as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                _, result_df, duration, intent = future.result() 
                
                final_combined_results[query_text] = result_df
                
                print(f"\n--- Result for Query: '{query_text}' ---")
                print(f"  Intent: **{intent}** | Duration: {duration:.2f}s")
                if not result_df.empty and 'Error' not in result_df.columns:
                    print("\nQuery Result:")
                    print(result_df.head().to_markdown(index=False)) 
                    if len(result_df) > 5:
                        print(f"... and {len(result_df) - 5} more rows.")
                elif 'Error' in result_df.columns:
                    # Log the Binder Error/Execution Error more clearly
                    print(f"[EXECUTION ERROR] {result_df['Error'].iloc[0]}")
                else:
                    print("[WARNING] Query executed but returned no results.")
                
            except Exception as exc:
                print(f"\n[CRITICAL ERROR] Sub-query processing failed for '{query_text}': {exc}")
                final_combined_results[query_text] = pd.DataFrame({'Error': [str(exc)]})
                
    overall_duration = time.time() - overall_start_time
    print("\n" + "="*80)
    print(f"       *** ORCHESTRATION COMPLETE in {overall_duration:.2f} seconds ***")
    print("="*80)

    return final_combined_results

def main():
    
    Config.validate_env()
    
    llm_client = setup_llm_client()
    if not llm_client: sys.exit(1)

    print("\n--- Data Pipeline Execution Started ---")
    
    all_parquet_files = [] # This list will hold ALL final Parquet paths

    # 1. Iterate over the explicit list of files
    for excel_input_path in Config.INPUT_FILE_PATHS:
        
        if not os.path.exists(excel_input_path):
            print(f"[WARNING] Input file not found at {excel_input_path}. Skipping.")
            continue
        
        try:
            # Step 1: Convert all sheets in this file to individual Parquet files
            sheet_parquet_paths = convert_excel_to_parquet(excel_input_path, Config.PARQUET_OUTPUT_DIR)
            all_parquet_files.extend(sheet_parquet_paths)
            
        except Exception as e:
            print(f"\n[FATAL] Pipeline failed during file processing of {excel_input_path}: {e}")
            sys.exit(1)
            
    if not all_parquet_files:
        print("[CRITICAL] No data files were successfully processed. Aborting.")
        sys.exit(1)
        
    # Step 2: Build the global catalog
    global_catalog = build_global_catalog(all_parquet_files)
        
    print("\n" + "="*50)
    print("      GLOBAL DATA CATALOG LOADED")
    print("="*50)
    print("\n### Logical Tables Available ###\n" + global_catalog)
    print("="*50)
    
    # --- STARTING EXECUTION ---
    print("\n\n" + "#"*50)
    print("### STARTING QUERY EXECUTION ###")
    print("#"*50)
    
    # We use the path to the FIRST excel file for the excel_path argument in RAG lookup
    first_excel_path = Config.INPUT_FILE_PATHS[0] 
    
    # Example 1: Multi-Intent Query (This is the failing query from the log)
    generate_and_execute_query(
        llm_client,
        "What is highest discount percentage for each price change reason",
        all_parquet_files,
        global_catalog,
        first_excel_path
    )
    
    print("\n--- Execution Complete ---")

if __name__ == "__main__":
    main()