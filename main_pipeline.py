import pandas as pd
import os
import sys
import json
from openai import AzureOpenAI
import atexit
# import uuid
# from datetime import datetime
from dotenv import load_dotenv
from typing import Literal, Tuple, List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed 

# --- NEW UTILITY IMPORTS ---
# Import all DuckDB/Data Processing functions from the new utility file
try:
    from duckdb_util import (
        get_parquet_context,
        convert_excel_to_parquet,
        build_global_catalog,
        execute_duckdb_query,
        setup_duckdb_azure_connection, 
        close_persistent_duckdb_connection
    )
except ImportError:
    print("[FATAL] Missing duckdb_util.py. Please create the file.")
    sys.exit(1)
# -------------------------


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


# Vector Database Retrieval - Using ChromaDB (Switch to retrival_util for MongoDB)
try:
    # ChromaDB version (alternative: use retrival_util for MongoDB)
    from chroma_retrieval_util import get_semantic_context_for_query_chroma as get_semantic_context_for_query
except ImportError:
    def get_semantic_context_for_query(query, file_name, limit):
        print("[WARNING] Using fallback semantic context.")
        return ""


load_dotenv()

class Config:
    """Centralized configuration constants."""
    LLM_DEPLOYMENT_NAME = os.getenv("azureOpenAIApiDeploymentName") 
    LLM_API_KEY = os.getenv("azureOpenAIApiKey")
    LLM_ENDPOINT = os.getenv("azureOpenAIEndpoint")
    LLM_API_VERSION = os.getenv("azureOpenAIApiVersion", "2024-08-01-preview")
    LLM_MODEL_NAME = "gpt-4o"
    
    # Blob Storage
    AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    # --- UPDATED FILE PATHS FOR MULTI-FILE/TAB SUPPORT ---
    INPUT_FILE_PATHS: List[str] = [
        '../sheets/file1.xlsx',
        '../sheets/file2.xlsx',
        '../sheets/loan.xlsx'
    ]
    PARQUET_OUTPUT_DIR = 'parquet_files/'
    ALL_PARQUET_GLOB_PATTERN = (
        f"azure://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/"
        f"{AZURE_STORAGE_CONTAINER_NAME}/{PARQUET_OUTPUT_DIR}*.parquet"
    )
    
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

# 3. LLM ROUTER & CORE RAG FUNCTION 

def route_query_intent(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame) -> Literal['SEMANTIC_SEARCH', 'SQL_QUERY']:
    """
    Uses the LLM to classify the user's question intent based on schema and sample data.
    """
    if not llm_client: return 'SQL_QUERY'

    ROUTER_SYSTEM_PROMPT = f"""
        You are an intelligent routing agent. Your task is to classify the user's question based on the provided database context.
        --- DATABASE CONTEXT ---
        SCHEMA:
        {schema}
        
        SAMPLE DATA (Top 5 Rows):
        {df_sample.to_markdown(index=False)}
        
        --- CLASSIFICATION RULES ---
        1. **SEMANTIC_SEARCH**: Choose this intent if the query involves: fuzzy matching, potentially misspelled names (like 'Smithe'), or conceptual descriptions that require vector similarity (like 'high value collateral'). 
        2. **SQL_QUERY**: Choose this intent if the query involves: direct calculations, aggregation functions (SUM, AVG, COUNT, MIN, MAX), or precise filtering on structured numeric or categorical columns.

        Your output MUST be a single JSON object with the key 'intent'.
        Example: {{"intent": "SEMANTIC_SEARCH"}}
        """
    
    try:
        response = llm_client.chat.completions.create(
            model=Config.LLM_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {user_question}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        json_output = json.loads(response.choices[0].message.content.strip())
        intent = json_output.get("intent", "SQL_QUERY").upper()
        return intent if intent in ['SEMANTIC_SEARCH', 'SQL_QUERY'] else 'SQL_QUERY'
            
    except Exception as e:
        return 'SQL_QUERY'

def process_single_query(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame, 
                         parquet_files: List[str], excel_path: str, join_key: str, config: Config) -> Tuple[str, pd.DataFrame, float, str]:
    """
    CORE WORKER: Processes a single, atomic query through the Hybrid RAG pipeline.
    CRITICAL UPDATE: Logic to build TABLE_DEFINITION for JOIN vs. UNION.
    (Added config argument for execute_duckdb_query)
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
    if len(parquet_files) > 1 and join_key and not (parquet_files == [Config.ALL_PARQUET_GLOB_PATTERN]):
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
        # Scenario: Single file, or multiple files using UNION_BY_NAME
        paths_str = ', '.join(f"'{p}'" for p in parquet_files)
        
        if parquet_files == [config.ALL_PARQUET_GLOB_PATTERN]:
             # General query across all files
             TABLE_DEFINITION = f"The data source is ALWAYS loaded as a single unified table using: 'read_parquet('{config.ALL_PARQUET_GLOB_PATTERN}', union_by_name=true)'"
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

        -- Sample Data (Top 50 rows from each table) --
        -- Note: In JOIN mode, __TABLE__ column shows which table each row is from (e.g., "T1:file1_Sheet1") --
        {df_sample.to_markdown(index=False)}
        -- End of Sample Data --

        {AUGMENTATION_HINT} 
        
        **CRITICAL RULES FOR SQL GENERATION:**
        1. **STRICTLY USE ONLY THE COLUMN NAMES** found in the "DYNAMIC DATABASE SCHEMA".
        2. **STRICTLY FOLLOW THE TABLE DEFINITION** in your FROM/JOIN clauses. If table aliases (T1, T2) are provided, you MUST use them and their corresponding join key.
        3. **SEMANTIC FILTERING (RAG):** If the **CONTEXTUAL SAMPLE DATA HINT** is present, you **MUST** use the exact, canonical values from the hint to form a precise `WHERE...IN (...)` clause. 
        4. **UNDERSTAND FORMATS:** for example if a user has given you a value to search look in sample to see what value it aligns with like (3 dec 25 can be in format 2025-12-03 or any other date format, 'Word' can be word, WORD etc)
        5. **OUTPUT REQUIREMENT**: Return ONLY the raw SQL query.
        """

    # 5. Call LLM for SQL
    response = llm_client.chat.completions.create(
        model=config.LLM_DEPLOYMENT_NAME,
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
        
    result_df = execute_duckdb_query(sql_query, config) # <-- PASS CONFIG HERE

    total_duration = time.time() - start_time
    
    return user_question, result_df, total_duration, intent


def generate_and_execute_query(llm_client: AzureOpenAI, user_question: str, all_parquet_files: List[str], global_catalog: str, excel_path: str, config: Config, enable_debug: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the new three-stage, multi-file query flow.
    (Added config argument and enable_debug flag)
    """
    if not llm_client:
        return {"ORCHESTRATION_FAILURE": pd.DataFrame({'Error': ["FATAL: LLM client not initialized."]}) }

    print("\n" + "="*80)
    print(f"       *** ORCHESTRATION START: '{user_question}' ***")
    print("="*80)

    overall_start_time = time.time()

    # 1. FILE/TABLE IDENTIFICATION (New LLM Step)
    required_tables_names, join_key = identify_required_tables(llm_client, user_question, config.LLM_DEPLOYMENT_NAME, global_catalog)

    # Map the logical names back to the actual file paths
    # Filter the master list of all_parquet_files to find the ones matching the logical names

    target_parquet_files = []
    use_union_by_name = False

    if required_tables_names == ["*"]:
        # Querying all Parquet files using the glob pattern
        target_parquet_files = [config.ALL_PARQUET_GLOB_PATTERN]
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
    # CRITICAL: Pass the glob pattern to the util function
    parquet_schema, df_sample = get_parquet_context(
        target_parquet_files,
        use_union_by_name=use_union_by_name,
        all_parquet_glob_pattern=config.ALL_PARQUET_GLOB_PATTERN, # Passed here
        config=config # <-- PASS CONFIG HERE
    )

    # DEBUG: Data inspection (only runs if enable_debug=True)
    if enable_debug and len(target_parquet_files) >= 2 and join_key:
        print("\n" + "="*80)
        print("       *** DEBUG MODE: DATA INSPECTION ***")
        print("="*80)

        from duckdb_util import execute_duckdb_query

        for i, p_path in enumerate(target_parquet_files[:2], 1):  # Check first 2 files
            table_name = os.path.splitext(os.path.basename(p_path))[0]
            print(f"\n--- DEBUG: Table {i} ({table_name}) ---")

            # Count rows and unique join keys
            debug_query = f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT {join_key}) as unique_{join_key}s,
                COUNT({join_key}) as non_null_{join_key}s
            FROM read_parquet('{p_path}')
            """
            result = execute_duckdb_query(debug_query, config)
            print(result.to_string(index=False))

            # Sample join key values
            sample_query = f"SELECT DISTINCT {join_key} FROM read_parquet('{p_path}') LIMIT 5"
            sample_keys = execute_duckdb_query(sample_query, config)
            print(f"\nSample {join_key} values:")
            print(sample_keys.to_string(index=False))

        # Check for common join keys between first two tables
        if len(target_parquet_files) >= 2:
            print(f"\n--- DEBUG: Common {join_key} values between tables ---")
            common_query = f"""
            SELECT COUNT(*) as common_{join_key}_count
            FROM (
                SELECT DISTINCT {join_key}
                FROM read_parquet('{target_parquet_files[0]}')
            ) T1
            INNER JOIN (
                SELECT DISTINCT {join_key}
                FROM read_parquet('{target_parquet_files[1]}')
            ) T2
            ON T1.{join_key} = T2.{join_key}
            """
            common_result = execute_duckdb_query(common_query, config)
            print(common_result.to_string(index=False))

            # Check data types
            print(f"\n--- DEBUG: Data types for {join_key} ---")
            for i, p_path in enumerate(target_parquet_files[:2], 1):
                table_name = os.path.splitext(os.path.basename(p_path))[0]
                type_query = f"DESCRIBE (SELECT {join_key} FROM read_parquet('{p_path}') LIMIT 1)"
                type_result = execute_duckdb_query(type_query, config)
                print(f"Table {i} ({table_name}): {type_result['column_type'][0]}")

        print("\n" + "="*80)
        print("       *** END DEBUG MODE ***")
        print("="*80 + "\n")

    # 3. DECOMPOSITION
    sub_queries = decompose_multi_intent_query(llm_client, user_question, config.LLM_DEPLOYMENT_NAME)
    
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
                llm_client, sub_query, parquet_schema, df_sample, target_parquet_files, excel_path, join_key, config # <-- PASS CONFIG HERE
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
    config = Config() # Instantiate the config object here
    
    try:
        setup_duckdb_azure_connection(config) 
        # 2. Register a cleanup function to close the connection on exit
        atexit.register(close_persistent_duckdb_connection) 
    except Exception as e:
        print(f"[FATAL] Initial DuckDB authentication failed: {e}")
        sys.exit(1)
        
    llm_client = setup_llm_client()
    if not llm_client: sys.exit(1)

    print("\n--- Data Pipeline Execution Started ---")
    
    all_parquet_files = [] # This list will hold ALL final Parquet paths

    # 1. PARALLEL FILE PROCESSING AND INGESTION
    MAX_WORKERS = 4 
    
    print(f"[INFO] Starting parallel file processing for {len(config.INPUT_FILE_PATHS)} files with {MAX_WORKERS} threads.")
    
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        future_to_file = {
            executor.submit(
                convert_excel_to_parquet, 
                excel_input_path, 
                config.PARQUET_OUTPUT_DIR,
                config
            ): excel_input_path for excel_input_path in config.INPUT_FILE_PATHS
            if os.path.exists(excel_input_path)
        }
        
        for future in as_completed(future_to_file):
            excel_input_path = future_to_file[future]
            try:
                sheet_parquet_paths = future.result()
                all_parquet_files.extend(sheet_parquet_paths)
            except Exception as exc:
                print(f"\n[FATAL] Pipeline failed during processing of {excel_input_path}: {exc}")
                
    if not all_parquet_files:
        print("[CRITICAL] No data files were successfully processed. Aborting.")
        sys.exit(1)
        
    # Step 2: Build the global catalog
    global_catalog = build_global_catalog(all_parquet_files, config) # <-- PASS CONFIG HERE
        
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
    first_excel_path = config.INPUT_FILE_PATHS[0] 
    
    # Example 1: Multi-Intent Query (This is the failing query from the log)
    generate_and_execute_query(
        llm_client,
        "What is Maximum Discount offered for each price change reason and what is loan amount of Harrison, Ters.",
        all_parquet_files,
        global_catalog,
        first_excel_path,
        config,
        enable_debug=False  # Set to True to enable detailed debug output (slower)
    )
    
    print("\n--- Execution Complete ---")

if __name__ == "__main__":
    main()