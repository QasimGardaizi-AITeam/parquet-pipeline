import pandas as pd
import os
import sys
import json
from openai import AzureOpenAI
import atexit
from typing import Literal, Tuple, List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed 

# --- NEW UTILITY IMPORTS ---
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

# Decomposition utility
try:
    from decomposition_util import decompose_multi_intent_query 
except ImportError:
    def decompose_multi_intent_query(llm_client, user_question, deployment_name):
        print("[WARNING] Using fallback decomposition. Create decomposition_util.py for full functionality.")
        return [user_question] 

# Multi-file utility
try:
    from multi_file_util import identify_required_tables 
except ImportError:
    def identify_required_tables(llm_client, user_question, deployment_name, catalog_schema) -> Tuple[List[str], str]:
        print("[WARNING] Using fallback file identification. Defaulting to all files ('*').")
        return ["*"], ""

# Vector Database Retrieval - Using NEW SMART RETRIEVAL with ChromaDB
try:
    # FIXED: Use the correct filename
    from chroma_retrieval_util import get_semantic_context_and_files
    SMART_RETRIEVAL_AVAILABLE = True
    print("[SUCCESS] Smart retrieval imported successfully!")
except ImportError as e:
    SMART_RETRIEVAL_AVAILABLE = False
    print(f"[WARNING] Smart retrieval not available: {e}")
    try:
        from chroma_retrieval_util import get_semantic_context_for_query_chroma as get_semantic_context_for_query
        print("[INFO] Using legacy retrieval method")
    except ImportError:
        def get_semantic_context_for_query(query, file_name, limit):
            print("[WARNING] Using fallback semantic context.")
            return ""

# Import universal configuration
from config import get_config, Config, VectorDBType

# Initialize configuration
config = get_config(VectorDBType.CHROMADB)

# 1. CLIENT SETUP AND UTILITIES
def setup_llm_client():
    """Configures the native AzureOpenAI client."""
    try:
        llm_client = AzureOpenAI(
            api_key=config.azure_openai.llm_api_key,
            azure_endpoint=config.azure_openai.llm_endpoint,
            api_version=config.azure_openai.llm_api_version
        )
        print(f"[INFO] AzureOpenAI Client configured successfully for deployment: {config.azure_openai.llm_deployment_name}")
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
        You are an intelligent routing agent. Your task is to classify the user's question intent based on the provided database context.
        --- DATABASE CONTEXT ---
        SCHEMA:
        {schema}
        
        SAMPLE DATA (Top 50 Rows):
        {df_sample.to_markdown(index=False)}
        
        --- CLASSIFICATION RULES ---
        1. **SEMANTIC_SEARCH**: Choose this intent if the query involves:
           a. **Fuzzy Matching / Names**: Searching for specific entities like applicant names, or conceptual descriptions. This is critical for handling potential misspellings or variations in names.
           b. **Conceptual lookups**: Finding a value based on a non-indexed, descriptive field.
        2. **SQL_QUERY**: Choose this intent if the query involves:
           a. **Direct calculations/Aggregation**: Functions like SUM, AVG, COUNT, MAX, MIN
           b. **Precise Filtering**: Filtering on structured numeric, categorical, or date columns

        Your output MUST be a single JSON object with the key 'intent'.
        Example: {{"intent": "SEMANTIC_SEARCH"}}
        """
    
    try:
        response = llm_client.chat.completions.create(
            model=config.azure_openai.llm_deployment_name,
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

def process_single_query(
    llm_client: AzureOpenAI, 
    user_question: str, 
    schema: str, 
    df_sample: pd.DataFrame, 
    parquet_files: List[str], 
    global_catalog_dict: Dict,  # CHANGED: Now expects a dict, not string
    join_key: str, 
    config: Config
) -> Tuple[str, pd.DataFrame, float, str]:
    """
    CORE WORKER: Processes a single, atomic query through the Hybrid RAG pipeline.
    UPDATED: Now uses smart retrieval that automatically identifies correct files.
    """
    
    start_time = time.time()
    
    # 1. Routing
    intent = route_query_intent(llm_client, user_question, schema, df_sample)
    
    AUGMENTATION_HINT = ""
    semantic_lookup_duration = 0
    rag_identified_parquet_files = []

    # 2. RAG Lookup (if needed) - UPDATED WITH SMART RETRIEVAL
    if intent == 'SEMANTIC_SEARCH':
        lookup_start = time.time()
        
        if SMART_RETRIEVAL_AVAILABLE:
            # NEW: Use smart retrieval that returns both context AND the correct parquet files
            print(f"[INFO] Using smart semantic retrieval for sub-query: '{user_question}'")
            try:
                semantic_context, rag_identified_parquet_files = get_semantic_context_and_files(
                    query=user_question,
                    file_name=None,  # Let it auto-detect
                    catalog=global_catalog_dict,  # FIXED: Pass the dictionary, not string
                    limit=7,
                    score_threshold=0.5
                )
                
                if semantic_context:
                    AUGMENTATION_HINT = (
                        "\n*** CONTEXTUAL SAMPLE DATA HINT (Retrieved via Semantic Search): ***\n"
                        f"{semantic_context}\n"
                        "*** END HINT ***\n"
                    )
                    
                    # CRITICAL: If RAG identified specific files, use those instead
                    if rag_identified_parquet_files:
                        print(f"[INFO] RAG identified specific files: {rag_identified_parquet_files}")
                        parquet_files = rag_identified_parquet_files
                        
                        # Update schema and sample for the correct files
                        print(f"[INFO] Updating schema context for RAG-identified files...")
                        schema, df_sample = get_parquet_context(
                            parquet_files,
                            use_union_by_name=True,
                            all_parquet_glob_pattern=config.azure_storage.glob_pattern,
                            config=config
                        )
                else:
                    print(f"[WARNING] Smart retrieval found no relevant context")
                    
            except Exception as e:
                print(f"[ERROR] Smart retrieval failed: {e}. Falling back to original files.")
                import traceback
                traceback.print_exc()
        else:
            # FALLBACK: Use old method
            print(f"[WARNING] Smart retrieval unavailable. Using fallback method.")
            rag_file_name = config.input_file_paths[0]
            semantic_context = get_semantic_context_for_query(user_question, file_name=rag_file_name, limit=7)
            
            if semantic_context:
                AUGMENTATION_HINT = (
                    "\n*** CONTEXTUAL SAMPLE DATA HINT (Retrieved via Semantic Search): ***\n"
                    f"{semantic_context}\n"
                    "*** END HINT ***\n"
                )
        
        semantic_lookup_duration = time.time() - lookup_start
            
    # 3. Dynamic Table Source Definition
    if len(parquet_files) > 1 and join_key and not (parquet_files == [config.azure_storage.glob_pattern]):
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
        
        if parquet_files == [config.azure_storage.glob_pattern]:
             TABLE_DEFINITION = f"The data source is ALWAYS loaded as a single unified table using: 'read_parquet('{config.azure_storage.glob_pattern}', union_by_name=true)'"
        else:
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
        {df_sample.to_markdown(index=False)}
        -- End of Sample Data --

        {AUGMENTATION_HINT} 
        
        **CRITICAL RULES FOR SQL GENERATION:**
        1. **STRICTLY USE ONLY THE COLUMN NAMES** found in the "DYNAMIC DATABASE SCHEMA".
        2. **STRICTLY FOLLOW THE TABLE DEFINITION** in your FROM/JOIN clauses.
        3. **SEMANTIC FILTERING (RAG):** If the **CONTEXTUAL SAMPLE DATA HINT** is present, you **MUST** use the exact, canonical values from the hint to form a precise `WHERE...IN (...)` clause. 
        4. **UNDERSTAND FORMATS:** Handle various date and text formats appropriately.
        5. **OUTPUT REQUIREMENT**: Return ONLY the raw SQL query.
        """

    # 5. Call LLM for SQL
    try:
        response = llm_client.chat.completions.create(
            model=config.azure_openai.llm_deployment_name,
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
        
        print("-" * 50 + "SQL Query" + "-" * 50)
        print(sql_query)
            
        result_df = execute_duckdb_query(sql_query, config)

        total_duration = time.time() - start_time
        
        return user_question, result_df, total_duration, intent
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_message = f"Failed to generate or execute SQL for query: {user_question}. Exception: {e}"
        error_df = pd.DataFrame({'Error': [error_message]})
        return user_question, error_df, duration, intent


def generate_and_execute_query(
    llm_client: AzureOpenAI, 
    user_question: str, 
    all_parquet_files: List[str], 
    global_catalog_string: str,  # For LLM display
    global_catalog_dict: Dict,   # For programmatic access
    config: Config, 
    enable_debug: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the multi-file query flow.
    """
    if not llm_client:
        return {"ORCHESTRATION_FAILURE": pd.DataFrame({'Error': ["FATAL: LLM client not initialized."]}) }

    print("\n" + "="*80)
    print(f"       *** ORCHESTRATION START: '{user_question}' ***")
    print("="*80)

    overall_start_time = time.time()

    # 1. FILE/TABLE IDENTIFICATION
    required_tables_names, join_key = identify_required_tables(
        llm_client, 
        user_question, 
        config.azure_openai.llm_deployment_name, 
        global_catalog_string  # Use string for LLM
    )

    target_parquet_files = []
    use_union_by_name = False

    if required_tables_names == ["*"]:
        target_parquet_files = [config.azure_storage.glob_pattern]
        use_union_by_name = True
        print(f"-> STEP 1: Identified ALL available files/tables via glob: {target_parquet_files[0]} (Mode: UNION_BY_NAME)")
    else:
        for name in required_tables_names:
            matching_paths = [p for p in all_parquet_files if os.path.splitext(os.path.basename(p))[0] == name]
            if matching_paths:
                target_parquet_files.append(matching_paths[0])
        
        mode = "JOIN" if len(target_parquet_files) > 1 and join_key else "UNION_BY_NAME (Fallback)"
        use_union_by_name = True if mode != "JOIN" else False
        
        print(f"-> STEP 1: Identified specific logical tables: {required_tables_names} -> {target_parquet_files} (Join Key: {join_key if join_key else 'N/A'} | Mode: {mode})")

    if not target_parquet_files:
        print("[CRITICAL] File identification resulted in no usable paths. Aborting.")
        return {"ORCHESTRATION_FAILURE": pd.DataFrame({'Error': ["No relevant files could be identified."]}) }
        
    # 2. Get SCHEMA and SAMPLE DATA
    parquet_schema, df_sample = get_parquet_context(
        target_parquet_files,
        use_union_by_name=use_union_by_name,
        all_parquet_glob_pattern=config.azure_storage.glob_pattern,
        config=config
    )

    # 3. DECOMPOSITION
    print("-> STEP 2: Decomposing Multi-Intent Query...")
    sub_queries = decompose_multi_intent_query(llm_client, user_question, config.azure_openai.llm_deployment_name)
    
    if len(sub_queries) == 1 and sub_queries[0] == user_question:
        print("-> Decomposition Success: Single-Topic Flow Detected.")
    else:
        print(f"-> Decomposition Success: Found {len(sub_queries)} sub-queries.")
        print(f"-> STEP 3: Multi-Topic Flow Detected. Sub-queries: {sub_queries}")
        print("-> Executing sub-queries concurrently...")

    final_combined_results: Dict[str, pd.DataFrame] = {}
    
    # 4. PARALLEL EXECUTION 
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor: 
        
        future_to_query = {
            executor.submit(
                process_single_query, 
                llm_client, 
                sub_query, 
                parquet_schema, 
                df_sample, 
                target_parquet_files, 
                global_catalog_dict,  # FIXED: Pass dictionary
                join_key, 
                config
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
                    print(result_df.to_markdown(index=False))
                elif 'Error' in result_df.columns:
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
    try:
        setup_duckdb_azure_connection(config) 
        atexit.register(close_persistent_duckdb_connection) 
    except Exception as e:
        print(f"[FATAL] Initial DuckDB authentication failed: {e}")
        sys.exit(1)
        
    llm_client = setup_llm_client()
    if not llm_client: sys.exit(1)

    print("\n--- Data Pipeline Execution Started ---")
    
    all_parquet_files = []

    # 1. PARALLEL FILE PROCESSING
    MAX_WORKERS = 4 
    
    print(f"[INFO] Starting parallel file processing for {len(config.input_file_paths)} files with {MAX_WORKERS} threads.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(
                convert_excel_to_parquet,
                excel_input_path,
                config.azure_storage.parquet_output_dir,
                config
            ): excel_input_path for excel_input_path in config.input_file_paths
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
        
    # 2. Build the global catalog - NOW RETURNS BOTH STRING AND DICT
    global_catalog_string, global_catalog_dict = build_global_catalog(all_parquet_files, config)
        
    print("\n" + "="*50)
    print("      GLOBAL DATA CATALOG LOADED")
    print("="*50)
    print("\n### Logical Tables Available ###")
    print(global_catalog_string)
    print("="*50)
    
    # --- STARTING EXECUTION ---
    print("\n\n" + "#"*50)
    print("### STARTING QUERY EXECUTION ###")
    print("#"*50)
    
    # Example: Multi-Intent Query
    generate_and_execute_query(
        llm_client,
        "what's loan amount of Kathleen Vasqez",
        all_parquet_files,
        global_catalog_string,  # For LLM display
        global_catalog_dict,    # For programmatic access
        config,
        enable_debug=False
    )
    
    print("\n--- Execution Complete ---")

if __name__ == "__main__":
    main()