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
from retrival_util import get_semantic_context_for_query 
from vector_ingestion_util import ingest_to_vector_db 

# Placeholder/Fallback logic for decomposition_util
try:
    from decomposition_util import decompose_multi_intent_query 
except ImportError:
    def decompose_multi_intent_query(llm_client, user_question, deployment_name):
        print("[WARNING] Using fallback decomposition. Create decomposition_util.py for full functionality.")
        return [user_question] 
    

load_dotenv()

class Config:
    """Centralized configuration constants."""
    LLM_DEPLOYMENT_NAME = os.getenv("azureOpenAIApiDeploymentName") 
    LLM_API_KEY = os.getenv("azureOpenAIApiKey")
    LLM_ENDPOINT = os.getenv("azureOpenAIEndpoint")
    LLM_API_VERSION = os.getenv("azureOpenAIApiVersion", "2024-08-01-preview")
    LLM_MODEL_NAME = "gpt-4o"

    INPUT_FILE_PATH = '../sheets/loan.xlsx' 
    PARQUET_OUTPUT_DIR = '../data/'
    PARQUET_FILE_PATH = "" 

    @staticmethod
    def get_parquet_path(input_file_path: str) -> str:
        """Generates the dynamic output Parquet file path."""
        base_name = os.path.basename(input_file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        unique_suffix = uuid.uuid4().hex[:8] 
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        parquet_file_name = f"{file_name_without_ext}_{timestamp}_{unique_suffix}.parquet"
        return os.path.join(Config.PARQUET_OUTPUT_DIR, parquet_file_name)

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

def get_parquet_context(parquet_path: str) -> Tuple[str, pd.DataFrame]:
    """Dynamically gets the column names and types (schema) and top 5 rows."""
    try:
        if not os.path.exists(parquet_path):
             raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

        conn = duckdb.connect()
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchdf()
        
        schema_lines = [f"Column: {row['column_name']} ({row['column_type']})" 
                        for index, row in schema_df.iterrows()]
        schema_string = "TABLE SCHEMA:\n" + "\n".join(schema_lines)
        
        df_sample = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 5").fetchdf()
        conn.close()
        return schema_string, df_sample
        
    except Exception as e:
        print(f"[FATAL ERROR] Failed to retrieve Parquet context: {e}")
        return "TABLE SCHEMA: Failed to load dynamic schema.", pd.DataFrame()

# 2. DATA PROCESSING AND DUCKDB EXECUTION
def convert_excel_to_parquet(input_path: str, output_path: str) -> None:
    """Reads Excel, converts to Parquet, saves locally, and frees memory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[INFO] Converting '{os.path.basename(input_path)}' to Parquet...")
    
    df = pd.read_excel(input_path, engine='openpyxl')
    
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    print(f"[SUCCESS] Parquet file saved at: {output_path}")
    
    # CRITICAL: Delete the DataFrame object to free up RAM
    del df
    print("[INFO] Pandas DataFrame deleted from memory.")

def execute_duckdb_query(query: str) -> pd.DataFrame:
    """Executes a SQL query against the local Parquet file using DuckDB."""
    try:
        conn = duckdb.connect()
        result_df = conn.execute(query).fetchdf()
        conn.close()
        return result_df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

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


def process_single_query(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame, parquet_path: str, excel_path: str) -> Tuple[str, pd.DataFrame, float, str]:
    """
    CORE WORKER: Processes a single, atomic query through the Hybrid RAG pipeline.
    This function contains the logic for routing, RAG lookup, SQL generation, and execution.
    """
    
    start_time = time.time()
    
    # 1. Routing
    intent = route_query_intent(llm_client, user_question, schema, df_sample)
    
    AUGMENTATION_HINT = ""
    semantic_lookup_duration = 0

    # 2. RAG Lookup (if needed)
    if intent == 'SEMANTIC_SEARCH':
        lookup_start = time.time()
        # The retrival_util is correctly used here
        semantic_context = get_semantic_context_for_query(user_question, file_name=excel_path, limit=7) 
        semantic_lookup_duration = time.time() - lookup_start
        
        if semantic_context:
            AUGMENTATION_HINT = (
                "\n*** CONTEXTUAL SAMPLE DATA HINT (Retrieved via Semantic Search): ***\n"
                f"{semantic_context}\n"
                "*** END HINT ***\n"
            )
            
    # 3. SQL Generation System Prompt
    SYSTEM_PROMPT = f"""
        You are an expert SQL Generator optimized for DuckDB. Your task is to translate a user's question into a single, valid, executable SQL query.

        --- DYNAMIC DATABASE CONTEXT ---
        The data source is ALWAYS loaded using the function: 'read_parquet('{parquet_path}')'

        --- DYNAMIC DATABASE SCHEMA ---
        {schema}
        --- END OF SCHEMA ---

        -- Sample Data (Top 5 rows for general structure) --
        {df_sample.to_markdown(index=False)}
        -- End of Sample Data --

        {AUGMENTATION_HINT} 
        
        **CRITICAL RULES FOR SQL GENERATION:**
        1. **STRICTLY USE ONLY THE COLUMN NAMES** found in the "DYNAMIC DATABASE SCHEMA".
        2. **SEMANTIC FILTERING (RAG):** If the **CONTEXTUAL SAMPLE DATA HINT** is present, you **MUST** use the exact, canonical values from the hint to form a precise `WHERE...IN (...)` clause. 
        3. **OUTPUT REQUIREMENT**: Return ONLY the raw SQL query.
        """

    # 4. Call LLM for SQL
    response = llm_client.chat.completions.create(
        model=Config.LLM_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    sql_query = response.choices[0].message.content.strip()
    
    # Clean up
    if sql_query.lower().startswith("```sql"):
        sql_query = sql_query[len("```sql"):].strip()
    if sql_query.endswith("```"):
        sql_query = sql_query[:-len("```")].strip()
        
    # 5. Execute DuckDB Query
    result_df = execute_duckdb_query(sql_query)

    total_duration = time.time() - start_time
    
    # Return all necessary results for the main handler
    return user_question, result_df, total_duration, intent


def generate_and_execute_query(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame, parquet_path: str, excel_path: str) -> Dict[str, pd.DataFrame]:
    """
    REFACED: Handles multi-intent queries by decomposition and parallel execution.
    Orchestrates the new multi-topic flow.
    """
    # FIX: Corrected the syntax for the error return (lines 251, 38, 39 in the error report)
    if not llm_client: 
        return {
            "ORCHESTRATION_FAILURE": pd.DataFrame({
                'Error': ["FATAL: LLM client not initialized. Cannot proceed with decomposition or query generation."]
            })
        }

    print("\n" + "="*80)
    print(f"       *** ORCHESTRATION START: '{user_question}' ***")
    print("="*80)
    
    overall_start_time = time.time()
    
    # 1. DECOMPOSITION
    sub_queries = decompose_multi_intent_query(llm_client, user_question, Config.LLM_DEPLOYMENT_NAME)
    
    if len(sub_queries) == 1 and sub_queries[0] == user_question:
        print("-> Single-Topic Flow Detected or Decomposition Fallback. Executing sequentially.")
    else:
        print(f"-> Multi-Topic Flow Detected. Sub-queries: {sub_queries}")
        print("-> Executing sub-queries concurrently using ThreadPoolExecutor...")
        

    final_combined_results: Dict[str, pd.DataFrame] = {}
    
    # 2. PARALLEL EXECUTION (Async Flow for I/O-bound tasks) 
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor: 
        
        # Prepare tasks for submission
        future_to_query = {
            executor.submit(
                process_single_query, 
                llm_client, sub_query, schema, df_sample, parquet_path, excel_path
            ): sub_query for sub_query in sub_queries
        }

        # Collect results
        for future in as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                # Unpack the tuple from the worker function
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
    
    excel_input_path = Config.INPUT_FILE_PATH
    dynamic_parquet_path = Config.get_parquet_path(excel_input_path)
    Config.PARQUET_FILE_PATH = dynamic_parquet_path 

    llm_client = setup_llm_client()
    if not llm_client: sys.exit(1)

    print("\n--- Data Pipeline Execution Started ---")
    
    if not os.path.exists(excel_input_path):
        print(f"[CRITICAL] Input file not found at {excel_input_path}")
        sys.exit(1)

    try:
        convert_excel_to_parquet(excel_input_path, dynamic_parquet_path)
        # ingest_to_vector_db is called here, relying on the top-level import
        ingestion_status = ingest_to_vector_db(excel_input_path) 
        
        if ingestion_status is True:
            print("\n[SUCCESS] Vector DB is up-to-date with the latest data.")
        else:
            print(f"\n[WARNING] Vector DB Ingestion Failed: {ingestion_status}. Semantic search will be unreliable.")
            
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed during file processing: {e}")
        sys.exit(1)
        
    parquet_schema, df_sample = get_parquet_context(dynamic_parquet_path)
    
    print("\n" + "="*50)
    print("      DYNAMIC DATABASE CONTEXT LOADED")
    print("="*50)
    print("\n### Database Schema ###\n" + parquet_schema)
    print("\n### Top 5 Rows (Sample Data) ###\n" + df_sample.to_markdown(index=False))
    print("="*50)
    
    print("\n\n" + "#"*50)
    print("### STARTING EXECUTION ###")
    print("#"*50)
    
    # Multi-Intent Test Case: A mix of fuzzy name match (SEMANTIC) and aggregation (SQL)
    generate_and_execute_query(
        llm_client,
        "What is the maximum income for all loans and what are the details for the client Kathleen Vasqez?",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    
    
    generate_and_execute_query(
        llm_client,
        "Find all Auto loan applicants and list their approval status and interest rates.",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    print("\n--- Execution Complete ---")

if __name__ == "__main__":
    main()