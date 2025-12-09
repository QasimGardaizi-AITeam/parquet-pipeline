import pandas as pd
import os
import sys
import duckdb
import json
from openai import AzureOpenAI
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal, Tuple
import time

# NOTE: Ensure retrival_util.py has the debugging print for context retrieval
from retrival_util import get_semantic_context_for_query 
# NOTE: Ensure vector_ingestion_util.py has the new wait_for_vector_sync logic
from vector_ingestion_util import ingest_to_vector_db 

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
        print(f"[ERROR] DuckDB query execution failed: {e}")
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
        
        1. **SEMANTIC_SEARCH**: Choose this intent if the query involves: fuzzy matching, potentially misspelled names (like 'Smithe'), or conceptual descriptions that require vector similarity (like 'high value collateral'). This path needs context retrieval (RAG).
        
        2. **SQL_QUERY**: Choose this intent if the query involves: direct calculations, aggregation functions (SUM, AVG, COUNT, MIN, MAX), or precise filtering on structured numeric or categorical columns (like 'credit_score > 750' or 'loan_status = Approved'). This path can be solved directly via SQL generation.

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
        print(f"[ERROR] LLM Router failed: {e}. Defaulting to SQL_QUERY.")
        return 'SQL_QUERY'


def generate_and_execute_query(llm_client: AzureOpenAI, user_question: str, schema: str, df_sample: pd.DataFrame, parquet_path: str, excel_path: str):
    """
    Hybrid RAG: Router -> Conditional Pipeline -> SQL Generation -> DuckDB Execution.
    """
    if not llm_client: return pd.DataFrame({'Error': ["LLM client not initialized."]})

    print(f"\n\n--- Processing Query ---")
    print(f"User Question: '{user_question}'")
    
    start_time = time.time()
    
    intent = route_query_intent(llm_client, user_question, schema, df_sample)
    
    print(f"-> STEP 1: Router Decision: **{intent}**")
    
    AUGMENTATION_HINT = ""
    semantic_lookup_duration = 0

    if intent == 'SEMANTIC_SEARCH':
        lookup_start = time.time()
        # The excel_path is passed as 'file_name' because the ingestion utility uses it 
        # to derive the MongoDB collection name (e.g., data_source_loan)
        print(f"-> STEP 2: Executing SEMANTIC SEARCH pipeline (using '{excel_path}' for context lookup)...")
        
        semantic_context = get_semantic_context_for_query(user_question, file_name=excel_path, limit=7) 
        semantic_lookup_duration = time.time() - lookup_start
        
        if semantic_context:
            AUGMENTATION_HINT = (
                "\n*** CONTEXTUAL SAMPLE DATA HINT (Retrieved via Semantic Search): ***\n"
                "This data is highly relevant to the user's question for fuzzy names/terms. "
                "Find all values that match or almost match the user query"
                "You MUST use the EXACT values found here to construct a precise `WHERE...IN (...)` clause. "
                f"{semantic_context}\n"
                "*** END HINT ***\n"
            )
            print("-> Context found. Prompt will be augmented.")
        else:
            print("-> WARNING: Semantic search failed to find context. Relying solely on schema.")
            
    else:
        print("-> STEP 2: Executing DIRECT SQL pipeline. No vector search needed.")

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
        2. **STRICTLY USE ONLY THE TABLE NAME** implied by the 'read_parquet' function.

        3. **SEMANTIC FILTERING (RAG):** If the **CONTEXTUAL SAMPLE DATA HINT** is present, and the user's query references a specific entity (name, ID, or fuzzy term), you **MUST** use the exact, canonical values from the hint to form a precise `WHERE...IN (...)` clause. Do not invent names or values.

        4. **CONTEXTUAL INFERENCE (Filtering Logic):** When the user query involves a filter based on common categories (e.g., location/State/City, Date, Amount, Applicant Name), you **MUST** analyze the DYNAMIC DATABASE SCHEMA and SAMPLE DATA to identify the most appropriate column for filtering.
        - **Example:** For a query like "loans in New York", identify the column name that clearly represents location (e.g., `state`, `city`, `location`, or `address`), and **NEVER** use a column clearly designed for personal identification (like an `applicant_name` or `ID` column) for a location filter.
        - **Example:** For filtering by time, identify the column with the date/datetime type.

        5. **OUTPUT REQUIREMENT**: Return ONLY the raw SQL query. Do not include any explanations, comments, or surrounding Markdown formatting (e.g., ```sql`).
        """

    print("-> STEP 3: Calling LLM to generate SQL...")
    response = llm_client.chat.completions.create(
        model=Config.LLM_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ],
        temperature=0.0
    )
    sql_query = response.choices[0].message.content.strip()
    
    # Clean up common LLM code block wrappers
    if sql_query.lower().startswith("```sql"):
        sql_query = sql_query[len("```sql"):].strip()
    if sql_query.endswith("```"):
        sql_query = sql_query[:-len("```")].strip()

    print(f"-> Generated SQL: {sql_query}")
        
    print("-> STEP 4: Executing DuckDB query...")
    result_df = execute_duckdb_query(sql_query)

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"--- Query Finished in {total_duration:.2f} seconds (Semantic lookup: {semantic_lookup_duration:.2f}s) ---")
    
    if not result_df.empty and 'Error' not in result_df.columns:
        print("\nQuery Result:")
        print(result_df.to_markdown(index=False))
    elif 'Error' in result_df.columns:
        print(f"\n[EXECUTION ERROR] {result_df['Error'].iloc[0]}")
    else:
        print("[WARNING] Query executed but returned no results.")

    return result_df

# 4. MAIN EXECUTION
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
        # CRITICAL: Use the excel_input_path (source file name) for ingestion
        # so it can determine the collection name (e.g., data_source_loan)
        ingestion_status = ingest_to_vector_db(excel_input_path) 
        
        if ingestion_status is True:
            print("\n[SUCCESS] Vector DB is up-to-date with the latest data.")
        else:
            # Prints the detailed error message from the ingest_to_vector_db return value
            print(f"\n[WARNING] Vector DB Ingestion Failed: {ingestion_status}. Semantic search will be unreliable.")
            # Do not exit, continue with SQL-only path
            
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
    
    # Test 1: Fuzzy Name Match (Should trigger SEMANTIC_SEARCH and utilize the RAG sync fix)
    generate_and_execute_query(
        llm_client,
        "Find the details for the client named Kathleen Vasqez.",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    
    
    # Test 2: Fuzzy Name Match (Should trigger SEMANTIC_SEARCH and utilize the RAG sync fix)
    generate_and_execute_query(
        llm_client,
        "What is credit Score of Harrison, Teresa.",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    
    # Test 3: Aggregation Query (Should trigger SQL_QUERY)
    generate_and_execute_query(
        llm_client,
        "What is the maximum applicant income for all loans?",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    
    # Test 4: Compound Filter Query (Should trigger SQL_QUERY)
    generate_and_execute_query(
        llm_client,
        "How many loans have a credit score above 650 and are 'Approved'?",
        parquet_schema, 
        df_sample,
        dynamic_parquet_path,
        excel_input_path
    )
    
    print("\n--- Execution Complete ---")

if __name__ == "__main__":
    main()