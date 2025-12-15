"""
LangGraph RAG Pipeline
A graph-based orchestration system for multi-intent query processing with hybrid RAG.
"""

from typing import TypedDict, List, Dict, Literal, Optional
from typing_extensions import TypedDict
import pandas as pd
from openai import AzureOpenAI
import time
import json 
import os

from langgraph.graph import StateGraph, END

# Import existing utilities
from config import get_config, Config, VectorDBType
from duckdb_util import (
    get_parquet_context,
    convert_excel_to_parquet,
    build_global_catalog,
    execute_duckdb_query,
    setup_duckdb_azure_connection,
    close_persistent_duckdb_connection
)
from decomposition_util import decompose_multi_intent_query
from multi_file_util import identify_required_tables
from chroma_retrieval_util import get_semantic_context_and_files
from summary_util import generate_simple_summary

def _df_to_json_result(df: pd.DataFrame) -> str:
    """Converts a DataFrame to a JSON string for serializable state storage."""
    if 'Error' in df.columns:
        # Store error message in a serializable dictionary
        return json.dumps({"Error": df['Error'].iloc[0]})
    # Store the data rows as JSON records string
    return df.to_json(orient='records')

def _json_to_df(json_str: str) -> pd.DataFrame:
    """Converts a JSON string (stored in state) back to a DataFrame."""
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and 'Error' in data:
            return pd.DataFrame({'Error': [data['Error']]})
        if isinstance(data, list):
            # Attempt to create DataFrame from list of records
            return pd.DataFrame(data)
    except:
        pass
    # Return an error DataFrame if deserialization fails
    return pd.DataFrame({'Error': [f"Failed to deserialize result. Original content: {json_str[:50]}..."]})

def _df_to_markdown_sample(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Converts a DataFrame sample to a Markdown string for LLM prompt context."""
    if df.empty or 'Error' in df.columns:
        return "No sample data"
    return df.head(max_rows).to_markdown(index=False)

def setup_llm_client(config: Config) -> AzureOpenAI:
    """Initialize Azure OpenAI client."""
    try:
        client = AzureOpenAI(
            api_key=config.azure_openai.llm_api_key,
            azure_endpoint=config.azure_openai.llm_endpoint,
            api_version=config.azure_openai.llm_api_version
        )
        print(f"[INFO] AzureOpenAI Client configured for deployment: {config.azure_openai.llm_deployment_name}")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to configure AzureOpenAI Client: {e}")
        raise

# STATE DEFINITION (Updated for serialization)

class GraphState(TypedDict):
    """
    Represents the state of the RAG pipeline graph.
    """
    # Input
    user_question: str
    
    # Decomposition
    sub_queries: List[str]
    current_query: Optional[str]
    query_index: int
    
    # Global context
    all_parquet_files: List[str]
    global_catalog_string: str
    global_catalog_dict: Dict
    config: Config
    
    # Query-specific context
    required_tables: List[str]
    join_key: str
    target_parquet_files: List[str]
    use_union_by_name: bool
    parquet_schema: str
    df_sample: str # CHANGED: Stored as Markdown string
    
    # Routing and execution
    intent: str
    semantic_context: str
    sql_query: str
    sql_explanation: str
    execution_duration: float
    
    # Outputs
    results: Dict[str, str] # CHANGED: Stored as JSON string results
    summary: str
    error: Optional[str]
    enable_debug: bool

# GRAPH NODES

def initialize_context(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 1: Initialize the graph with global context.
    Sets up DuckDB connection.
    """
    print("\n" + "-"*80)
    print("NODE: Initialize Context")
    print("-"*80)
    
    try:
        # Setup DuckDB connection
        setup_duckdb_azure_connection(state["config"])
        
        # Initialize results dictionary
        state["results"] = {} 
        state["query_index"] = 0
        state["error"] = None
        
        print("[SUCCESS] Context initialized")
        return state
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        state["error"] = f"Initialization error: {str(e)}"
        return state


def decompose_query(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    print("\n" + "-"*80)
    print("NODE: Decompose Query")
    print("-"*80)
    
    try:
        sub_queries = decompose_multi_intent_query(
            llm_client=llm_client,
            user_question=state["user_question"],
            deployment_name=state["config"].azure_openai.llm_deployment_name
        )
        
        state["sub_queries"] = sub_queries
        print(f"[SUCCESS] Decomposed into {len(sub_queries)} sub-queries")
        
        if state.get("enable_debug", False):
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Query decomposition failed: {e}")
        state["error"] = f"Decomposition error: {str(e)}"
        state["sub_queries"] = [state["user_question"]]  # Fallback to original
        return state


def identify_data_sources(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    print("\n" + "-"*80)
    print(f"NODE: Identify Data Sources (Query {state['query_index'] + 1}/{len(state['sub_queries'])})")
    print("-"*80)
    
    try:
        current_query = state["sub_queries"][state["query_index"]]
        state["current_query"] = current_query
        
        print(f"Current query: {current_query}")
        
        # Identify required tables
        required_tables, join_key = identify_required_tables(
            llm_client=llm_client,
            user_question=current_query,
            deployment_name=state["config"].azure_openai.llm_deployment_name,
            catalog_schema=state["global_catalog_string"]
        )
        
        state["required_tables"] = required_tables
        state["join_key"] = join_key
        
        # Determine target parquet files
        if "*" in required_tables or required_tables == ["*"]:
            state["target_parquet_files"] = state["all_parquet_files"]
            state["use_union_by_name"] = True
            mode = "UNION (all files)"
        else:
            # Map logical tables to parquet paths
            target_files = []
            for table_name in required_tables:
                if table_name in state["global_catalog_dict"]:
                    target_files.append(state["global_catalog_dict"][table_name]["parquet_path"])
            
            state["target_parquet_files"] = target_files if target_files else state["all_parquet_files"]
            state["use_union_by_name"] = len(target_files) > 1 and not join_key
            mode = "UNION" if state["use_union_by_name"] else "JOIN"
        
        print(f"[SUCCESS] Tables: {required_tables} -> Files: {len(state['target_parquet_files'])} ({mode})")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Data source identification failed: {e}")
        state["error"] = f"Data source identification error: {str(e)}"
        # Fallback to all files
        state["target_parquet_files"] = state["all_parquet_files"]
        state["use_union_by_name"] = True
        return state


def load_query_context(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 4: Load schema and sample data for current query.
    Prepares context needed for routing and SQL generation.
    """
    print("\n" + "-"*80)
    print("NODE: Load Query Context")
    print("-"*80)
    
    try:
        schema, df_sample = get_parquet_context(
            parquet_paths=state["target_parquet_files"],
            use_union_by_name=state["use_union_by_name"],
            all_parquet_glob_pattern=state["config"].azure_storage.glob_pattern,
            config=state["config"]
        )
        
        state["parquet_schema"] = schema
        
        # Convert DataFrame sample to a serializable Markdown string
        state["df_sample"] = _df_to_markdown_sample(df_sample, max_rows=20)
        
        # Explicitly delete the DataFrame to free memory immediately after conversion
        del df_sample
        
        print(f"[SUCCESS] Schema loaded (Sample data: {len(state['df_sample'])} chars)")
        
        if state.get("enable_debug", False):
            print("\nSchema preview:")
            print(schema[:500] + "..." if len(schema) > 500 else schema)
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Context loading failed: {e}")
        state["error"] = f"Context loading error: {str(e)}"
        state["parquet_schema"] = ""
        state["df_sample"] = "No sample data due to error."
        return state


def route_query_intent(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    print("\n" + "-"*80)
    print("NODE: Route Query Intent")
    print("-"*80)
    
    try:
        import json
        
        ROUTER_SYSTEM_PROMPT = f"""
You are an intelligent routing agent. Your task is to classify the user's question intent based on the available database context and execution capabilities.

--- EXECUTION CONTEXT ---
SCHEMA (For understanding available columns):
{state["parquet_schema"]}

SAMPLE DATA (For understanding data values/types):
{state["df_sample"]}

--- CLASSIFICATION RULES ---
1. **SQL_QUERY**: Choose if the question requires:
   - Aggregations (SUM, AVG, COUNT, MAX, MIN, grouping)
   - Precise numerical filtering (e.g., WHERE amount > 100)
   - Date range filtering
   - Any query that can be answered entirely by direct structured SQL logic.

2. **SEMANTIC_SEARCH**: Choose if the question requires:
   - Conceptual lookups (e.g., 'What is OTC B2B Wall Units?')
   - Any filtering or matching based on unstructured text, descriptive fields, or names (e.g., 'find the closest match for the device ID 04SEP24...').
   - Handling typos or fuzzy matching on string columns.

**OUTPUT INSTRUCTIONS**:
Provide your classification as a single JSON object with the key "intent".
Ensure the output strictly follows the JSON format.

Example Output: {{"intent": "SEMANTIC_SEARCH"}} or {{"intent": "SQL_QUERY"}}
"""
        
        response = llm_client.chat.completions.create(
            model=state["config"].azure_openai.llm_deployment_name,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {state['current_query']}. Output the classification in JSON format."} 
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        json_output = json.loads(response.choices[0].message.content.strip())
        intent = json_output.get("intent", "SQL_QUERY").upper()
        
        state["intent"] = intent if intent in ['SEMANTIC_SEARCH', 'SQL_QUERY'] else 'SQL_QUERY'
        
        print(f"[SUCCESS] Intent: {state['intent']}")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Routing failed: {e}")
        state["intent"] = "SQL_QUERY"  # Default fallback
        return state

def perform_semantic_search(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 6a: Perform semantic search using vector database.
    Retrieves relevant context from ChromaDB.
    """
    print("\n" + "-"*80)
    print("NODE: Semantic Search")
    print("-"*80)
    
    try:
        # Prepare allowed collections based on required tables
        allowed_collections = None
        if state["required_tables"] and state["required_tables"] != ["*"]:
            allowed_collections = state["required_tables"]
        
        semantic_context, retrieved_files = get_semantic_context_and_files(
            query=state["current_query"],
            catalog=state["global_catalog_dict"],
            limit=10,
            score_threshold=0.5,
            allowed_collections=allowed_collections
        )
        
        state["semantic_context"] = semantic_context
        
        # Update target files if semantic search found more specific ones
        if retrieved_files:
            state["target_parquet_files"] = retrieved_files
            print(f"[INFO] Updated target files based on semantic search: {len(retrieved_files)} files")
            
            # Reload context with updated files
            schema, df_sample = get_parquet_context(
                parquet_paths=retrieved_files,
                use_union_by_name=True,
                all_parquet_glob_pattern=state["config"].azure_storage.glob_pattern,
                config=state["config"]
            )
            state["parquet_schema"] = schema
            
            # Convert DataFrame sample to a serializable Markdown string
            state["df_sample"] = _df_to_markdown_sample(df_sample, max_rows=20)
            
            # Explicitly delete the DataFrame to free memory immediately after conversion
            del df_sample

        print(f"[SUCCESS] Semantic context retrieved ({len(semantic_context)} chars)")
        print("\nSemantic context preview:\n")
        print(semantic_context)
        if state.get("enable_debug", False):
            print("\nSemantic context preview:")
            print(semantic_context[:300] + "..." if len(semantic_context) > 300 else semantic_context)
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Semantic search failed: {e}")
        state["semantic_context"] = ""
        return state


def generate_sql_query(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 7: Generate SQL query using LLM.
    Creates optimized SQL based on schema and context.
    """
    print("\n" + "-"*80)
    print("NODE: Generate SQL Query")
    print("-"*80)
    
    try:
        # Build augmentation hint based on intent
        augmentation_hint = ""
        if state["intent"] == "SEMANTIC_SEARCH" and state.get("semantic_context"):
            augmentation_hint = f"""
--- SEMANTIC CONTEXT (Vector Search Results) ---
{state["semantic_context"]}

CRITICAL: Use the above semantic context to:
1. Identify exact column values (names, IDs, categories) mentioned
2. Handle potential typos or variations in user input
3. Apply EXACT values from semantic context in WHERE clauses
"""
        
        # Generate CRITICAL path mapping for the LLM
        path_map = {}
        if state["required_tables"] and state["required_tables"] != ["*"]:
            for logical_name in state["required_tables"]:
                if logical_name in state["global_catalog_dict"]:
                    path_map[logical_name] = state["global_catalog_dict"][logical_name]['parquet_path']
        
        PATH_HINT = "\n--- FILE PATH MAPPING (CRITICAL) ---\n"
        if path_map:
             PATH_HINT += "When querying, you MUST use the full URI with read_parquet() or glob() to access the data. DO NOT use local paths or generic blob storage paths.\n"
             for logical_name, uri in path_map.items():
                 # Use os.path.basename() to extract the file name for clarity
                 file_name = os.path.basename(uri)
                 PATH_HINT += f"Logical Table '{logical_name}' (File: {file_name}) is located at: '{uri}'\n"
             # Example for LLM
             PATH_HINT += f"\nExample Query Structure:\nSELECT ... FROM read_parquet('{list(path_map.values())[0]}') WHERE ...\n"
        else:
            PATH_HINT = "\n--- FILE PATH MAPPING ---\nRun query against all files using a glob pattern if needed, but if a specific file is identified, use its full URI.\n"


        # Prepare SQL generation prompt
        sql_prompt = f"""
You are an expert SQL query generator for DuckDB working with Parquet files on Azure Blob Storage.

{augmentation_hint}

{PATH_HINT}

--- DATABASE SCHEMA ---
{state["parquet_schema"]}

--- SAMPLE DATA ---
{state["df_sample"]}

--- USER QUERY ---
{state["current_query"]}

--- INSTRUCTIONS ---
1. Generate a valid DuckDB SQL query.
2. CRITICAL: Use the FULL AZURE URI provided in the FILE PATH MAPPING with the read_parquet() function.
3. Use the EXACT table aliases and structure shown in the schema.
4. If semantic context is provided, use EXACT values from it (case-sensitive).
5. Include appropriate WHERE, GROUP BY, ORDER BY clauses.
6. Use aggregate functions (SUM, COUNT, AVG, etc.) where appropriate.
7. Handle NULL values appropriately.
8. Return results in a user-friendly format.

Your response MUST be valid JSON:
{{
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation of what the query does"
}}
"""
        
        response = llm_client.chat.completions.create(
            model=state["config"].azure_openai.llm_deployment_name,
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        
        state["sql_query"] = result.get("sql_query", "")
        state["sql_explanation"] = result.get("explanation", "")
        
        print(f"[SUCCESS] SQL query generated\n"+state["sql_query"]+"\n")
        print(f"Explanation: {state['sql_explanation']}")
        
        if state.get("enable_debug", False):
            print(f"\nGenerated SQL:\n{state['sql_query']}")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] SQL generation failed: {e}")
        state["error"] = f"SQL generation error: {str(e)}"
        state["sql_query"] = ""
        state["sql_explanation"] = ""
        return state


def execute_query(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 8: Execute the generated SQL query.
    Runs query against DuckDB and handles results.
    """
    print("\n" + "-"*80)
    print("NODE: Execute Query")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        if not state["sql_query"]:
            raise ValueError("No SQL query to execute")
        
        # result_df is a temporary DataFrame (non-serializable)
        result_df = execute_duckdb_query(
            query=state["sql_query"],
            config=state["config"]
        )
        
        state["execution_duration"] = time.time() - start_time
        
        # Convert temporary DataFrame to serializable JSON string
        serializable_result_str = _df_to_json_result(result_df)
        
        # Store serializable string in results dictionary
        state["results"][state["current_query"]] = serializable_result_str
        
        if 'Error' in result_df.columns:
            error_msg = result_df['Error'].iloc[0]
            print(f"[ERROR] Query execution error: {error_msg}")
            state["error"] = error_msg
        else:
            print(f"[SUCCESS] Query executed in {state['execution_duration']:.2f}s")
            print(f"Result shape: {result_df.shape}")
            state["error"] = None # Clear error on success
            
            if state.get("enable_debug", False) and not result_df.empty:
                print("\nResult preview:")
                print(result_df.head(5).to_markdown(index=False))

        # Explicitly delete the DataFrame to free memory
        del result_df
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Query execution failed: {e}")
        # Store error as a serializable JSON string
        state["results"][state["current_query"]] = json.dumps({'Error': str(e)})
        state["execution_duration"] = time.time() - start_time
        state["error"] = str(e)
        return state


def check_more_queries(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 9: Check if there are more sub-queries to process.
    Increments query index and determines if loop should continue.
    """
    state["query_index"] += 1
    
    remaining = len(state["sub_queries"]) - state["query_index"]
    print(f"\n[INFO] Query {state['query_index']}/{len(state['sub_queries'])} complete. Remaining: {remaining}")
    
    return state


def generate_summary(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Node 10: Generate natural language summary of all results.
    Synthesizes findings into user-friendly response.
    """
    print("\n" + "-"*80)
    print("NODE: Generate Summary")
    print("-"*80)
    
    try:
        if not state["results"]:
            state["summary"] = "No results to summarize."
            return state
        
        # Convert serializable results (JSON strings) back to DataFrames for the summary utility
        summarizer_results: Dict[str, pd.DataFrame] = {}
        for query, json_str in state["results"].items():
            summarizer_results[query] = _json_to_df(json_str)

        summary = generate_simple_summary(
            results=summarizer_results, # Use the temporary DataFrames
            llm_client=llm_client,
            deployment_name=state["config"].azure_openai.llm_deployment_name
        )
        
        state["summary"] = summary
        
        # Explicitly delete temporary DataFrames
        del summarizer_results
        
        print(f"[SUCCESS] Summary generated")
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Summary generation failed: {e}")
        state["summary"] = f"Failed to generate summary: {str(e)}"
        return state


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue_processing(state: GraphState) -> Literal["identify_sources", "generate_summary"]:
    """
    Determines if there are more queries to process.
    """
    if state["query_index"] < len(state["sub_queries"]):
        return "identify_sources"
    else:
        return "generate_summary"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_rag_graph(llm_client: AzureOpenAI) -> StateGraph:
    """
    Constructs the LangGraph RAG pipeline.
    
    Graph Flow:
    1. initialize → decompose
    2. decompose → identify_sources
    3. identify_sources → load_context
    4. load_context → route_intent
    5. route_intent → semantic_search (if SEMANTIC_SEARCH) or generate_sql
    6. semantic_search → generate_sql
    7. generate_sql → execute
    8. execute → check_more
    9. check_more → identify_sources (if more queries) or generate_summary
    10. generate_summary → END
    """
    
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Create wrapper functions that inject llm_client
    def initialize_with_client(state: GraphState) -> GraphState:
        return initialize_context(state, llm_client)
    
    def decompose_with_client(state: GraphState) -> GraphState:
        return decompose_query(state, llm_client)
    
    def identify_with_client(state: GraphState) -> GraphState:
        return identify_data_sources(state, llm_client)
    
    def load_with_client(state: GraphState) -> GraphState:
        return load_query_context(state, llm_client)
    
    def route_with_client(state: GraphState) -> GraphState:
        return route_query_intent(state, llm_client)
    
    def semantic_with_client(state: GraphState) -> GraphState:
        return perform_semantic_search(state, llm_client)
    
    def generate_sql_with_client(state: GraphState) -> GraphState:
        return generate_sql_query(state, llm_client)
    
    def execute_with_client(state: GraphState) -> GraphState:
        return execute_query(state, llm_client)
    
    def check_with_client(state: GraphState) -> GraphState:
        return check_more_queries(state, llm_client)
    
    def summary_with_client(state: GraphState) -> GraphState:
        return generate_summary(state, llm_client)
    
    # Add nodes
    workflow.add_node("initialize", initialize_with_client)
    workflow.add_node("decompose", decompose_with_client)
    workflow.add_node("identify_sources", identify_with_client)
    workflow.add_node("load_context", load_with_client)
    workflow.add_node("route_intent", route_with_client)
    workflow.add_node("semantic_search", semantic_with_client)
    workflow.add_node("generate_sql", generate_sql_with_client)
    workflow.add_node("execute", execute_with_client)
    workflow.add_node("check_more", check_with_client)
    workflow.add_node("generate_summary", summary_with_client)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    workflow.add_edge("initialize", "decompose")
    workflow.add_edge("decompose", "identify_sources")
    workflow.add_edge("identify_sources", "load_context")
    workflow.add_edge("load_context", "route_intent")
    
    # Conditional edge based on intent
    workflow.add_conditional_edges(
        "route_intent",
        lambda state: "semantic_search" if state["intent"] == "SEMANTIC_SEARCH" else "generate_sql",
        {
            "semantic_search": "semantic_search",
            "generate_sql": "generate_sql"
        }
    )
    
    workflow.add_edge("semantic_search", "generate_sql")
    workflow.add_edge("generate_sql", "execute")
    workflow.add_edge("execute", "check_more")
    
    # Conditional edge for loop or summary
    workflow.add_conditional_edges(
        "check_more",
        should_continue_processing,
        {
            "identify_sources": "identify_sources",
            "generate_summary": "generate_summary"
        }
    )
    
    workflow.add_edge("generate_summary", END)
    
    return workflow


# MAIN EXECUTION INTERFACE
def run_rag_pipeline(
    user_question: str,
    all_parquet_files: List[str],
    global_catalog_string: str,
    global_catalog_dict: Dict,
    config: Config,
    enable_debug: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Main interface to run the RAG pipeline using LangGraph.
    
    Args:
        user_question: User's natural language query
        all_parquet_files: List of all available parquet file paths
        global_catalog_string: String representation of data catalog
        global_catalog_dict: Dictionary representation of data catalog
        config: Application configuration
        enable_debug: Enable debug output
    
    Returns:
        Dictionary mapping queries to their result DataFrames (deserialized for output)
    """
    print("\n" + "#"*80)
    print("### LANGGRAPH RAG PIPELINE EXECUTION ###")
    print("#"*80)
    print(f"\nUser Question: {user_question}")
    
    # Initialize LLM client OUTSIDE the state
    llm_client = setup_llm_client(config)
    
    # Create graph and pass the client
    workflow = create_rag_graph(llm_client)
    
    # Compile WITHOUT checkpointer to avoid serialization issues
    app = workflow.compile()
    
    # Initialize state WITHOUT llm_client
    initial_state: GraphState = {
        "user_question": user_question,
        "all_parquet_files": all_parquet_files,
        "global_catalog_string": global_catalog_string,
        "global_catalog_dict": global_catalog_dict,
        "config": config,
        "enable_debug": enable_debug,
        
        # These will be set during execution
        "sub_queries": [],
        "current_query": None,
        "query_index": 0,
        "required_tables": [],
        "join_key": "",
        "target_parquet_files": [],
        "use_union_by_name": False,
        "parquet_schema": "",
        "df_sample": "", 
        "intent": "",
        "semantic_context": "",
        "sql_query": "",
        "sql_explanation": "",
        "execution_duration": 0.0,
        "results": {}, 
        "summary": "",
        "error": None
    }
    
    # Execute graph
    start_time = time.time()
    
    # Final result container (will map query -> DataFrame)
    final_results_df: Dict[str, pd.DataFrame] = {}
    
    try:
        # Run the graph
        final_state = app.invoke(initial_state)
        
        total_duration = time.time() - start_time
        
        print("\n" + "#"*80)
        print(f"### PIPELINE COMPLETE in {total_duration:.2f}s ###")
        print("#"*80)
        
        # Display summary
        if final_state.get("summary"):
            print("\n" + "-"*80)
            print("SUMMARY")
            print("-"*80)
            print(final_state["summary"])
            print("-"*80)

        # Deserialize final results for return value
        for query, json_str in final_state.get("results", {}).items():
            final_results_df[query] = _json_to_df(json_str)
        
        return final_results_df
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {}



if __name__ == "__main__":
    """
    Example usage of the LangGraph RAG pipeline.
    This demonstrates how to integrate with existing code.
    """
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Initialize configuration
    config = get_config(VectorDBType.CHROMADB)
    
    # Setup DuckDB connection
    try:
        setup_duckdb_azure_connection(config)
        import atexit
        atexit.register(close_persistent_duckdb_connection)
    except Exception as e:
        print(f"[FATAL] Initial DuckDB authentication failed: {e}")
        sys.exit(1)
    
    print("\n--- LangGraph RAG Pipeline Demo ---")
    
    # Process input files to parquet
    all_parquet_files = []
    
    print(f"\n[INFO] Processing {len(config.input_file_paths)} input files...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                convert_excel_to_parquet,
                excel_path,
                config.azure_storage.parquet_output_dir,
                config
            ): excel_path for excel_path in config.input_file_paths
            if os.path.exists(excel_path)
        }
        
        for future in as_completed(future_to_file):
            excel_path = future_to_file[future]
            try:
                sheet_parquet_paths = future.result()
                all_parquet_files.extend(sheet_parquet_paths)
                print(f"[SUCCESS] Processed {os.path.basename(excel_path)}")
            except Exception as exc:
                print(f"[ERROR] Failed to process {excel_path}: {exc}")
    
    if not all_parquet_files:
        print("[CRITICAL] No data files processed. Exiting.")
        sys.exit(1)
    
    # Build global catalog
    global_catalog_string, global_catalog_dict = build_global_catalog(all_parquet_files, config)
    
    print("\n" + "-"*50)
    print("GLOBAL DATA CATALOG")
    print("-"*50)
    print(global_catalog_string)
    print("-"*50)
    
    # Example queries
    queries = [
        "top 5 people with max loan?"
        # "give details of Connor Walts"
        # "What were the volumes for Canada Kit in each month from January to June?"
    ]
    
    # Run pipeline for each query
    for query in queries:
        print("\n\n" + "-"*80)
        
        results = run_rag_pipeline(
            user_question=query,
            all_parquet_files=all_parquet_files,
            global_catalog_string=global_catalog_string,
            global_catalog_dict=global_catalog_dict,
            config=config,
            enable_debug=False
        )
        
        print("\n" + "-"*80)
    
    print("\n--- Pipeline Executed ---")