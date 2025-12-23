"""
Multi-Intent Query Processing Pipeline
Handles both SQL_QUERY and SUMMARY_SEARCH intents with parallel/sequential execution
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from openai import AzureOpenAI

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class QueryAnalysis:
    """Structured output for query analysis"""

    sub_question: str
    intent: str  # "SQL_QUERY" or "SUMMARY_SEARCH"
    required_files: List[str]
    join_key: str = ""
    depends_on_index: int = -1  # -1 means independent, otherwise index of dependency


@dataclass
class AnalysisResult:
    """Complete analysis result"""

    total_questions: int
    analyses: List[QueryAnalysis]


@dataclass
class GraphState:
    """State object passed through the execution graph"""

    config: Any
    current_query: str
    intent: str
    required_tables: List[str]
    global_catalog_dict: Dict[str, Any]
    parquet_schema: str
    df_sample: str
    semantic_context: str
    sql_query: str
    sql_explanation: str
    results: Dict[str, str]
    execution_duration: float
    error: Optional[str]
    enable_debug: bool


# ============================================================================
# DUCKDB EXECUTION
# ============================================================================


def execute_duckdb_query(query: str, config: Any) -> pd.DataFrame:
    """
    Execute DuckDB query against Azure Blob Storage.

    Args:
        query: SQL query to execute
        config: Configuration object with Azure credentials

    Returns:
        DataFrame with query results or error
    """
    conn = duckdb.connect()
    try:
        # Install and load Azure extension
        conn.execute("INSTALL azure;")
        conn.execute("LOAD azure;")

        # Set Azure connection string
        escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")

        # Execute the query
        result_df = conn.execute(query).fetchdf()

        return result_df

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] DuckDB execution failed: {error_msg}")
        return pd.DataFrame({"Error": [error_msg]})
    finally:
        conn.close()


# ============================================================================
# QUERY ANALYSIS
# ============================================================================


def analyze_user_query(
    llm_client: AzureOpenAI,
    user_question: str,
    deployment_name: str,
    catalog_schema: str,
) -> AnalysisResult:
    """
    Single unified LLM call to:
    1. Decompose multi-intent queries
    2. Detect intent for each sub-question
    3. Identify required files and join keys
    4. Detect dependencies between sub-questions
    """

    UNIFIED_TOOL_SPEC = {
        "type": "function",
        "function": {
            "name": "analyze_query",
            "description": "Analyze user query and decompose into structured sub-questions with intent and file requirements",
            "parameters": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "description": "List of analyzed sub-questions. Each question must be independent unless explicitly dependent.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sub_question": {
                                    "type": "string",
                                    "description": "A single, atomic question extracted from user query",
                                },
                                "intent": {
                                    "type": "string",
                                    "enum": ["SQL_QUERY", "SUMMARY_SEARCH"],
                                    "description": "SQL_QUERY for aggregations/filters/calculations on structured data (DuckDB). SUMMARY_SEARCH for fuzzy matching/conceptual lookups/hybrid RAG/GraphDB.",
                                },
                                "required_files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of EXACT file names from the catalog (e.g., ['housing.parquet', 'loan.parquet']). Use ['*'] ONLY if truly all files are needed.",
                                },
                                "join_key": {
                                    "type": "string",
                                    "description": "Common column name if multiple files need to be joined (e.g., 'loan_application_id'). Empty string if single file or no join needed.",
                                },
                                "depends_on_index": {
                                    "type": "integer",
                                    "description": "Index (0-based) of the sub-question this depends on. Use -1 if independent. Example: If Q2 needs results from Q1, set to 0.",
                                },
                            },
                            "required": ["sub_question", "intent", "required_files"],
                        },
                    }
                },
                "required": ["analyses"],
            },
        },
    }

    SYSTEM_PROMPT = f"""
    You are an expert query analyzer. Analyze the user's question and provide structured analysis.

    --- AVAILABLE DATA CATALOG ---
    {catalog_schema}
    --- END CATALOG ---

    **DECOMPOSITION RULES:**
    1. Break into MULTIPLE sub-questions ONLY if they are atomically independent OR one depends on another's result
    2. Keep as SINGLE question if all parts share the same filter/aggregation
    3. Examples of decomposition:
    - "What's max income AND status for Smith?" → 2 questions (independent topics)
    - "Find auto loans with status and interest rates" → 1 question (single filter applied)

    **INTENT CLASSIFICATION:**
    - SQL_QUERY: Use if the request can be fulfilled entirely by precise filtering, aggregation (SUM/AVG/COUNT/MAX/MIN), date ranges, or numerical comparisons on the structured data (DuckDB).
    - SUMMARY_SEARCH: Use if the request requires fuzzy logic, conceptual search, or information that would be sourced from hybrid RAG, GraphDB, or document systems.

    **FILE IDENTIFICATION (CRITICAL):**
    1. Look at the catalog and identify EXACT file names (e.g., "housing.parquet", "loan.parquet")
    2. Include ONLY the specific files that contain the columns needed for the query
    3. Use the EXACT file name as it appears in the catalog
    4. If columns span multiple files, list ALL relevant files and specify join_key
    5. Never use wildcards or generic names - use actual file names from the catalog
    6. Use ['*'] ONLY if the query genuinely needs ALL files in the catalog

    **DEPENDENCY DETECTION:**
    - Set depends_on_index to the index of the question whose result is needed
    - Example: "Find highest income applicant, then get their loan status" → Q2 depends on Q1 (index 0)

    Return comprehensive analysis for all aspects of the query.
    """

    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this query: {user_question}"},
            ],
            tools=[UNIFIED_TOOL_SPEC],
            tool_choice={"type": "function", "function": {"name": "analyze_query"}},
            temperature=0.0,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        analyses = [
            QueryAnalysis(
                sub_question=item["sub_question"],
                intent=item["intent"],
                required_files=item["required_files"],
                join_key=item.get("join_key", ""),
                depends_on_index=item.get("depends_on_index", -1),
            )
            for item in args["analyses"]
        ]

        result = AnalysisResult(total_questions=len(analyses), analyses=analyses)

        print("\n" + "=" * 80)
        print("QUERY ANALYSIS RESULT")
        print("=" * 80)
        print(f"Total Questions: {result.total_questions}")
        print("\n")

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Total Tokens: {total_tokens}")

        for idx, analysis in enumerate(result.analyses):
            print(f"--- Question {idx + 1} ---")
            print(f"Sub-Question: {analysis.sub_question}")
            print(f"Intent: {analysis.intent}")
            print(f"Required Files: {', '.join(analysis.required_files)}")
            if len(analysis.required_files) > 1 and analysis.join_key:
                print(f"Join Method: JOIN ON {analysis.join_key}")
            if analysis.depends_on_index >= 0:
                print(
                    f"⚠️  Depends on Question {analysis.depends_on_index + 1} (must execute after)"
                )
            print()

        return result

    except Exception as e:
        print(f"[ERROR] Query analysis failed: {e}")
        # Fallback to treating as single query
        return AnalysisResult(
            total_questions=1,
            analyses=[
                QueryAnalysis(
                    sub_question=user_question,
                    intent="SQL_QUERY",
                    required_files=["*"],
                    join_key="",
                    depends_on_index=-1,
                )
            ],
        )


# ============================================================================
# SQL QUERY AGENT
# ============================================================================


def generate_sql_query(state: GraphState, llm_client: AzureOpenAI) -> GraphState:
    """
    Generate SQL query using LLM.
    Creates optimized SQL based on schema and context.
    """
    print("\n" + "-" * 80)
    print("NODE: Generate SQL Query")
    print("-" * 80)

    try:
        # Build augmentation hint based on intent
        augmentation_hint = ""
        if state.intent == "SQL_QUERY" and state.semantic_context:
            augmentation_hint = f"""
--- SEMANTIC CONTEXT (Vector Search Results) ---
{state.semantic_context}

CRITICAL: Use the above semantic context to:
1. Identify exact column values (names, IDs, categories) mentioned
2. Handle potential typos or variations in user input
3. Apply EXACT values from semantic context in WHERE clauses
"""

        # Generate CRITICAL path mapping for the LLM
        path_map = {}
        if state.required_tables and state.required_tables != ["*"]:
            for file_name in state.required_tables:
                # Look up the file in the catalog
                for logical_name, file_info in state.global_catalog_dict.items():
                    if (
                        file_info.get("file_name") == file_name
                        or logical_name == file_name
                    ):
                        path_map[file_name] = file_info.get(
                            "parquet_path"
                        ) or file_info.get("azure_uri")
                        break

        PATH_HINT = "\n--- FILE PATH MAPPING (CRITICAL) ---\n"
        if path_map:
            PATH_HINT += "When querying, you MUST use the EXACT full Azure URI with read_parquet() function.\n"
            PATH_HINT += "DO NOT use placeholders like '<your-azure-blob-storage-path>' or wildcards.\n"
            PATH_HINT += "Use the EXACT URIs provided below:\n\n"
            for file_name, uri in path_map.items():
                PATH_HINT += f"File '{file_name}' → '{uri}'\n"

            if len(path_map) == 1:
                PATH_HINT += f"\nExample Query:\nSELECT * FROM read_parquet('{list(path_map.values())[0]}') WHERE ...\n"
            else:
                uris = list(path_map.values())
                PATH_HINT += f"\nExample Query with JOIN:\nSELECT * FROM read_parquet('{uris[0]}') AS t1 JOIN read_parquet('{uris[1]}') AS t2 ON ...\n"
        else:
            PATH_HINT = "\n--- FILE PATH MAPPING ---\nNo specific files identified. Query all available files if needed.\n"

        # Prepare SQL generation prompt
        sql_prompt = f"""
You are an expert SQL query generator for DuckDB working with Parquet files on Azure Blob Storage.

{augmentation_hint}

{PATH_HINT}

--- DATABASE SCHEMA ---
{state.parquet_schema}

--- SAMPLE DATA ---
{state.df_sample}

--- USER QUERY ---
{state.current_query}

--- CRITICAL INSTRUCTIONS ---
1. Generate a valid DuckDB SQL query.
2. **MANDATORY**: Use the EXACT full Azure URI provided in FILE PATH MAPPING with read_parquet() function.
3. **NEVER** use placeholders like '<your-azure-blob-storage-path>' or wildcards like '*.parquet'.
4. **ALWAYS** use the complete URI shown in the mapping (e.g., 'azure://account.blob.core.windows.net/container/file.parquet').
5. Use the EXACT table aliases and structure shown in the schema.
6. If semantic context is provided, use EXACT values from it (case-sensitive).
7. Include appropriate WHERE, GROUP BY, ORDER BY clauses.
8. Use aggregate functions (SUM, COUNT, AVG, etc.) where appropriate.
9. Handle NULL values appropriately.
10. Return results in a user-friendly format.

Your response MUST be valid JSON:
{{
    "sql_query": "SELECT ...",
    "explanation": "Brief explanation of what the query does"
}}
"""

        response = llm_client.chat.completions.create(
            model=state.config.azure_openai.llm_deployment_name,
            messages=[{"role": "user", "content": sql_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content.strip())

        state.sql_query = result.get("sql_query", "")
        state.sql_explanation = result.get("explanation", "")

        print(f"[SUCCESS] SQL query generated\n{state.sql_query}\n")
        print(f"Explanation: {state.sql_explanation}")

        if state.enable_debug:
            print(f"\nGenerated SQL:\n{state.sql_query}")

        return state

    except Exception as e:
        print(f"[ERROR] SQL generation failed: {e}")
        state.error = f"SQL generation error: {str(e)}"
        state.sql_query = ""
        state.sql_explanation = ""
        return state


def execute_query(
    state: GraphState, llm_client: AzureOpenAI, execute_duckdb_query_fn
) -> GraphState:
    """
    Execute the generated SQL query.
    Runs query against DuckDB and handles results.
    """
    print("\n" + "-" * 80)
    print("NODE: Execute Query")
    print("-" * 80)

    start_time = time.time()

    try:
        if not state.sql_query:
            raise ValueError("No SQL query to execute")

        # Execute query using provided function
        result_df = execute_duckdb_query_fn(query=state.sql_query, config=state.config)

        state.execution_duration = time.time() - start_time

        # Convert DataFrame to serializable JSON string
        serializable_result_str = _df_to_json_result(result_df)

        # Store serializable string in results dictionary
        state.results[state.current_query] = serializable_result_str

        if "Error" in result_df.columns:
            error_msg = result_df["Error"].iloc[0]
            print(f"[ERROR] Query execution error: {error_msg}")
            state.error = error_msg
        else:
            print(f"[SUCCESS] Query executed in {state.execution_duration:.2f}s")
            print(f"Result shape: {result_df.shape}")
            state.error = None

            if state.enable_debug and not result_df.empty:
                print("\nResult preview:")
                print(result_df.head(5).to_markdown(index=False))

        del result_df

        return state

    except Exception as e:
        print(f"[ERROR] Query execution failed: {e}")
        state.results[state.current_query] = json.dumps({"Error": str(e)})
        state.execution_duration = time.time() - start_time
        state.error = str(e)
        return state


def _df_to_json_result(df) -> str:
    """Helper function to convert DataFrame to JSON string"""
    try:
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        return json.dumps({"Error": f"Failed to serialize results: {str(e)}"})


def process_sql_query(
    analysis: QueryAnalysis,
    config: Any,
    llm_client: AzureOpenAI,
    global_catalog_dict: Dict[str, Any],
    execute_duckdb_query_fn,
    enable_debug: bool = False,
    dependency_result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete SQL query processing pipeline for a single query.

    Args:
        analysis: QueryAnalysis object containing query details
        config: Configuration object
        llm_client: Azure OpenAI client
        global_catalog_dict: Dictionary containing catalog information
        execute_duckdb_query_fn: Function to execute DuckDB queries
        enable_debug: Enable debug output
        dependency_result: Result from dependent query (if any)

    Returns:
        Dictionary containing query results and metadata
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING SQL QUERY: {analysis.sub_question}")
    print(f"{'='*80}")

    # Build schema and sample data from catalog
    parquet_schema = ""
    df_sample = ""

    if analysis.required_files and analysis.required_files != ["*"]:
        for file_name in analysis.required_files:
            for logical_name, file_info in global_catalog_dict.items():
                if file_info.get("file_name") == file_name or logical_name == file_name:
                    # Extract schema info
                    if "columns" in file_info:
                        parquet_schema += f"\n--- {file_name} Schema ---\n"
                        for col in file_info["columns"]:
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", "unknown")
                            parquet_schema += f"  - {col_name}: {col_type}\n"

                    # Add sample data info if available
                    if "row_count" in file_info:
                        df_sample += f"\n{file_name}: {file_info['row_count']} rows\n"

                    break

    # Initialize state
    state = GraphState(
        config=config,
        current_query=analysis.sub_question,
        intent=analysis.intent,
        required_tables=analysis.required_files,
        global_catalog_dict=global_catalog_dict,
        parquet_schema=parquet_schema,
        df_sample=df_sample,
        semantic_context=dependency_result if dependency_result else "",
        sql_query="",
        sql_explanation="",
        results={},
        execution_duration=0.0,
        error=None,
        enable_debug=enable_debug,
    )

    # Step 1: Generate SQL query
    state = generate_sql_query(state, llm_client)

    if state.error:
        return {
            "sub_question": analysis.sub_question,
            "intent": analysis.intent,
            "status": "error",
            "error": state.error,
            "execution_duration": 0.0,
        }

    # Step 2: Execute query
    state = execute_query(state, llm_client, execute_duckdb_query_fn)

    return {
        "sub_question": analysis.sub_question,
        "intent": analysis.intent,
        "status": "success" if not state.error else "error",
        "sql_query": state.sql_query,
        "sql_explanation": state.sql_explanation,
        "results": state.results.get(state.current_query, ""),
        "execution_duration": state.execution_duration,
        "error": state.error,
    }


# ============================================================================
# SUMMARY SEARCH AGENT (PLACEHOLDER)
# ============================================================================


def process_summary_search(
    analysis: QueryAnalysis,
    config: Any,
    llm_client: AzureOpenAI,
    global_catalog_dict: Dict[str, Any],
    enable_debug: bool = False,
    dependency_result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Placeholder function for SUMMARY_SEARCH intent.
    This will be implemented with RAG/GraphDB/Vector search logic.

    Args:
        analysis: QueryAnalysis object containing query details
        config: Configuration object
        llm_client: Azure OpenAI client
        global_catalog_dict: Dictionary containing catalog information
        enable_debug: Enable debug output
        dependency_result: Result from dependent query (if any)

    Returns:
        Dictionary containing search results and metadata
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY SEARCH: {analysis.sub_question}")
    print(f"{'='*80}")
    print("[PLACEHOLDER] Summary search not yet implemented")

    # TODO: Implement actual summary search logic
    # This would include:
    # - Vector similarity search
    # - Graph database queries
    # - RAG pipeline
    # - Hybrid search approaches

    return {
        "sub_question": analysis.sub_question,
        "intent": analysis.intent,
        "status": "placeholder",
        "results": json.dumps(
            {
                "message": "Summary search functionality will be implemented here",
                "query": analysis.sub_question,
                "required_files": analysis.required_files,
            }
        ),
        "execution_duration": 0.0,
        "error": None,
    }


# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================


def execute_independent_queries(
    analyses: List[QueryAnalysis],
    config: Any,
    llm_client: AzureOpenAI,
    global_catalog_dict: Dict[str, Any],
    execute_duckdb_query_fn,
    enable_debug: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Execute multiple independent queries in parallel.

    Args:
        analyses: List of tuples (index, QueryAnalysis) for independent queries
        config: Configuration object
        llm_client: Azure OpenAI client
        global_catalog_dict: Dictionary containing catalog information
        execute_duckdb_query_fn: Function to execute DuckDB queries
        enable_debug: Enable debug output

    Returns:
        Dictionary mapping original indices to results
    """
    print(f"\n{'#'*80}")
    print(f"PARALLEL EXECUTION: {len(analyses)} independent queries")
    print(f"{'#'*80}")

    results = {}

    with ThreadPoolExecutor(max_workers=min(len(analyses), 4)) as executor:
        future_to_info = {}

        for idx, analysis in analyses:
            if analysis.intent == "SQL_QUERY":
                future = executor.submit(
                    process_sql_query,
                    analysis,
                    config,
                    llm_client,
                    global_catalog_dict,
                    execute_duckdb_query_fn,
                    enable_debug,
                )
            else:  # SUMMARY_SEARCH
                future = executor.submit(
                    process_summary_search,
                    analysis,
                    config,
                    llm_client,
                    global_catalog_dict,
                    enable_debug,
                )

            future_to_info[future] = (idx, analysis)

        for future in as_completed(future_to_info):
            idx, analysis = future_to_info[future]
            try:
                result = future.result()
                results[idx] = result
                print(f"\n✓ Completed: {analysis.sub_question}")
            except Exception as e:
                print(f"\n✗ Failed: {analysis.sub_question} - {str(e)}")
                results[idx] = {
                    "sub_question": analysis.sub_question,
                    "intent": analysis.intent,
                    "status": "error",
                    "error": str(e),
                    "execution_duration": 0.0,
                }

    return results


def execute_all_queries(
    analyses: List[QueryAnalysis],
    config: Any,
    llm_client: AzureOpenAI,
    global_catalog_dict: Dict[str, Any],
    execute_duckdb_query_fn,
    enable_debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute all queries respecting dependencies.
    Independent queries run in parallel, dependent queries run sequentially.

    Args:
        analyses: List of QueryAnalysis objects
        config: Configuration object
        llm_client: Azure OpenAI client
        global_catalog_dict: Dictionary containing catalog information
        execute_duckdb_query_fn: Function to execute DuckDB queries
        enable_debug: Enable debug output

    Returns:
        List of results in original order
    """
    results = [None] * len(analyses)
    executed_results = {}

    # Build dependency graph
    dependencies = {}
    for idx, analysis in enumerate(analyses):
        dependencies[idx] = analysis.depends_on_index

    # Process queries in waves
    remaining = set(range(len(analyses)))

    while remaining:
        # Find queries that can be executed now (dependencies satisfied)
        ready = []
        for idx in remaining:
            dep_idx = dependencies[idx]
            if dep_idx == -1 or dep_idx in executed_results:
                ready.append((idx, analyses[idx]))

        if not ready:
            # Circular dependency or invalid dependency chain
            print("\n[ERROR] Circular dependency or invalid dependency chain detected!")
            for idx in remaining:
                results[idx] = {
                    "sub_question": analyses[idx].sub_question,
                    "intent": analyses[idx].intent,
                    "status": "error",
                    "error": "Circular dependency or invalid dependency chain",
                    "execution_duration": 0.0,
                }
            break

        # Execute all ready queries in parallel
        if len(ready) > 1:
            # Parallel execution
            batch_results = execute_independent_queries(
                ready,
                config,
                llm_client,
                global_catalog_dict,
                execute_duckdb_query_fn,
                enable_debug,
            )
        else:
            # Single query execution
            idx, analysis = ready[0]
            dependency_result = None

            if analysis.depends_on_index >= 0:
                dep_result = executed_results.get(analysis.depends_on_index)
                if dep_result and dep_result.get("status") == "success":
                    dependency_result = dep_result.get("results", "")
                    print(
                        f"\n→ Query {idx + 1} depends on Query {analysis.depends_on_index + 1}"
                    )

            try:
                if analysis.intent == "SQL_QUERY":
                    result = process_sql_query(
                        analysis,
                        config,
                        llm_client,
                        global_catalog_dict,
                        execute_duckdb_query_fn,
                        enable_debug,
                        dependency_result,
                    )
                else:  # SUMMARY_SEARCH
                    result = process_summary_search(
                        analysis,
                        config,
                        llm_client,
                        global_catalog_dict,
                        enable_debug,
                        dependency_result,
                    )

                batch_results = {idx: result}
                print(f"\n✓ Completed: {analysis.sub_question}")

            except Exception as e:
                print(f"\n✗ Failed: {analysis.sub_question} - {str(e)}")
                batch_results = {
                    idx: {
                        "sub_question": analysis.sub_question,
                        "intent": analysis.intent,
                        "status": "error",
                        "error": str(e),
                        "execution_duration": 0.0,
                    }
                }

        # Store results and update remaining
        for idx, result in batch_results.items():
            results[idx] = result
            executed_results[idx] = result
            remaining.discard(idx)

    return results


def process_multi_intent_query(
    user_question: str,
    config: Any,
    llm_client: AzureOpenAI,
    global_catalog_dict: Dict[str, Any],
    catalog_schema: str,
    execute_duckdb_query_fn,
    enable_debug: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestration function for multi-intent query processing.

    Args:
        user_question: Original user question
        config: Configuration object
        llm_client: Azure OpenAI client
        global_catalog_dict: Dictionary containing catalog information
        catalog_schema: JSON string of catalog schema
        execute_duckdb_query_fn: Function to execute DuckDB queries
        enable_debug: Enable debug output

    Returns:
        Dictionary containing all results and metadata
    """
    start_time = time.time()

    # Step 1: Analyze query
    analysis_result = analyze_user_query(
        llm_client,
        user_question,
        config.azure_openai.llm_deployment_name,
        catalog_schema,
    )

    # Step 2: Count independent and dependent queries
    independent_count = sum(
        1 for a in analysis_result.analyses if a.depends_on_index == -1
    )
    dependent_count = len(analysis_result.analyses) - independent_count

    # Step 3: Execute all queries (handles both independent and dependent)
    all_results = execute_all_queries(
        analysis_result.analyses,
        config,
        llm_client,
        global_catalog_dict,
        execute_duckdb_query_fn,
        enable_debug,
    )

    total_duration = time.time() - start_time

    return {
        "original_question": user_question,
        "total_questions": analysis_result.total_questions,
        "independent_count": independent_count,
        "dependent_count": dependent_count,
        "results": all_results,
        "total_duration": total_duration,
        "status": (
            "success"
            if all(r and r.get("status") != "error" for r in all_results)
            else "partial_success"
        ),
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def main():
    """Example usage of the multi-intent query processing pipeline"""
    from config import VectorDBType, get_config

    # Load actual configuration
    config = get_config(VectorDBType.CHROMADB)

    llm_client = AzureOpenAI(
        api_key=config.azure_openai.llm_api_key,
        azure_endpoint=config.azure_openai.llm_endpoint,
        api_version=config.azure_openai.llm_api_version,
    )

    # Load catalog schema
    CATALOG_FILE = "x.json"
    try:
        with open(CATALOG_FILE, "r") as f:
            catalog_data = json.load(f)
            catalog_schema = json.dumps(catalog_data, indent=2)

            # Build global catalog dict from the catalog data
            global_catalog_dict = {}
            if isinstance(catalog_data, list):
                for item in catalog_data:
                    file_name = item.get("file_name", "")
                    if file_name:
                        global_catalog_dict[file_name] = item
            elif isinstance(catalog_data, dict):
                global_catalog_dict = catalog_data

    except Exception as e:
        print(f"Error loading catalog: {e}")
        return

    # Test queries
    test_queries = [
        # "What is the maximum loan amount and what is the status for applicant Smith?",
        # "Find all auto loans with their approval status and interest rates",
        # "Get the highest income applicant, then show me their loan details",
        "explain all files",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {query}")
        print(f"{'='*80}")

        result = process_multi_intent_query(
            user_question=query,
            config=config,
            llm_client=llm_client,
            global_catalog_dict=global_catalog_dict,
            catalog_schema=catalog_schema,
            execute_duckdb_query_fn=execute_duckdb_query,
            enable_debug=True,
        )

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
