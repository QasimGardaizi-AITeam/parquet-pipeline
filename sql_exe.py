import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from openai import AzureOpenAI


def _estimate_tokens_for_payload(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
) -> int:
    """
    Estimates the number of tokens in the prompt payload (messages + tools).
    (Uses character-to-token ratio plus fixed overheads as a substitute for tiktoken.)
    """

    # 1. Messages Content Length
    messages_char_count = 0
    for message in messages:
        messages_char_count += len(message.get("content", ""))

    # 2. Tool Specification Length
    # Serializing the tool specification compactly.
    tool_json_string = json.dumps(tools[0]["function"], separators=(",", ":"))
    tool_char_count = len(tool_json_string)

    # 3. Fixed Overheads (Approximate based on GPT/OpenAI API structure)
    fixed_overhead = 25 + (
        len(messages) * 3
    )  # 25 for tool structure + 3 tokens per message

    # Standard estimate: 4 characters per token
    estimated_token_count = (
        round((messages_char_count + tool_char_count) / 4) + fixed_overhead
    )

    return max(estimated_token_count, 1)


@dataclass
class QueryAnalysis:
    """Structured output for query analysis"""

    sub_question: str
    intent: str  # "SQL_QUERY" or "SEMANTIC_SEARCH"
    required_files: List[str]
    join_key: str = ""
    depends_on_index: int = -1  # -1 means independent, otherwise index of dependency


@dataclass
class AnalysisResult:
    """Complete analysis result"""

    total_questions: int
    analyses: List[QueryAnalysis]


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
                                    "description": "List of file names (e.g., ['loan.parquet']) needed. Use ['*'] if all files needed.",
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

    **FILE IDENTIFICATION:**
    1. Identify ALL columns needed (SELECT, WHERE, GROUP BY, ORDER BY)
    2. Map each column to its source file(s) from the catalog
    3. Include ONLY files containing required columns
    4. If columns span multiple files, list ALL and specify join_key
    5. Never join unnamed columns with other unnamed columns

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
        print(f"Total Independent Questions: {result.total_questions}")
        print("\n")

        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            print(f"*** Actual Prompt Tokens (from API): {prompt_tokens} ***")
            print(f"Completion Tokens: {completion_tokens}")
            print(f"Total Tokens: {total_tokens}")
        else:
            print(f"*** Actual Prompt Tokens: N/A (Usage info not in response) ***")

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


# Example usage
def main():
    from openai import AzureOpenAI

    from config import VectorDBType, get_config

    config = get_config(VectorDBType.CHROMADB)

    llm_client = AzureOpenAI(
        api_key=config.azure_openai.llm_api_key,
        azure_endpoint=config.azure_openai.llm_endpoint,
        api_version=config.azure_openai.llm_api_version,
    )

    CATALOG_FILE = "x.json"
    catalog_schema = ""
    try:
        with open(CATALOG_FILE, "r") as f:
            # Load the content from the file
            catalog_data = json.load(f)
            # Dump it back to a string for the LLM prompt
            catalog_schema = json.dumps(catalog_data, indent=2)
    except FileNotFoundError:
        return "File not Found"
    except json.JSONDecodeError:
        return "Invalid JSON"
    except Exception as e:
        return f"An unexpected error occurred while reading the catalog file: {e}"

    # Test cases
    test_queries = [
        "What is the maximum loan amount and what is the status for applicant Smith?",
        "Find all auto loans with their approval status and interest rates",
        "Get the highest income applicant, then show me their loan details",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"USER QUERY: {query}")
        result = analyze_user_query(
            llm_client, query, config.azure_openai.llm_deployment_name, catalog_schema
        )

        # Access structured data
        print(f"\n📊 Structured Output Available:")
        print(json.dumps([asdict(a) for a in result.analyses], indent=2))


if __name__ == "__main__":
    main()
