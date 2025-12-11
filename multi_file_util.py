# multi_file_util.py (Updated with clearer LLM instructions)

import json
from openai import AzureOpenAI
from typing import List, Dict, Any, Tuple

# Assume CATALOG_SCHEMA is a detailed string summary of ALL available parquet files/sheets
# Example: "Logical Table: Loans_2024 (Columns: id, amount, status). Logical Table: Clients (Columns: id, name, income)."

FILE_IDENTIFIER_TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "identify_data_sources",
        "description": "Identify the logical tables required to answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "tables_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 1 or more logical table IDs (e.g., 'Loans_Sheet1', 'Clients_Sheet2') from the provided catalog schema needed to answer the query. Use '*' if the query should be run against all data combined."
                },
                "join_key": {
                    "type": "string",
                    # CLARIFICATION: Emphasize that the join key is mandatory for multi-table queries
                    "description": "The common column (e.g., 'customer_id') that links the tables in 'tables_required'. This field MUST be populated if two or more distinct tables are required."
                }
            },
            "required": ["tables_required"]
        }
    }
}
FILE_TOOLS_LIST = [FILE_IDENTIFIER_TOOL_SPEC]


def identify_required_tables(llm_client: AzureOpenAI, user_question: str, deployment_name: str, catalog_schema: str) -> Tuple[List[str], str]:
    """Uses LLM to map user query to internal logical tables."""
    
    SYSTEM_PROMPT = f"""
        You are a data source selection agent. Your task is to identify ALL logical tables needed to answer the user's question based on the Data Catalog.
        --- DATA CATALOG ---
        {catalog_schema}
        --- END CATALOG ---
        CRITICAL RULES:
        1. **Analyze ALL filtering criteria** in the user's question (dates, product types, names, categories, etc.)
        2. **Map each filter to its column**: Identify which table contains each column needed for filtering
        3. **Include ALL tables**: If filters require columns from multiple tables, you MUST list ALL of them"""
    
    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User Query: {user_question}"}
            ],
            tools=FILE_TOOLS_LIST,
            tool_choice={"type": "function", "function": {"name": "identify_data_sources"}},
            temperature=0.0
        )
        
        call = response.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)
        
        tables = args.get("tables_required", ["*"])
        join_key = args.get("join_key", "")
        
        return tables, join_key
        
    except Exception as e:
        print(f"[ERROR] File Identification failed: {e}. Defaulting to all files ('*').")
        return ["*"], ""