import json
from openai import AzureOpenAI
from typing import List, Dict, Any

SQL_TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_single_sql_query",
        "description": "Decompose the user's complex request into distinct, simple parts, and generate a function call for EACH part.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_part": {
                    "type": "string",
                    "description": "A concise natural language statement representing ONE single, atomic question."
                }
            },
            "required": ["query_part"]
        }
    }
}
TOOLS_LIST = [SQL_TOOL_SPEC]


def decompose_multi_intent_query(llm_client: AzureOpenAI, user_question: str, deployment_name: str) -> List[str]:
    """
    Decomposes a multi-intent query into a list of single-intent sub-queries 
    using AzureOpenAI Function Calling.
    """
    print("-> STEP 1: Decomposing Multi-Intent Query...")
    
    SYSTEM_PROMPT = """
        You are a master query decomposition agent. Your task is to break the user's request into the minimum number of distinct, executable questions.

        CRITICAL RULE: Only decompose the request if the parts are ATOMICALLY INDEPENDENT. If all parts of the request share the same primary filter, condition, or aggregation requirement, you MUST treat it as a single request to be handled by one complex SQL query.

        **Decompose only for distinct, independent topics, such as:**
        1. An aggregation request (e.g., 'What is the maximum amount?') AND a filtered request (e.g., 'What is the status for applicant X?'). (Two calls)
        2. A request about one category (e.g., 'All automotive-related applications') AND a request about a completely different category (e.g., 'All housing-related applications'). (Two calls)
        
        You MUST call the 'generate_single_sql_query' tool for EACH distinct, independent part of the user's request.
        
        **Example of Single Call (No Decomposition):**
        If the request is: 'Find all applications related to the automotive category and list their approval status and interest rates.' 
        -> This is ONE single, filtered request, and you MUST call the tool only ONCE.
        
        **Example of Two Calls (Decomposition):**
        If the request is: 'What is the highest applicant income, and what is the current status of the loan for applicant Smith?'
        -> This requires two independent pieces of information, so you MUST call the tool twice.
        """
    
    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question}
            ],
            tools=TOOLS_LIST,
            tool_choice="auto", 
            temperature=0.0
        )
        
        tool_calls = response.choices[0].message.tool_calls
        query_parts: List[str] = []
        
        if tool_calls:
            for call in tool_calls:
                if call.function.name == "generate_single_sql_query":
                    # Arguments are returned as a JSON string, which needs to be parsed
                    args = json.loads(call.function.arguments)
                    query_parts.append(args.get("query_part"))
            
            if len(query_parts) > 1:
                print(f"-> Decomposition Success: Found {len(query_parts)} sub-queries.")
                return query_parts
        
        # Fallback: Treat as a single query if zero or one distinct call was made
        print("-> Decomposition Fallback: Treating as a single query.")
        return [user_question]
        
    except Exception as e:
        print(f"[ERROR] LLM Decomposition failed: {e}. Defaulting to single query.")
        return [user_question]