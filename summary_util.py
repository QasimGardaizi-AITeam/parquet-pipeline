"""
Simple Summary Agent - Describes answers according to questions
"""

import pandas as pd
from typing import Dict
from openai import AzureOpenAI


def generate_simple_summary(
    results: Dict[str, pd.DataFrame],
    llm_client: AzureOpenAI,
    deployment_name: str
) -> str:
    """
    Simple summary agent that describes answers according to questions.
    
    Args:
        results: Dictionary mapping queries to result DataFrames
        llm_client: Azure OpenAI client
        deployment_name: LLM deployment name
        
    Returns:
        Natural language summary as a string
    """
    
    print("\n" + "="*80)
    print("       *** GENERATING SUMMARY ***")
    print("="*80)
    
    # Prepare all Q&A pairs for the LLM
    qa_pairs = []
    
    for query, result_df in results.items():
        # Format the result data
        if result_df.empty or 'Error' in result_df.columns:
            answer = "No data found or error occurred"
        elif len(result_df) <= 10:
            answer = result_df.to_markdown(index=False)
        else:
            answer = result_df.head(10).to_markdown(index=False) + f"\n... and {len(result_df) - 10} more rows"
        
        qa_pairs.append({
            "question": query,
            "answer": answer
        })
    
    # Build prompt for LLM
    qa_text = ""
    for i, qa in enumerate(qa_pairs, 1):
        qa_text += f"\n**Question {i}:** {qa['question']}\n"
        qa_text += f"**Data:**\n{qa['answer']}\n"
    
    SUMMARY_PROMPT = f"""
You are a helpful assistant that describes query results in natural language.

Given the following questions and their data results, provide a clear, conversational summary that answers each question directly.

{qa_text}

Write a natural language summary that:
1. Answers each question clearly and concisely
2. Uses specific values and numbers from the data
3. Is written in a conversational, easy-to-understand style
4. Keeps each answer to 1-2 sentences

Your summary:
"""
    
    try:
        response = llm_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": SUMMARY_PROMPT}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content.strip()
        
        print("\n" + "="*80)
        print("       *** SUMMARY ***")
        print("="*80)
        print(f"\n{summary}\n")
        print("="*80)
        
        return summary
        
    except Exception as e:
        print(f"[ERROR] Failed to generate summary: {e}")
        return "Failed to generate summary"