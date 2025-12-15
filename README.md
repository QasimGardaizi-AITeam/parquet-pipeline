# LangGraph RAG Pipeline Documentation

## Overview

This document provides a comprehensive guide to the LangGraph-based RAG (Retrieval-Augmented Generation) pipeline that orchestrates multi-intent query processing with hybrid retrieval strategies.

## Architecture

### Graph Structure

The pipeline is implemented as a **StateGraph** with the following nodes and edges:

```
┌─────────────┐
│ Initialize  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Decompose  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐◄───┐
│ Identify Sources│    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│  Load Context   │    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│  Route Intent   │    │
└────────┬────────┘    │
         │             │
    ┌────┴────┐        │
    │         │        │
    ▼         ▼        │
┌─────┐   ┌─────┐     │
│Sem. │   │ SQL │     │
│Search   │Query│     │
└──┬──┘   └──┬──┘     │
   │         │        │
   └────┬────┘        │
        │             │
        ▼             │
┌──────────────┐      │
│   Execute    │      │
└──────┬───────┘      │
       │              │
       ▼              │
┌──────────────┐      │
│ Check More   │──────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Summary    │
└──────┬───────┘
       │
       ▼
     [END]
```

## State Schema

The `GraphState` TypedDict contains all information flowing through the graph:

### Input State
- `user_question`: Original user query
- `all_parquet_files`: List of available data files
- `global_catalog_string`: Human-readable catalog
- `global_catalog_dict`: Programmatic catalog access
- `config`: Application configuration
- `enable_debug`: Debug mode flag

### Processing State
- `sub_queries`: Decomposed queries
- `current_query`: Query being processed
- `query_index`: Current position in query list
- `required_tables`: Identified data sources
- `join_key`: Multi-table join key
- `target_parquet_files`: Specific files for query
- `use_union_by_name`: Union vs Join mode
- `parquet_schema`: Schema information
- `df_sample`: Sample data

### Execution State
- `intent`: Routing decision (SEMANTIC_SEARCH or SQL_QUERY)
- `semantic_context`: Retrieved vector context
- `sql_query`: Generated SQL
- `sql_explanation`: Query explanation
- `result_df`: Query results
- `execution_duration`: Timing information

### Output State
- `results`: Dictionary of query → DataFrame results
- `summary`: Natural language summary
- `error`: Error messages if any

## Node Descriptions

### 1. Initialize Context
**Purpose**: Set up the execution environment
- Establishes DuckDB connection
- Initializes Azure OpenAI client
- Prepares result storage
- Sets up error handling

### 2. Decompose Query
**Purpose**: Break down multi-intent queries
- Uses LLM with function calling
- Identifies atomic sub-queries
- Handles single-intent fallback
- Critical for parallel processing

**Example**:
```
Input: "What is the max loan amount and who is Kathleen Vasquez?"
Output: ["What is the max loan amount?", "Who is Kathleen Vasquez?"]
```

### 3. Identify Data Sources
**Purpose**: Map queries to specific data tables
- Uses LLM to understand data requirements
- Matches query needs to available tables
- Determines JOIN vs UNION strategy
- Optimizes data access patterns

**Strategies**:
- `*` → All files with UNION
- Single table → Direct access
- Multiple tables → UNION or JOIN based on join_key

### 4. Load Query Context
**Purpose**: Retrieve schema and sample data
- Loads relevant table schemas
- Fetches sample rows for context
- Prepares metadata for LLM
- Optimizes based on identified sources

### 5. Route Query Intent
**Purpose**: Determine execution strategy
- Classifies query as SEMANTIC_SEARCH or SQL_QUERY
- Analyzes for string matching needs
- Considers fuzzy matching requirements
- Routes to appropriate path

**Classification Rules**:
- **SEMANTIC_SEARCH**: Names, fuzzy text, typos, conceptual queries
- **SQL_QUERY**: Calculations, aggregations, numeric filtering

### 6a. Semantic Search (Optional)
**Purpose**: Retrieve relevant context from vector DB
- Queries ChromaDB with query embedding
- Retrieves top-k similar chunks
- Maps collections to parquet files
- Provides exact values for SQL generation

**Benefits**:
- Handles typos and variations
- Finds exact matches in data
- Improves SQL accuracy
- Reduces hallucinations

### 7. Generate SQL Query
**Purpose**: Create executable SQL
- Uses schema and sample data
- Incorporates semantic context if available
- Generates DuckDB-compatible SQL
- Includes query explanation

**Prompt Components**:
1. Semantic context (if SEMANTIC_SEARCH)
2. Database schema
3. Sample data
4. User query
5. Generation instructions

### 8. Execute Query
**Purpose**: Run SQL and collect results
- Executes against DuckDB
- Handles errors gracefully
- Times execution
- Stores results for summary

### 9. Check More Queries
**Purpose**: Loop control
- Increments query counter
- Determines if more queries exist
- Routes back to identify_sources or forward to summary

### 10. Generate Summary
**Purpose**: Synthesize natural language response
- Combines all query results
- Uses LLM for natural language generation
- Provides user-friendly output
- Handles multiple results coherently

## Key Features

### 1. Hybrid RAG Approach
Combines semantic search and structured query execution:
- **Semantic Layer**: Vector similarity for fuzzy matching
- **SQL Layer**: Precise structured queries
- **Hybrid Mode**: Best of both worlds

### 2. Multi-Intent Query Handling
Automatically decomposes complex queries:
```python
"What is the total for X and the status of Y?"
→ ["What is the total for X?", "What is the status of Y?"]
```

### 3. Smart Data Source Identification
Intelligently maps queries to tables:
- Analyzes column requirements
- Minimizes data scanned
- Optimizes query performance

### 4. Adaptive Routing
Chooses optimal execution path:
- Name lookups → Semantic search first
- Calculations → Direct SQL
- Hybrid queries → Combined approach

### 5. State Persistence
Uses MemorySaver for checkpointing:
- Enables debugging
- Supports recovery
- Facilitates analysis

## Usage Examples

### Basic Usage

```python
from langgraph_rag_pipeline import run_rag_pipeline
from config import get_config, VectorDBType

# Initialize
config = get_config(VectorDBType.CHROMADB)

# Run query
results = run_rag_pipeline(
    user_question="What is the total loan amount for Kathleen Vasquez?",
    all_parquet_files=my_files,
    global_catalog_string=catalog_str,
    global_catalog_dict=catalog_dict,
    config=config,
    enable_debug=False
)

# Access results
for query, df in results.items():
    print(f"Query: {query}")
    print(df.to_markdown())
```

### Advanced Usage with Multiple Queries

```python
queries = [
    "What is the maximum discount in the Moscow region?",
    "List all products in the OTC category",
    "Show monthly volumes for Canada Kit from Jan to Jun"
]

for query in queries:
    results = run_rag_pipeline(
        user_question=query,
        all_parquet_files=all_files,
        global_catalog_string=catalog_str,
        global_catalog_dict=catalog_dict,
        config=config,
        enable_debug=True  # Enable for detailed logs
    )
    
    print(f"\n{'='*80}")
    print(f"Results for: {query}")
    print(f"{'='*80}")
    for sub_query, df in results.items():
        print(f"\n{sub_query}:")
        print(df.to_markdown(index=False))
```

### Integration with Existing Pipeline

```python
# Replace the existing generate_and_execute_query call
from langgraph_rag_pipeline import run_rag_pipeline

# Old code:
# final_results = generate_and_execute_query(
#     llm_client, query, files, catalog_str, catalog_dict, config
# )

# New code:
final_results = run_rag_pipeline(
    user_question=query,
    all_parquet_files=files,
    global_catalog_string=catalog_str,
    global_catalog_dict=catalog_dict,
    config=config,
    enable_debug=False
)
```

## Benefits of LangGraph Architecture

### 1. Modularity
Each node is independent and testable:
```python
# Test individual nodes
from langgraph_rag_pipeline import decompose_query

state = {"user_question": "test", "llm_client": client, "config": config}
result = decompose_query(state)
```

### 2. Observability
Clear execution flow and logging:
- Each node logs its actions
- State transitions are visible
- Errors are isolated to specific nodes

### 3. Extensibility
Easy to add new capabilities:
```python
def validate_results(state: GraphState) -> GraphState:
    """New node to validate query results"""
    # Validation logic
    return state

# Add to graph
workflow.add_node("validate", validate_results)
workflow.add_edge("execute", "validate")
workflow.add_edge("validate", "check_more")
```

### 4. Debugging
State inspection at any point:
```python
# Access intermediate state
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Run and inspect
result = app.invoke(initial_state, config)
# Inspect state after each node via checkpointer
```

### 5. Parallel Processing (Future Enhancement)
Graph structure enables parallelization:
- Multiple sub-queries can execute simultaneously
- Independent data sources can load in parallel
- Results can be aggregated efficiently

## Performance Considerations

### Optimization Strategies

1. **Connection Pooling**: DuckDB connection is persistent
2. **Batch Processing**: Multiple queries processed efficiently
3. **Lazy Loading**: Data loaded only when needed
4. **Caching**: Results can be cached at checkpoints

### Benchmarks

Typical execution times:
- Simple query (single table): 2-5 seconds
- Complex query (multi-table): 5-10 seconds
- Multi-intent query (3 sub-queries): 10-20 seconds

## Error Handling

The pipeline implements comprehensive error handling:

```python
try:
    results = run_rag_pipeline(...)
except Exception as e:
    print(f"Pipeline failed: {e}")
    # Check state["error"] for specific node failures
```

Each node handles its own errors and sets the `error` field in state, allowing the pipeline to gracefully continue or terminate.

## Migration Guide

### From Original Pipeline to LangGraph

**Step 1**: Install LangGraph
```bash
pip install langgraph
```

**Step 2**: Replace function call
```python
# Before
from main_pipeline import generate_and_execute_query
results = generate_and_execute_query(llm_client, query, ...)

# After
from langgraph_rag_pipeline import run_rag_pipeline
results = run_rag_pipeline(user_question=query, ...)
```

**Step 3**: Update result handling (if needed)
The return format is the same: `Dict[str, pd.DataFrame]`

## Visualization

To visualize the graph structure:

```python
from langgraph_rag_pipeline import create_rag_graph
from langgraph.graph import StateGraph

workflow = create_rag_graph()
app = workflow.compile()

# Generate Mermaid diagram
print(app.get_graph().draw_mermaid())
```

## Future Enhancements

### Planned Features
1. **Parallel Query Execution**: Process independent sub-queries simultaneously
2. **Result Caching**: Cache frequently accessed results
3. **Query Optimization**: Automatic query plan optimization
4. **Streaming Results**: Stream large result sets
5. **Human-in-the-Loop**: Interactive query refinement
6. **Multi-Modal Support**: Handle images and documents

### Extensibility Points
- Custom routing strategies
- Additional vector stores
- Alternative SQL generators
- Custom summarization approaches

## Troubleshooting

### Common Issues

**Issue**: Graph execution hangs
**Solution**: Enable debug mode to identify stuck node

**Issue**: SQL generation fails
**Solution**: Check schema loading in load_context node

**Issue**: Semantic search returns no results
**Solution**: Verify ChromaDB collections exist and query embedding works

**Issue**: Summary is empty
**Solution**: Check that results dictionary contains valid DataFrames
