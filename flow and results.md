# Data Pipeline Execution Log - Complete Documentation

## System Initialization

```bash
[INFO] Configuration loaded successfully
[INFO] Vector DB: chromadb
[INFO] Configuration validation passed
[INFO] ChromaDB client initialized. Persistence directory: ./chroma_db
[INFO] Azure OpenAI client initialized for embedding (Retrieval Utility).
[INFO] ChromaDB client initialized for retrieval. Persistence directory: ./chroma_db
[SUCCESS] Smart retrieval imported successfully!
[INFO] Configuring persistent DuckDB Azure connection for storage: auxeestorage
[SUCCESS] Persistent DuckDB Azure authentication configured successfully.
[INFO] Azure connection warmed up and ready.
[INFO] AzureOpenAI Client configured successfully for deployment: auxee-gpt-4o
```

---

## Global Data Catalog

### Logical Tables Available

| Logical Table | Columns |
|---------------|---------|
| **PAID_NARCAN_Sample_Data_Sheet1** | Unnamed: 0, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec, Total, Q1, Q2, Q3, Q4 |
| **Formulation_Test_Sheet1** | Formulation, Spray distance, Storage temperature, Relative humidity., SprayTec Device ID, Dv(10) (µm), Dv(50) (µm), Dv(90) (µm), Span, Droplets < 10 µm (%), Initial Weight (g), Final Weight (g), Dose Weight (mg), Target Dose, Unnamed: 14 |
| **Formulation2_Sheet1** | Formulation 1 - 3cm; Condition: 5°C, Unnamed: 1, Unnamed: 2, Unnamed: 3, Unnamed: 4, Unnamed: 5, Unnamed: 6, Unnamed: 7, Unnamed: 8, Unnamed: 9, Unnamed: 10, Unnamed: 11, Unnamed: 12, Unnamed: 13, Unnamed: 14, Unnamed: 15, Unnamed: 16, Unnamed: 17, Unnamed: 18, Unnamed: 19 |

---

## Query Execution

### User Query (Multi-Intent)

```
What is the average Final Weight (g) for all runs conducted at a Storage temperature of 40°C compared to those at 25°C and List the volumes for Canada Kit for every month (Jan through Jun) to identify when activity began.
```

### Decomposed Sub-queries

1. **Sub-query 1:** Find the average Final Weight (g) for all runs conducted at a Storage temperature of 40°C and compare it to those at 25°C.

2. **Sub-query 2:** List the volumes for Canada Kit for every month from January through June to identify when activity began.

---

## Sub-query 1: Temperature Comparison Analysis

### Intent Classification
- **Intent:** `SQL_QUERY`
- **Reason:** Aggregation operation (AVG) on numeric field
- **Duration:** 10.47 seconds

### Table Identification
```bash
-> STEP 3: Sub-query 'Find the average Final Weight (g) for all runs con...' 
   -> Identified tables: ['Formulation_Test_Sheet1'] 
   -> ['azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/Formulation_Test_Sheet1.parquet'] 
   (Mode: UNION_BY_NAME)
```

### Generated SQL
```sql
SELECT 
    "Storage temperature", 
    AVG("Final Weight (g)") AS average_final_weight
FROM 
    read_parquet(['azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/Formulation_Test_Sheet1.parquet'], union_by_name=true)
WHERE 
    "Storage temperature" IN ('40°C', '25°C')
GROUP BY 
    "Storage temperature";
```

### Result
**Status:** ✅ Success

| Storage temperature | average_final_weight |
|---------------------|---------------------|
| 40°C                | 4.91235             |
| 25°C                | 4.913               |

### Key Insights
- **Temperature Comparison:** Both storage temperatures show nearly identical average final weights
- **40°C Average:** 4.91235g
- **25°C Average:** 4.913g
- **Difference:** Only 0.00065g difference (negligible)
- **Conclusion:** Storage temperature (between 25°C and 40°C) has minimal impact on final weight

---

## Sub-query 2: Canada Kit Monthly Volumes

### Intent Classification
- **Intent:** `SEMANTIC_SEARCH`
- **Reason:** String-based lookup for "Canada Kit" requiring fuzzy matching
- **Duration:** 12.18 seconds

### Table Identification
```bash
-> STEP 3: Sub-query 'List the volumes for Canada Kit for every month fr...' 
   -> Identified tables: ['PAID_NARCAN_Sample_Data_Sheet1'] 
   -> ['azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/PAID_NARCAN_Sample_Data_Sheet1.parquet'] 
   (Mode: UNION_BY_NAME)
```

### Smart Retrieval Process
```bash
[INFO] Using smart semantic retrieval for sub-query: 'List the volumes for Canada Kit for every month from January through June to identify when activity began.'

[INFO] Relevant collections identified:
  - data_source_PAID_NARCAN_Sample_Data_Sheet1 (table: PAID_NARCAN_Sample_Data_Sheet1, score: 0.8206)
  - data_source_Formulation2_Sheet1 (table: Formulation2_Sheet1, score: 0.7363)
  - data_source_Formulation_Test_Sheet1 (table: Formulation_Test_Sheet1, score: 0.7173)

[INFO] Smart search identified 6 relevant collection(s)

[SUCCESS] Retrieved semantic context from 2 collection(s)
  Top score: 0.8206 from collection: data_source_PAID_NARCAN_Sample_Data_Sheet1
  Logical tables: ['Formulation2_Sheet1', 'PAID_NARCAN_Sample_Data_Sheet1']
  Mapped to parquet files: [
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/Formulation2_Sheet1.parquet',
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/PAID_NARCAN_Sample_Data_Sheet1.parquet'
  ]

[INFO] RAG identified specific files: [
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/Formulation2_Sheet1.parquet',
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/PAID_NARCAN_Sample_Data_Sheet1.parquet'
]

[INFO] Updating schema context for RAG-identified files...
```

### Generated SQL
```sql
SELECT "Jan", "Feb", "Mar", "Apr", "May", "Jun"
FROM read_parquet([
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/Formulation2_Sheet1.parquet', 
    'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/PAID_NARCAN_Sample_Data_Sheet1.parquet'
], union_by_name=true)
WHERE "Unnamed: 0" = 'Canada Kits';
```

### Result
**Status:** ✅ Success

| Month | Volume |
|-------|--------|
| Jan   | 192,126 |
| Feb   | 15,954.5 |
| Mar   | 558,795 |
| Apr   | 357,234 |
| May   | 421,665 |
| Jun   | 151,007 |

### Key Insights
- **Activity Pattern:** Significant activity spike in March (558,795 units)
- **Lowest Activity:** February shows the lowest volume (15,954.5 units)
- **Initial Volume:** January started with 192,126 units, indicating operations were already active
- **Q2 Trend:** Volumes remain substantial through Q2 (April-June)
- **Smart Retrieval Success:** System correctly identified PAID_NARCAN_Sample_Data_Sheet1 with highest relevance score (0.8206)

---

## Performance Metrics

### Execution Summary
- **Total Orchestration Time:** 20.87 seconds
- **Sub-queries Executed:** 2
- **Successful Results:** 2 (100% success rate)
- **Files Processed:** 3 Excel files
- **Collections Created:** 3 ChromaDB collections
- **Query Mode:** Parallel Execution
- **Total Vectors Generated:** 104 embeddings (5 + 51 + 48)

### Individual Query Performance
| Query | Intent | Duration | Status |
|-------|--------|----------|--------|
| Temperature Comparison | SQL_QUERY | 10.47s | ✅ Success |
| Canada Kit Volumes | SEMANTIC_SEARCH | 12.18s | ✅ Success |

### Vector Database Statistics
| Collection | Documents | Data Rows | Processing Time |
|------------|-----------|-----------|-----------------|
| PAID_NARCAN_Sample_Data_Sheet1 | 5 | 9 | 2.49s |
| Formulation_Test_Sheet1 | 51 | 101 | 4.13s |
| Formulation2_Sheet1 | 48 | 95 | 4.44s |

### System Performance
- ✅ Smart retrieval successfully identified correct data sources (0.8206 relevance score)
- ✅ Semantic search correctly matched "Canada Kit" string across collections
- ✅ Multi-intent query decomposition working flawlessly
- ✅ Temperature query successfully executed with proper data format handling
- ✅ Parallel processing optimized total execution time
- ✅ Both SQL_QUERY and SEMANTIC_SEARCH intents working correctly

---

## Technical Architecture Highlights

### Components Used
1. **Vector Database:** ChromaDB (persistent storage)
2. **Query Engine:** DuckDB (Azure-connected)
3. **LLM:** Azure OpenAI GPT-4
4. **Retrieval:** Smart semantic retrieval with auto-file detection
5. **Processing:** Parallel execution with ThreadPoolExecutor

### Data Flow
```
User Query 
  → Multi-Intent Decomposition 
    → Intent Classification (Router)
      → Smart Semantic Retrieval (if needed)
        → File Identification
          → Schema Loading
            → SQL Generation
              → DuckDB Execution
                → Results Aggregation
```

### Smart Retrieval Architecture

The smart retrieval system demonstrated exceptional performance in this execution:

#### Collection Relevance Scoring
```
Query: "List the volumes for Canada Kit..."
  ↓
Relevance Analysis:
  ✓ data_source_PAID_NARCAN_Sample_Data_Sheet1: 0.8206 (82% confidence - SELECTED)
  ✓ data_source_Formulation2_Sheet1: 0.7363 (74% confidence - SELECTED)
  ○ data_source_Formulation_Test_Sheet1: 0.7173 (72% confidence)
  ○ data_source_file1_Sheet1: 0.7168 (legacy collection)
  ○ data_source_file2_Sheet1: 0.7011 (legacy collection)
  ○ data_source_loan_Data: 0.7004 (legacy collection)
```

#### Automatic File Mapping
```
Collections → Parquet Files
  data_source_PAID_NARCAN_Sample_Data_Sheet1 
    → azure://.../PAID_NARCAN_Sample_Data_Sheet1.parquet
  
  data_source_Formulation2_Sheet1 
    → azure://.../Formulation2_Sheet1.parquet
```

#### Dynamic Schema Update
After identifying correct files, the system:
1. Retrieves semantic context from top collections
2. Maps collections to physical parquet files
3. Updates schema context with correct table structures
4. Provides LLM with accurate column names and sample data

This ensures the generated SQL queries against the correct data sources with proper column references.

---

## Appendix: System Configuration

### Files Processed
- `Formulation_Test.xlsx` → `Formulation_Test_Sheet1.parquet`
- `Formulation2.xlsx` → `Formulation2_Sheet1.parquet`
- `PAID NARCAN Sample Data.xlsx` → `PAID_NARCAN_Sample_Data_Sheet1.parquet`

### Azure Storage
- **Container:** auxee-upload-files
- **Path:** parquet_files/
- **Connection:** Persistent with authentication

### Vector Database
- **Type:** ChromaDB
- **Persistence:** ./chroma_db
- **Collections:** 3 (one per sheet)
- **Embedding Model:** Azure Text Embedding Ada-002

---

## Query Classification Rules Reference

### SEMANTIC_SEARCH Intent Triggers
The system uses semantic search when queries involve:

1. **Fuzzy Matching / Names**
   - Person names, company names, proper nouns
   - Handles misspellings, variations, case differences

2. **String-based Lookups**
   - Product names, descriptions, titles
   - Location names (cities, countries, addresses)
   - Category names or labels
   - Any user-provided text needing column matching
   - Case variations (TRUCK, Truck, truck)

3. **Conceptual/Descriptive Lookups**
   - Non-indexed, descriptive fields
   - Natural language descriptions

4. **Pattern Matching**
   - When exact format/spelling is unknown
   - Example: "Moscow" might be stored as "moscow", "MOSCOW", or "Moscow, Russia"

### SQL_QUERY Intent Triggers
The system uses direct SQL when queries involve:

1. **Direct Calculations/Aggregation**
   - Functions: SUM, AVG, COUNT, MAX, MIN, TOTAL

2. **Precise Filtering on Structured Data**
   - Numeric columns (IDs, amounts, quantities)
   - Dates, booleans
   - Exact categorical values (clearly structured)

3. **Mathematical Operations**
   - Comparisons, arithmetic on numeric fields

4. **Date/Time Operations**
   - Specific dates, date ranges, time periods

### Decision Criteria
- **String value to search?** → `SEMANTIC_SEARCH`
- **String with variations/typos/case differences?** → `SEMANTIC_SEARCH`
- **Pure numeric/date operations?** → `SQL_QUERY`
- **When in doubt about strings?** → Prefer `SEMANTIC_SEARCH`

### Examples
| Query | Intent | Reason |
|-------|--------|--------|
| "What is the credit score for Harrison?" | SEMANTIC_SEARCH | Name lookup |
| "Find max discount for Moscow truck" | SEMANTIC_SEARCH | Strings "Moscow" and "truck" need fuzzy matching |
| "Show me data for Premium Package" | SEMANTIC_SEARCH | Text matching for "Premium Package" |
| "What is the total amount?" | SQL_QUERY | Pure aggregation, no string lookup |
| "Count all records where date > 2024" | SQL_QUERY | Date filtering, no string matching |
| "Average price for product category 'Electronics'" | SEMANTIC_SEARCH | Involves string "Electronics" |

---