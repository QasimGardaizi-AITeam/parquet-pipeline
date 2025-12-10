## ✅ Azure OpenAI Client
**[INFO]** AzureOpenAI Client configured successfully for deployment: `auxee-gpt-4o`

---

# 🚀 Data Pipeline Execution Started

### 🟦 Excel → Parquet
- **[INFO]** Converting `loan.xlsx` to Parquet...
- **[SUCCESS]** Parquet saved: `../data/loan_20251210131416_08263cf8.parquet`
- **[INFO]** Pandas DataFrame deleted from memory.

---

# 🟩 Ingestion: `data_source_loan`
- Data loaded: **1000 rows**
- Generated: **500 chunks**
- Starting vector embedding via deployment: `auxzee-text-embedding-ada-002`
- Embedding complete → Vector dimension: **1536**
- Preparing to insert **500 documents** into MongoDB
- Clearing old documents...
- Inserting documents (synchronously)...
- **[SUCCESS]** Inserted **500 chunks**

---

# ⚙️ EXECUTION START

---

## 🔍 Query 1
### **Query:**  
**“What is the maximum income for all loans and what are the details for the client Kathleen Vasqez?”**

### → Step 1: Multi-Intent Decomposition  
- Found **2 sub-queries**
- Sub-queries:  
  1. Find the maximum income for all loans  
  2. Retrieve details for *Kathleen Vasqez*  
- Executing using **ThreadPoolExecutor**

---

### ### 🧩 Sub-Query A  
#### **Intent:** `SQL_QUERY`  
**Duration:** 2.42s  

**Result:**
| max_income |
|-----------:|
|     119998 |

Semantic context retrieved (Score: **0.8817**)

---

### ### 🧩 Sub-Query B  
#### **Intent:** `SEMANTIC_SEARCH`  
**Duration:** 7.15s  

**Result:**
| loan_application_id                  | applicant_name   | loan_amount_requested | loan_type | applicant_income | credit_score | loan_status | repayment_schedule | interest_rate | loan_term | collateral | application_date | approved_amount | repayment_start_date | is_employed | monthly_payment |
|--------------------------------------|-------------------|------------------------|-----------|-------------------|--------------|--------------|--------------------|----------------|------------|------------|-------------------|------------------|-----------------------|--------------|------------------|
| 1b2fa939-9af5-4610-81a0-bf379320a656 | Kathleen Vasquez | 32300.7               | Student   | 72110.1           | 519          | Approved     | Quarterly          | 8.04           | 24         | college    | 2025-05-11        | 38840.8           |                       | True         | 886              |

---

### ✅ ORCHESTRATION COMPLETE — **10.88s**

---

## 🔍 Query 2
### **Query:**  
**“What is credit Score of Harrison, Ters.”**

### → Step 1: Decomposition Fallback  
- Treated as a **single query**
- Semantic search score: **0.8758**

---

### ### 🧩 Result  
#### **Intent:** `SEMANTIC_SEARCH`  
**Duration:** 5.77s  

| credit_score |
|--------------|
| 620 |

---

### ✅ ORCHESTRATION COMPLETE — **6.84s**

---

# 🎉 Execution Complete
