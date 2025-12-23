# Pipeline Execution Logs

## Command Executed

    python3 -m pipeline.execution.main

---

## Environment Information

- **Virtual Environment:** `venv`
- **Python Version:** 3.9
- **System Warning:**

      urllib3 v2 only supports OpenSSL 1.1.1+.
      Current SSL: LibreSSL 2.8.3

  ⚠️ Non-blocking warning. Pipeline execution was not affected.

---

## Configuration Status

- ✅ Configuration loaded successfully  
- ✅ Configuration validation passed  

---

## Business Query

**Objective**

Using **Region**, **Product Category**, and quarterly sales (**Q1–Q4**), determine the product category in each region with the **highest total annual sales**.

---

## Pipeline Execution Flow

### 1. Validate Input

- **Node:** Validate Input  
- **Status:** ✅ Success  

---

### 2. Analyze Query

- **Node:** Analyze Query  
- **Sub-questions Identified:** 1  
- **Intent:** `SQL_QUERY`  
- **Tokens Used:** 4,584  

**Sub-question**

    Find the product category in each region that has the highest total annual sales.

---

### 3. Identify Ready Queries

- **Ready Queries:** 1  
- **Status:** ✅ Success  

---

### 4. Execute SQL Query

#### Data Source

- **Format:** Parquet  
- **Storage:** Azure Blob Storage  
- **Exact URI Used**

      azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/TDQP_parquet

---

#### Table Schema: `TDQP_parquet`

| Column | Type | Description |
|------|------|-------------|
| region | VARCHAR | Sales region |
| product_category | VARCHAR | Product category |
| q1_sales | BIGINT | Q1 sales |
| q2_sales | BIGINT | Q2 sales |
| q3_sales | BIGINT | Q3 sales |
| q4_sales | BIGINT | Q4 sales |
| profit | DOUBLE | Profit |
| revenue | DOUBLE | Revenue |
| total_sales | BIGINT | Total sales |
| average_sales | DOUBLE | Average quarterly sales |

---

#### Generated SQL Query

    SELECT
      region,
      product_category,
      SUM(q1_sales + q2_sales + q3_sales + q4_sales) AS total_annual_sales
    FROM read_parquet(
      'azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/TDQP_parquet'
    )
    WHERE
      q1_sales IS NOT NULL
      AND q2_sales IS NOT NULL
      AND q3_sales IS NOT NULL
      AND q4_sales IS NOT NULL
    GROUP BY
      region,
      product_category
    QUALIFY
      ROW_NUMBER() OVER (
        PARTITION BY region
        ORDER BY SUM(q1_sales + q2_sales + q3_sales + q4_sales) DESC
      ) = 1;

---

#### SQL Execution Details

- **Status:** ✅ Success  
- **Execution Time:** 6.87 seconds  
- **Rows Returned:** 4  

---

## Query Results

### Top Product Category by Region

| Region | Product Category | Total Annual Sales |
|-------|------------------|--------------------|
| West  | Books            | 60,335             |
| South | Electronics      | 58,989             |
| East  | Office Supplies  | 69,448             |
| North | Office Supplies  | 58,380             |

---

## Final Summary

- **East Region** recorded the highest total annual sales, led by *Office Supplies*.
- **Office Supplies** dominated both **East** and **North** regions.
- **Electronics** performed best in the **South** region.
- **Books** uniquely led sales in the **West** region.

These results highlight strong regional differences in product performance and can inform inventory planning, marketing focus, and strategic decisions.

---

## Execution Metadata

- **Total Questions:** 1  
- **Independent Queries:** 1  
- **Dependent Queries:** 0  
- **Errors:** 0  
- **Total Pipeline Duration:** ~20.9 seconds  
- **Final Status:** ✅ `success`
