# 🤖 Hybrid RAG Text-to-SQL System

This project implements a robust Hybrid Retrieval-Augmented Generation (RAG) pipeline to translate natural language questions into executable DuckDB SQL queries. It uses a dynamic router to choose between two execution paths:
1.  **Semantic Search (RAG):** For fuzzy matching, names, and conceptual queries (using Azure OpenAI Embeddings and MongoDB Atlas Vector Search).
2.  **Direct SQL:** For structured aggregation and precise filtering (using Azure OpenAI GPT-4o and dynamic context).

## ✨ Features

* **Dynamic Data Ingestion:** Converts source Excel (`.xlsx`) files into fast Parquet (`.parquet`) for DuckDB.
* **Automated Vector Indexing:** Programmatically creates the MongoDB Atlas Vector Search index upon data ingestion.
* **Intelligent Routing:** Uses an LLM router to direct queries to the most appropriate execution engine.
* **DuckDB Execution:** High-performance local SQL execution using the generated queries.

## ⚙️ Setup and Installation

### 1. Prerequisites

* Python 3.9+
* A MongoDB Atlas Cluster (M10+ recommended for Vector Search performance)
* An Azure OpenAI Service deployment with:
    * A **Chat/Completion Model** (e.g., `gpt-4o`)
    * An **Embedding Model** (e.g., `text-embedding-ada-002`)

### 2. Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate