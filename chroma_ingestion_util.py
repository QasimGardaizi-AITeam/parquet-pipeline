import os
import sys
import pandas as pd
import time
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Union
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb

# --- 1. Configuration & Client Setup ---

from config import get_config, VectorDBType

# Get config
config = get_config(VectorDBType.CHROMADB)

# Initialize Azure OpenAI client
try:
    # Define openai_client here so it can be passed to the worker function
    openai_client = AzureOpenAI(
        azure_endpoint=config.azure_openai.embedding_endpoint,
        api_key=config.azure_openai.embedding_api_key,
        api_version=config.azure_openai.embedding_api_version
    )
    # Adding a check to confirm the client works (optional, but good practice)
    # You might consider moving the client creation to a utility function or the main pipeline script.
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Ingestion): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(
        path=config.vector_db.chromadb.persist_directory,
        settings=Settings(
            anonymized_telemetry=config.vector_db.chromadb.anonymized_telemetry
        )
    )
    print(f"[INFO] ChromaDB client initialized. Persistence directory: {config.vector_db.chromadb.persist_directory}")
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Ingestion): Error initializing ChromaDB client: {e}")
    sys.exit(1)


# ------------------------------------------------
# Helper function to read Parquet from Azure
# ------------------------------------------------

def read_parquet_from_azure(file_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file from either local path or Azure Blob Storage URI.
    Uses DuckDB for Azure URIs.
    """
    if file_path.startswith('azure://'):
        try:
            conn = duckdb.connect()

            conn.execute("INSTALL azure;")
            conn.execute("LOAD azure;")

            if config.azure_storage.connection_string:
                escaped_conn_str = config.azure_storage.connection_string.replace("'", "''")
                conn.execute(f"SET azure_storage_connection_string='{escaped_conn_str}';")

            df = conn.execute(f"SELECT * FROM read_parquet('{file_path}')").fetchdf()
            conn.close()

            print(f"[INFO] Successfully read Parquet from Azure: {os.path.basename(file_path)}")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to read Parquet from Azure URI '{file_path}': {e}")
            raise
    else:
        return pd.read_parquet(file_path)


# ------------------------------------------------
# 2. Utility Functions
# ------------------------------------------------

def chunk_dataframe_dynamic(df: pd.DataFrame, max_tokens_per_chunk: int = 1000) -> List[Dict[str, Any]]:
    """Dynamically chunk a DataFrame into text blocks."""
    chunks = []
    headers = list(df.columns)
    current_chunk = []
    current_indices = []
    current_size = 0

    for idx, row in df.iterrows():
        row_values = [str(row[col]) if pd.notna(row[col]) else "NULL" for col in headers]
        row_text = " | ".join([f"{col}:{val}" for col, val in zip(headers, row_values)])

        row_index_label = df.index.name if df.index.name else 'index'
        row_text_with_index = f"[{row_index_label} {idx}] {row_text}"
        row_size = len(row_text_with_index)

        if current_size + row_size > max_tokens_per_chunk and current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "metadata": {"row_indices": list(current_indices), "columns": list(headers)}
            })
            current_chunk, current_indices, current_size = [], [], 0

        current_chunk.append(row_text_with_index)
        current_indices.append(idx)
        current_size += row_size

    if current_chunk:
        chunks.append({
            "text": "\n\n".join(current_chunk),
            "metadata": {"row_indices": list(current_indices), "columns": list(headers)}
        })
    return chunks


# --- MODIFIED: Client is now passed as an argument for thread safety ---
def get_embedding_for_chunk(client: AzureOpenAI, chunk_texts: List[str]) -> List[List[float]]:
    """Worker function to get embeddings for a batch of chunk texts."""
    try:
        # Use the passed client
        response = client.embeddings.create(
            model=config.azure_openai.embedding_deployment_name,
            input=chunk_texts,
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"[EMBEDDING ERROR] Failed to embed batch: {e}")
        return [[]] * len(chunk_texts)


# ------------------------------------------------
# 3. Core Ingestion Function for ChromaDB
# ------------------------------------------------
def ingest_to_chroma_db(file_path: str, sheet_name: str = None, collection_prefix: str = None) -> Union[bool, str]:
    """
    Loads data from the given Parquet file (local or Azure URI),
    chunks, embeds (in parallel), and inserts into ChromaDB.
    """

    # NOTE: openai_client and chroma_client are global and used here.
    # If this function is imported and called elsewhere, those globals must be defined first.
    
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix

    # 1. Determine Dynamic Collection Name
    base_name = os.path.splitext(os.path.basename(file_path.split('/')[-1]))[0]

    # ChromaDB collection names must be alphanumeric with underscores/hyphens, 3-63 chars
    if sheet_name:
        sheet_name_clean = "".join(c for c in sheet_name if c.isalnum() or c in ('_',)).rstrip().lower()
        COLLECTION_NAME = f"{collection_prefix}_{base_name}"
    else:
        COLLECTION_NAME = f"{collection_prefix}_{base_name}"

    # Ensure collection name meets ChromaDB requirements
    COLLECTION_NAME = COLLECTION_NAME.replace('-', '_')[:63]

    print(f"\n--- Starting ChromaDB Ingestion for Collection: '{COLLECTION_NAME}' (Source: {os.path.basename(file_path)}) ---")

    # 2. Load Data
    try:
        df = read_parquet_from_azure(file_path)
        print(f"Data loaded: {len(df)} rows.")

    except FileNotFoundError:
        return f"Error: Input file not found at: {file_path}"
    except Exception as e:
        return f"Error during data loading: {e}"

    # 3. Chunking & Embedding

    chunks = chunk_dataframe_dynamic(df, max_tokens_per_chunk=1000)
    print(f"Generated {len(chunks)} chunks.")
    chunk_texts = [c["text"] for c in chunks]

    print(f"Starting parallel vector embedding using Azure Deployment: {config.azure_openai.embedding_deployment_name}...")

    BATCH_SIZE = 200
    MAX_THREADS = 8
    text_batches = [chunk_texts[i:i + BATCH_SIZE] for i in range(0, len(chunk_texts), BATCH_SIZE)]

    all_embeddings = []
    embedding_start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_batch = {
                # --- MODIFIED: Passing openai_client explicitly ---
                executor.submit(get_embedding_for_chunk, openai_client, batch): batch 
                for batch in text_batches
            }

            for future in as_completed(future_to_batch):
                batch_embeddings = future.result()
                all_embeddings.extend(batch_embeddings)

        embedding_duration = time.time() - embedding_start_time

        if not all_embeddings or any(not e for e in all_embeddings):
            return f"Error during parallel embedding: One or more batches failed. Duration: {embedding_duration:.2f}s"

        embedding_dim = len(all_embeddings[0])
        print(f"Embedding complete ({len(all_embeddings)} vectors) in {embedding_duration:.2f}s. Vector dimension: {embedding_dim}")

    except Exception as e:
        return f"Error during ThreadPool or Azure OpenAI API call: {e}"

    # 4. Prepare Documents for ChromaDB
    try:
        # Get or create collection (ChromaDB will automatically create if it doesn't exist)
        # Delete existing collection if it exists to ensure fresh data
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # Collection doesn't exist, that's fine

        # Create new collection
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity like MongoDB
        )
        print(f"[INFO] Created new ChromaDB collection '{COLLECTION_NAME}'")

        # Prepare data for ChromaDB
        ids = [f"{COLLECTION_NAME}_chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                **chunk["metadata"],
                "source_file": os.path.basename(file_path),
                "row_indices": str(chunk["metadata"]["row_indices"]),  # Convert list to string
                "columns": str(chunk["metadata"]["columns"])  # Convert list to string
            }
            for chunk in chunks
        ]

        # Insert documents in batches (ChromaDB recommends batches for large datasets)
        CHROMA_BATCH_SIZE = 1000
        total_inserted = 0

        for i in range(0, len(ids), CHROMA_BATCH_SIZE):
            batch_end = min(i + CHROMA_BATCH_SIZE, len(ids))
            batch_num = (i // CHROMA_BATCH_SIZE) + 1
            total_batches = (len(ids) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE

            collection.add(
                ids=ids[i:batch_end],
                embeddings=all_embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

            total_inserted += (batch_end - i)
            print(f"[PROGRESS] Batch {batch_num}/{total_batches}: Inserted {batch_end - i} documents. Total: {total_inserted}/{len(ids)}")

        print(f"[SUCCESS] All {total_inserted} chunks inserted successfully into ChromaDB.")
        print(f"[INFO] Collection '{COLLECTION_NAME}' now contains {collection.count()} documents.")

        return True

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return f"An error occurred during ChromaDB insertion: {e}"