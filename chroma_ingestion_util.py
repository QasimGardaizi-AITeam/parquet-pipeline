import os
import sys
import pandas as pd
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Union
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import duckdb

# --- 1. Environment & Client Setup ---

load_dotenv()

# --- Configuration Mapping ---
try:
    AZURE_ENDPOINT = f"https://{os.environ['OPENAI_EMBEDDING_RESOURCE']}.openai.azure.com/"
    AZURE_API_KEY = os.environ['OPENAI_EMBEDDING_API_KEY']
    AZURE_API_VERSION = os.environ['OPENAI_EMBEDDING_VERSION']
    AZURE_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_MODEL']

    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    # ChromaDB specific config
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "data_source")

except KeyError as e:
    print(f"FATAL ERROR (ChromaDB Ingestion): Missing environment variable {e}. Ingestion cannot proceed.")
    sys.exit(1)

try:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Ingestion): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

try:
    # Initialize ChromaDB client with persistence
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    print(f"[INFO] ChromaDB client initialized. Persistence directory: {CHROMA_PERSIST_DIRECTORY}")
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

            if AZURE_STORAGE_CONNECTION_STRING:
                escaped_conn_str = AZURE_STORAGE_CONNECTION_STRING.replace("'", "''")
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


def get_embedding_for_chunk(chunk_texts: List[str]) -> List[List[float]]:
    """Worker function to get embeddings for a batch of chunk texts."""
    try:
        response = openai_client.embeddings.create(
            model=AZURE_DEPLOYMENT_NAME,
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

    if collection_prefix is None:
        collection_prefix = CHROMA_COLLECTION_PREFIX

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

    print(f"Starting parallel vector embedding using Azure Deployment: {AZURE_DEPLOYMENT_NAME}...")

    BATCH_SIZE = 200
    MAX_THREADS = 8
    text_batches = [chunk_texts[i:i + BATCH_SIZE] for i in range(0, len(chunk_texts), BATCH_SIZE)]

    all_embeddings = []
    embedding_start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_batch = {
                executor.submit(get_embedding_for_chunk, batch): batch
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
