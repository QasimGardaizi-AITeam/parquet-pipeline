import os
import sys
import pandas as pd
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient
import pymongo 
from typing import List, Dict, Any, Union, Optional
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
    
    MONGO_URI = os.environ['MONGO_URI']
    DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "vector_rag_db")
    VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index")
    
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

except KeyError as e:
    print(f"FATAL ERROR (Ingestion): Missing environment variable {e}. Ingestion cannot proceed.")
    sys.exit(1)

try:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
except Exception as e:
    print(f"FATAL ERROR (Ingestion): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

try:
    # CRITICAL FIX: Increase timeouts and add retry logic
    mongo_client = MongoClient(
        MONGO_URI, 
        serverSelectionTimeoutMS=60000,
        socketTimeoutMS=120000,  # NEW: Socket timeout (2 minutes)
        connectTimeoutMS=30000,   # NEW: Connection timeout
        retryWrites=True,         # NEW: Enable retry writes
        maxPoolSize=10            # NEW: Connection pool size
    )
    db = mongo_client[DATABASE_NAME]
    mongo_client.admin.command('ping')
    print("[INFO] MongoDB connection established successfully.")
except Exception as e:
    print(f"FATAL ERROR (Ingestion): Error connecting to MongoDB Atlas: {e}")
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

def ensure_vector_search_index(db, collection_name: str, embedding_dim: int = 1536, index_name: str = "vector_index") -> bool:
    """
    Creates the required Atlas Vector Search index if it does not already exist.
    """
    collection = db[collection_name]
    MAX_WAIT_TIME = 120
    POLL_INTERVAL = 10
    start_time = time.time()
    
    index_creation_requested = False

    try:
        existing_indexes = [index['name'] for index in collection.list_search_indexes()]

        if index_name in existing_indexes:
            print(f"[INFO] Atlas Vector Search Index '{index_name}' already exists on '{collection_name}'. Checking status...")
        else:
            index_definition = {
                "type": "vectorSearch",
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": embedding_dim,
                        "similarity": "cosine"
                    }
                ]
            }

            print(f"[INFO] Creating Atlas Vector Search Index '{index_name}' on '{collection_name}'...")
            collection.create_search_index( 
                name=index_name,
                definition=index_definition
            )
            index_creation_requested = True
            print(f"[SUCCESS] Index creation command sent. Atlas is building the index.")

        while time.time() - start_time < MAX_WAIT_TIME:
            try:
                indexes = collection.list_search_indexes()
                target_index = next((i for i in indexes if i['name'] == index_name), None)

                if target_index and target_index.get('status') == 'READY':
                    print(f"[SUCCESS] Index '{index_name}' is active and ready for search.")
                    return True

                status = target_index.get('status', 'PENDING') if target_index else 'PENDING'
                print(f"[WAITING] Index status: {status}. Waiting {POLL_INTERVAL}s...")
            except Exception as conn_err:
                print(f"[WARNING] Transient MongoDB error during index status check: {conn_err}")
                 
            time.sleep(POLL_INTERVAL)

        if index_creation_requested:
            print(f"[FATAL ERROR] Index '{index_name}' did not become active within {MAX_WAIT_TIME} seconds.")
            return False
        
        print(f"[WARNING] Index '{index_name}' did not become active within {MAX_WAIT_TIME} seconds.")
        return True

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(f"[FATAL ERROR] Failed to create or verify Atlas Vector Search Index: {e}")
        return False

def wait_for_vector_sync(db, collection_name, inserted_id, index_name="vector_index", timeout=45):
    """
    Waits until a newly inserted document is discoverable by Atlas Search.
    """
    collection = db[collection_name]
    start_time = time.time()
    
    print(f"[SYNC CHECK] Starting sync check for new document ID: {str(inserted_id)}. Max Wait: {timeout}s")
    
    while time.time() - start_time < timeout:
        try:
            pipeline = [
                {
                    '$search': {
                        'index': index_name,
                        'compound': {
                            'must': [{'term': {'query': str(inserted_id), 'path': '_id'}}]
                        }
                    }
                },
                {'$limit': 1}
            ]
            
            result = list(collection.aggregate(pipeline))
            
            if result:
                print(f"[SYNC SUCCESS] Document ID found in Atlas Search after {time.time() - start_time:.2f} seconds.")
                return True
            
        except Exception:
            pass

        print(f"[SYNC WAIT] Document not yet indexed. Waiting 3s...")
        time.sleep(3)
    
    print(f"[SYNC ERROR] Timeout waiting for document ID {str(inserted_id)} to be indexed in Atlas Search.")
    return False

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
# NEW: Batched MongoDB Insertion with Progress
# ------------------------------------------------

def insert_documents_in_batches(collection, documents: List[Dict], batch_size: int = 100) -> bool:
    """
    Inserts documents in smaller batches with progress tracking to avoid timeouts.
    """
    total_docs = len(documents)
    total_inserted = 0
    
    print(f"[INFO] Inserting {total_docs} documents in batches of {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        try:
            result = collection.insert_many(batch, ordered=False)
            total_inserted += len(result.inserted_ids)
            print(f"[PROGRESS] Batch {batch_num}/{total_batches}: Inserted {len(result.inserted_ids)} documents. Total: {total_inserted}/{total_docs}")
            
        except pymongo.errors.BulkWriteError as bwe:
            # Handle partial success in bulk writes
            inserted_count = bwe.details.get('nInserted', 0)
            total_inserted += inserted_count
            print(f"[WARNING] Batch {batch_num} partial success: {inserted_count} inserted, {len(bwe.details.get('writeErrors', []))} errors")
            
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed completely: {e}")
            return False
    
    print(f"[SUCCESS] Total documents inserted: {total_inserted}/{total_docs}")
    return total_inserted == total_docs

# ------------------------------------------------
# 3. Core Ingestion Function (FIXED)
# ------------------------------------------------
def ingest_to_vector_db(file_path: str, sheet_name: str = None, collection_prefix: str = "data_source") -> Union[bool, str]:
    """
    Loads data from the given Parquet file (local or Azure URI), 
    chunks, embeds (in parallel), and inserts into MongoDB Atlas.
    """
    
    # 1. Determine Dynamic Collection Name
    base_name = os.path.splitext(os.path.basename(file_path.split('/')[-1]))[0]
    
    if sheet_name:
        sheet_name_clean = "".join(c for c in sheet_name if c.isalnum() or c in ('_',)).rstrip().lower()
        COLLECTION_NAME = f"{collection_prefix}_{base_name}"
    else:
        COLLECTION_NAME = f"{collection_prefix}_{base_name}"
    
    # CRITICAL FIX: Use weaker write concern to avoid blocking
    write_concern = pymongo.WriteConcern(w=1, j=False)  # j=False = don't wait for journal
    collection = db.get_collection(COLLECTION_NAME, write_concern=write_concern) 
    
    print(f"\n--- Starting Ingestion for Collection: '{COLLECTION_NAME}' (Source: {os.path.basename(file_path)}) ---")

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

    # 4. Prepare Documents
    mongo_documents = []
    first_chunk_id = None 
    
    for i, chunk in enumerate(chunks):
        doc_id = f"{COLLECTION_NAME}_chunk_{i}"
        if i == 0:
            first_chunk_id = doc_id
            
        mongo_doc = {
            "_id": doc_id, 
            "text": chunk["text"],
            "metadata": {**chunk["metadata"], "source_file": os.path.basename(file_path)},
            "embedding": all_embeddings[i]
        }
        mongo_documents.append(mongo_doc)

    if mongo_documents:
        try:
            # 5. Clear old data
            print(f"Clearing old documents from '{COLLECTION_NAME}'...")
            delete_result = collection.delete_many({})
            print(f"Deleted {delete_result.deleted_count} old documents.")
            
            # 6. CRITICAL FIX: Use batched insertion
            insertion_success = insert_documents_in_batches(
                collection, 
                mongo_documents, 
                batch_size=100  # Smaller batches to avoid timeout
            )
            
            if not insertion_success:
                return "Error: MongoDB insertion failed or incomplete."
            
            print(f"[SUCCESS] All {len(mongo_documents)} chunks inserted successfully.")
            
            # 7. Skip sync check if too many documents (it's slow)
            if len(mongo_documents) <= 100:
                SYNC_TIMEOUT = 30
                if first_chunk_id and not wait_for_vector_sync(
                    db, 
                    COLLECTION_NAME, 
                    first_chunk_id, 
                    index_name=VECTOR_INDEX_NAME, 
                    timeout=SYNC_TIMEOUT
                ):
                    print("[WARNING] Atlas Search sync timed out, but continuing...")
            else:
                print(f"[INFO] Skipping sync check for large dataset ({len(mongo_documents)} docs). Index will sync in background.")
            
            # 8. Ensure index exists
            index_success = ensure_vector_search_index(db, COLLECTION_NAME, embedding_dim=embedding_dim, index_name=VECTOR_INDEX_NAME)
            if not index_success:
                return "Error: Data inserted, but Atlas Vector Index creation failed."
            
            return True
            
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return f"An error occurred during MongoDB insertion: {e}"
    
    return "No documents were generated for insertion."