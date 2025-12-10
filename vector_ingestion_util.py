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
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for parallel embedding

# --- 1. Environment & Client Setup ---

# Load environment variables once at the module level
load_dotenv()

# --- Configuration Mapping (Using user's EXACT environment variable names) ---
try:
    # Azure OpenAI Configuration - Building the endpoint URL exactly as specified by the user
    AZURE_ENDPOINT = f"https://{os.environ['OPENAI_EMBEDDING_RESOURCE']}.openai.azure.com/"
    AZURE_API_KEY = os.environ['OPENAI_EMBEDDING_API_KEY']
    AZURE_API_VERSION = os.environ['OPENAI_EMBEDDING_VERSION']
    AZURE_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_MODEL']
    
    # MongoDB Configuration
    MONGO_URI = os.environ['MONGO_URI']
    DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "vector_rag_db")
    VECTOR_INDEX_NAME = os.getenv("MONGO_VECTOR_INDEX_NAME", "vector_index")

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
    # Increased timeout for large bulk operations
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=60000)
    db = mongo_client[DATABASE_NAME]
    mongo_client.admin.command('ping')

except Exception as e:
    print(f"FATAL ERROR (Ingestion): Error connecting to MongoDB Atlas: {e}")
    sys.exit(1)


# ------------------------------------------------
# 2. Utility Functions
# ------------------------------------------------

def ensure_vector_search_index(db, collection_name: str, embedding_dim: int = 1536, index_name: str = "vector_index") -> bool:
    """
    Creates the required Atlas Vector Search index if it does not already exist,
    using the dedicated pymongo method (collection.create_search_index) and polling for the 'READY' status.
    """
    collection = db[collection_name]
    MAX_WAIT_TIME = 90
    POLL_INTERVAL = 10
    start_time = time.time()

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
                    },
                    {"type": "filter", "path": "metadata"}
                ]
            }

            print(f"[INFO] Creating Atlas Vector Search Index '{index_name}' on '{collection_name}'...")
            collection.create_search_index(
                model=index_definition, 
                name=index_name
            )
            print(f"[SUCCESS] Index creation command sent. Atlas is building the index.")

        while time.time() - start_time < MAX_WAIT_TIME:
            indexes = collection.list_search_indexes()
            target_index = next((i for i in indexes if i['name'] == index_name), None)

            if target_index and target_index.get('status') == 'READY':
                print(f"[SUCCESS] Index '{index_name}' is active and ready for search.")
                return True

            status = target_index.get('status', 'PENDING') if target_index else 'PENDING'
            print(f"[WAITING] Index status: {status}. Waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)

        print(f"[WARNING] Index '{index_name}' did not become active within {MAX_WAIT_TIME} seconds. Search may still fail.")
        return True 

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(f"[FATAL ERROR] Failed to create or verify Atlas Vector Search Index: {e}")
        return False

def wait_for_vector_sync(db, collection_name, inserted_id, index_name="vector_index", timeout=30):
    """
    Waits until a newly inserted document is discoverable by Atlas Search using its _id.
    This resolves the eventual consistency problem.
    """
    collection = db[collection_name]
    start_time = time.time()
    
    print(f"[SYNC CHECK] Starting sync check for new document ID: {str(inserted_id)}. Max Wait: {timeout}s")
    
    while time.time() - start_time < timeout:
        try:
            # Use Atlas Search's 'term' operator to search for the exact ID
            pipeline = [
                {
                    '$search': {
                        'index': index_name,
                        'compound': {
                            # Must search for the exact string representation of the ID
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
            # Silently ignore connection or transient errors during the sync check
            pass

        print(f"[SYNC WAIT] Document not yet indexed. Waiting 3s...")
        time.sleep(3) # Wait 3 seconds between checks
    
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
        # CRITICAL: Robustly handle potential NaN values and ensure string conversion
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
        return [[]] * len(chunk_texts) # Return empty embeddings on failure

# ------------------------------------------------
# 3. Core Ingestion Function (Updated for Multi-Source)
# ------------------------------------------------
def ingest_to_vector_db(file_path: str, sheet_name: str, collection_prefix: str = "data_source") -> Union[bool, str]:
    """
    Loads data from the given Parquet file (representing a unique logical table/sheet), 
    chunks, embeds (in parallel), and inserts into a uniquely named MongoDB Atlas Collection.
    """
    
    # 1. Determine Dynamic Collection Name and Collection Object
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sheet_name_clean = "".join(c for c in sheet_name if c.isalnum() or c in ('_',)).rstrip().lower()
    
    # Use the filename (which contains the sheet name) for the unique collection name
    COLLECTION_NAME = f"{collection_prefix}_{base_name}" 
    
    write_concern = pymongo.WriteConcern(w=1)
    # NOTE: The pymongo.MongoClient is thread-safe and can be reused across threads.
    collection = db.get_collection(COLLECTION_NAME, write_concern=write_concern) 
    
    print(f"\n--- Starting Ingestion for Collection: '{COLLECTION_NAME}' (Source: {os.path.basename(file_path)}) ---")

    # 2. Load Data (Only Parquet is expected here)
    try:
        if not file_path.endswith('.parquet'):
            print("[WARNING] Expected Parquet file path. Attempting to load.")
        
        df = pd.read_parquet(file_path)
        print(f"Data loaded: {len(df)} rows.")
        
    except FileNotFoundError:
        return f"Error: Input file not found at: {file_path}"
    except Exception as e:
        return f"Error during data loading: {e}"
    
    # 3. Chunking & Embedding (Parallelized)
    
    # a. Chunking
    chunks = chunk_dataframe_dynamic(df, max_tokens_per_chunk=1000)
    print(f"Generated {len(chunks)} chunks.")
    chunk_texts = [c["text"] for c in chunks]

    # b. Parallel Embedding using ThreadPoolExecutor
    print(f"Starting parallel vector embedding using Azure Deployment: {AZURE_DEPLOYMENT_NAME}...")
    
    # Split chunks into batches for concurrent API calls (OpenAI limit is 2048/batch)
    # Using a smaller batch size to maximize concurrency, e.g., 200 chunks per batch/thread
    BATCH_SIZE = 200 
    MAX_THREADS = 8 # Limit threads to manage rate limits
    text_batches = [chunk_texts[i:i + BATCH_SIZE] for i in range(0, len(chunk_texts), BATCH_SIZE)]
    
    all_embeddings = []
    embedding_start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_batch = {
                executor.submit(get_embedding_for_chunk, batch): batch
                for batch in text_batches
            }
            
            # Collect results in order of completion
            for future in as_completed(future_to_batch):
                batch_embeddings = future.result()
                all_embeddings.extend(batch_embeddings)
                
        embedding_duration = time.time() - embedding_start_time
        
        if not all_embeddings or any(not e for e in all_embeddings):
             # Check if any embedding failed (returned [])
             return f"Error during parallel embedding: One or more batches failed to return embeddings. Duration: {embedding_duration:.2f}s"
             
        embedding_dim = len(all_embeddings[0]) 
        print(f"Embedding complete ({len(all_embeddings)} vectors) in {embedding_duration:.2f}s. Vector dimension: {embedding_dim}")

    except Exception as e:
        return f"Error during ThreadPool or Azure OpenAI API call: {e}"

    # 4. Prepare and Insert Documents
    mongo_documents = []
    first_chunk_id = None 
    
    for i, chunk in enumerate(chunks):
        doc_id = f"{COLLECTION_NAME}_chunk_{i}"
        if i == 0:
            first_chunk_id = doc_id
            
        mongo_doc = {
            "_id": doc_id, 
            "text": chunk["text"],
            "metadata": {**chunk["metadata"], "source_file": os.path.basename(file_path)}, # Add source file to metadata
            "embedding": all_embeddings[i] # Use the parallelized embedding
        }
        mongo_documents.append(mongo_doc)

    if mongo_documents:
        try:
            # 5. MongoDB Insertion 
            print(f"Clearing old documents from '{COLLECTION_NAME}'...")
            collection.delete_many({}) 
            
            print(f"Attempting to insert {len(mongo_documents)} documents (synchronously)...")
            
            result = collection.insert_many(
                mongo_documents, 
                ordered=False, 
            )
            
            if len(result.inserted_ids) == len(mongo_documents):
                print(f"Successfully inserted {len(mongo_documents)} chunks. (Write Acknowledged)")
                
                # --- ATLAS SEARCH SYNC CHECK ---
                SYNC_TIMEOUT = 30 
                if first_chunk_id and not wait_for_vector_sync(
                    db, 
                    COLLECTION_NAME, 
                    first_chunk_id, 
                    index_name=VECTOR_INDEX_NAME, 
                    timeout=SYNC_TIMEOUT
                ):
                    return "Error: Atlas Search synchronization timed out. Retrieval will be unreliable."
                
                # --- AUTOMATICALLY ENSURE INDEX EXISTS ---
                index_success = ensure_vector_search_index(db, COLLECTION_NAME, embedding_dim=embedding_dim, index_name=VECTOR_INDEX_NAME)
                if not index_success:
                    return "Error: Data inserted, but Atlas Vector Index creation failed."
                
                return True
            else:
                return "Error: MongoDB insertion count mismatch."
            
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return f"An error occurred during MongoDB insertion: {e}"
    
    return "No documents were generated for insertion."