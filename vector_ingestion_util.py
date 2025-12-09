import os
import sys
import pandas as pd
import time # CRITICAL: Ensure this is imported
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient
import pymongo 
from typing import List, Dict, Any, Union
import traceback 

# --- 1. Environment & Client Setup ---

# Load environment variables once at the module level
load_dotenv()

# --- Configuration Mapping ---
try:
    # Azure OpenAI Configuration
    AZURE_ENDPOINT = f"https://{os.environ['OPENAI_EMBEDDING_RESOURCE']}.openai.azure.com/"
    AZURE_API_KEY = os.environ['OPENAI_EMBEDDING_API_KEY']
    AZURE_API_VERSION = os.environ['OPENAI_EMBEDDING_VERSION']
    AZURE_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_MODEL']
    
    # MongoDB Configuration
    MONGO_URI = os.environ['MONGO_URI']
    DATABASE_NAME = "vector_rag_db" 
    VECTOR_INDEX_NAME = "vector_index" # Ensure this variable exists for the sync function
    
except KeyError as e:
    print(f"FATAL ERROR (Ingestion): Missing environment variable {e}. Ingestion cannot proceed.")
    sys.exit(1)

# Initialize Clients
# ... (Client initialization code remains the same) ...
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

# --- NEW FUNCTION FOR READ-AFTER-WRITE SYNC ---
def wait_for_vector_sync(db, collection_name, inserted_id, index_name="vector_index", timeout=30):
    """
    Waits until a newly inserted document is discoverable by Atlas Search using its _id.
    This resolves the eventual consistency problem.
    """
    collection = db[collection_name]
    start_time = time.time()
    
    print(f"[SYNC CHECK] Starting sync check for new document ID: {str(inserted_id)}.")
    
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

        print(f"[SYNC WAIT] Document not yet indexed. Waiting 2s...")
        time.sleep(2) # Wait 2 seconds between checks
    
    print(f"[SYNC ERROR] Timeout waiting for document ID {str(inserted_id)} to be indexed in Atlas Search.")
    return False

# ... (chunk_dataframe_dynamic remains the same) ...
def chunk_dataframe_dynamic(df: pd.DataFrame, max_tokens_per_chunk: int = 1000) -> List[Dict[str, Any]]:
    """Dynamically chunk a DataFrame into text blocks."""
    chunks = []
    headers = list(df.columns)
    current_chunk = []
    current_indices = []
    current_size = 0

    for idx, row in df.iterrows():
        # CRITICAL: Ensure row values are converted to string for reliable chunking
        row_text = " | ".join([f"{col}:{str(row[col])}" for col in headers]) 
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

# ------------------------------------------------
# 3. Core Ingestion Function (Updated)
# ------------------------------------------------
def ingest_to_vector_db(file_path: str, collection_prefix: str = "data_source") -> Union[bool, str]:
    """
    Loads data from the given file, chunks, embeds, and inserts into MongoDB Atlas.
    """
    
    # 1. Determine Dynamic Collection Name and Collection Object
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    COLLECTION_NAME = f"{collection_prefix}_{base_name}"
    
    write_concern = pymongo.WriteConcern(w=1)
    collection = db.get_collection(COLLECTION_NAME, write_concern=write_concern)
    
    print(f"\n--- Starting Ingestion for Collection: '{COLLECTION_NAME}' ---")

    # 2. Load Data (Remains the same)
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            return f"Error: Unsupported file format for {file_path}. Must be .xlsx or .parquet."
            
        print(f"Data loaded: {len(df)} rows.")
        
    except FileNotFoundError:
        return f"Error: Input file not found at: {file_path}"
    except Exception as e:
        return f"Error during data loading: {e}"
    
    # ... (Chunking and Embedding remains the same) ...
    # 3. Chunking & Embedding
    chunks = chunk_dataframe_dynamic(df, max_tokens_per_chunk=1000)
    print(f"Generated {len(chunks)} chunks.")
    chunk_texts = [c["text"] for c in chunks]

    print(f"Starting vector embedding using Azure Deployment: {AZURE_DEPLOYMENT_NAME}...")
    try:
        response = openai_client.embeddings.create(
            model=AZURE_DEPLOYMENT_NAME,
            input=chunk_texts,
        )
        chunk_embeddings = [data.embedding for data in response.data]
        embedding_dim = len(chunk_embeddings[0]) 
        print(f"Embedding complete. Vector dimension: {embedding_dim}")

    except Exception as e:
        return f"Error during Azure OpenAI API call: {e}"

    # 4. Prepare and Insert Documents
    mongo_documents = []
    # --- Capture the first ID for synchronization check ---
    first_chunk_id = None 
    
    for i, chunk in enumerate(chunks):
        doc_id = f"{COLLECTION_NAME}_chunk_{i}"
        if i == 0:
            first_chunk_id = doc_id
            
        mongo_doc = {
            "_id": doc_id, 
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": chunk_embeddings[i]
        }
        mongo_documents.append(mongo_doc)
        
    print(f"Preparing to insert {len(mongo_documents)} documents into MongoDB...")
    # ... (rest of prep remains the same) ...

    if mongo_documents:
        try:
            # 5. MongoDB Insertion 
            print(f"Clearing old documents from '{COLLECTION_NAME}'...")
            collection.delete_many({}) 
            
            print(f"Attempting to insert {len(mongo_documents)} documents (synchronously)...")
            
            result = collection.insert_many(
                mongo_documents, 
                ordered=False, # Set to False for faster insertion
            )
            
            # Explicitly check if insertion was successful
            if len(result.inserted_ids) == len(mongo_documents):
                print(f"Successfully inserted {len(mongo_documents)} chunks. (Write Acknowledged)")
                
                # --- CRITICAL FIX: ATLAS SEARCH SYNC CHECK ---
                if first_chunk_id and not wait_for_vector_sync(db, COLLECTION_NAME, first_chunk_id, index_name=VECTOR_INDEX_NAME):
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