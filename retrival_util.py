import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymongo import MongoClient
from typing import List, Dict

# --- 1. Configuration and Initialization ---

load_dotenv()

# --- Configuration (Must match the ingestion script) ---
try:
    # Azure OpenAI Configuration
    AZURE_ENDPOINT = f"https://{os.environ['OPENAI_EMBEDDING_RESOURCE']}.openai.azure.com/"
    AZURE_API_KEY = os.environ['OPENAI_EMBEDDING_API_KEY']
    AZURE_API_VERSION = os.environ['OPENAI_EMBEDDING_VERSION']
    
    # Embedding Model (Used to vectorize the query)
    EMBEDDING_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_MODEL']
    
    # MongoDB Configuration
    MONGO_URI = os.environ['MONGO_URI']
    DATABASE_NAME = "vector_rag_db"
    VECTOR_INDEX_NAME = "vector_index" 
    
except KeyError as e:
    print(f"FATAL ERROR (Retrieval): Missing environment variable {e}.")
    sys.exit(1)

# Initialize Clients
try:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
except Exception as e:
    print(f"FATAL ERROR (Retrieval): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    mongo_client.admin.command('ping')
except Exception as e:
    print(f"FATAL ERROR (Retrieval): Error connecting to MongoDB Atlas: {e}")
    sys.exit(1)


# ------------------------------------------------
# 2. Retrieval Functions
# ------------------------------------------------

def get_query_embedding(query: str) -> List[float]:
    """Generates the vector embedding for the query using Azure OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_NAME,
            input=query,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"ERROR: Failed to create query embedding: {e}")
        return []

def retrieve_chunks(collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Performs vector search in MongoDB Atlas to find relevant chunks.
    """
    collection = db[collection_name]
    
    # The MongoDB Aggregation Pipeline for Vector Search
    pipeline = [
        {
            # Stage 1: Perform the vector search
            '$vectorSearch': {
                'index': VECTOR_INDEX_NAME,
                'path': 'embedding',         
                'queryVector': query_vector,
                'numCandidates': 50,         
                'limit': limit              
            }
        },
        {
            # Stage 2: Project (include) the text and the score
            '$project': {
                '_id': 0,
                'text': 1,
                'metadata': 1,
                'score': {'$meta': 'vectorSearchScore'}
            }
        }
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"ERROR: MongoDB Vector Search failed on collection '{collection_name}': {e}")
        return []


def get_semantic_context_for_query(query: str, file_name: str, collection_prefix: str = "data_source", limit: int = 7) -> str:
    """
    Executes the full vector retrieval process and returns the concatenated text context, 
    dynamically determining the collection name from the file name.
    """
    
    # 1. Determine Dynamic Collection Name
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    collection_name = f"{collection_prefix}_{base_name}"
    
    # 2. Embed the Query
    query_vector = get_query_embedding(query)
    if not query_vector:
        return ""

    # 3. Retrieve Context from MongoDB Atlas
    retrieved_docs = retrieve_chunks(collection_name, query_vector, limit=limit)
    
    if not retrieved_docs:
        return ""

    # 4. Process Retrieved Documents and return concatenated text
    context_list = [doc['text'] for doc in retrieved_docs]
    full_context = "\n\n".join(context_list)
    
    # --- DEBUGGING LINE ADDED HERE ---
    print(f'-> SUCCESS: Semantic context retrieved (Score: {retrieved_docs[0]["score"]:.4f}). Context preview:')
    print('context: ',full_context) 
    # ---------------------------------
    
    return full_context