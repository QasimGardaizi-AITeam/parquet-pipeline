import os
import sys
from openai import AzureOpenAI
from pymongo import MongoClient
from typing import List, Dict

# --- 1. Configuration & Client Setup ---

from config import get_config, VectorDBType

# Get config
config = get_config(VectorDBType.MONGODB)

try:
    openai_client = AzureOpenAI(
        azure_endpoint=config.azure_openai.embedding_endpoint,
        api_key=config.azure_openai.embedding_api_key,
        api_version=config.azure_openai.embedding_api_version
    )
except Exception as e:
    print(f"FATAL ERROR (Retrieval): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

try:
    mongo_client = MongoClient(config.vector_db.mongodb.uri)
    db = mongo_client[config.vector_db.mongodb.database_name]
    mongo_client.admin.command('ping')
except Exception as e:
    print(f"FATAL ERROR (Retrieval): Error connecting to MongoDB Atlas: {e}")
    sys.exit(1)



# 2. Retrieval Functions
def get_query_embedding(query: str) -> List[float]:
    """Generates the vector embedding for the query using Azure OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model=config.azure_openai.embedding_deployment_name,
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
    
    pipeline = [
        {
            # Stage 1: Perform the vector search
            '$vectorSearch': {
                'index': config.vector_db.mongodb.vector_index_name,
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
    
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    collection_name = f"{collection_prefix}_{base_name}"
    
    query_vector = get_query_embedding(query)
    if not query_vector:
        return ""

    retrieved_docs = retrieve_chunks(collection_name, query_vector, limit=limit)
    
    if not retrieved_docs:
        return ""

    # 4. Process Retrieved Documents and return concatenated text
    context_list = [doc['text'] for doc in retrieved_docs]
    full_context = "\n\n".join(context_list)
    
    print(f'-> SUCCESS: Semantic context retrieved (Score: {retrieved_docs[0]["score"]:.4f}). Context preview:')
    
    return full_context