import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict

# --- 1. Configuration and Initialization ---
load_dotenv()

try:
    AZURE_ENDPOINT = f"https://{os.environ['OPENAI_EMBEDDING_RESOURCE']}.openai.azure.com/"
    AZURE_API_KEY = os.environ['OPENAI_EMBEDDING_API_KEY']
    AZURE_API_VERSION = os.environ['OPENAI_EMBEDDING_VERSION']

    EMBEDDING_DEPLOYMENT_NAME = os.environ['OPENAI_EMBEDDING_MODEL']

    # ChromaDB specific config
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "data_source")

except KeyError as e:
    print(f"FATAL ERROR (ChromaDB Retrieval): Missing environment variable {e}.")
    sys.exit(1)

try:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Retrieval): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)

try:
    # Initialize ChromaDB client with persistence
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    print(f"[INFO] ChromaDB client initialized for retrieval. Persistence directory: {CHROMA_PERSIST_DIRECTORY}")
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Retrieval): Error initializing ChromaDB client: {e}")
    sys.exit(1)


# 2. Retrieval Functions
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


def retrieve_chunks_from_chroma(collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Performs vector search in ChromaDB to find relevant chunks.
    """
    try:
        # Get the collection
        collection = chroma_client.get_collection(name=collection_name)

        # Query the collection
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )

        # Format results to match MongoDB structure
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # ChromaDB returns distances (lower is better), convert to score (higher is better)
                # For cosine similarity: score = 1 - distance
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1.0 - distance

                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': score
                })

        return formatted_results

    except Exception as e:
        print(f"ERROR: ChromaDB Vector Search failed on collection '{collection_name}': {e}")
        return []


def get_semantic_context_for_query_chroma(query: str, file_name: str, collection_prefix: str = None, limit: int = 7) -> str:
    """
    Executes the full vector retrieval process using ChromaDB and returns the concatenated text context,
    dynamically determining the collection name from the file name.
    """

    if collection_prefix is None:
        collection_prefix = CHROMA_COLLECTION_PREFIX

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    collection_name = f"{collection_prefix}_{base_name}"

    # Ensure collection name meets ChromaDB requirements
    collection_name = collection_name.replace('-', '_')[:63]

    query_vector = get_query_embedding(query)
    if not query_vector:
        return ""

    retrieved_docs = retrieve_chunks_from_chroma(collection_name, query_vector, limit=limit)

    if not retrieved_docs:
        print(f"[WARNING] No documents retrieved from ChromaDB collection '{collection_name}'")
        return ""

    # 4. Process Retrieved Documents and return concatenated text
    context_list = [doc['text'] for doc in retrieved_docs]
    full_context = "\n\n".join(context_list)

    print(f'-> SUCCESS: Semantic context retrieved from ChromaDB (Score: {retrieved_docs[0]["score"]:.4f}). Context preview:')

    return full_context


def list_chroma_collections() -> List[str]:
    """
    Lists all available collections in ChromaDB.
    Useful for debugging and understanding what data is available.
    """
    try:
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"[INFO] Available ChromaDB collections: {collection_names}")
        return collection_names
    except Exception as e:
        print(f"[ERROR] Failed to list ChromaDB collections: {e}")
        return []


def get_collection_info(collection_name: str) -> Dict:
    """
    Gets information about a specific ChromaDB collection.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        metadata = collection.metadata

        info = {
            "name": collection_name,
            "count": count,
            "metadata": metadata
        }

        print(f"[INFO] Collection '{collection_name}': {count} documents, metadata: {metadata}")
        return info

    except Exception as e:
        print(f"[ERROR] Failed to get info for collection '{collection_name}': {e}")
        return {}
