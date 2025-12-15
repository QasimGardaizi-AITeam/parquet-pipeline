import os
import sys
from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple, Optional

# --- 1. Configuration and Initialization ---

try:
    from config import get_config, VectorDBType
    config = get_config(VectorDBType.CHROMADB)
except ImportError:
    print("[FATAL ERROR] Cannot import 'config' module. Ensure 'config.py' is accessible.")
    sys.exit(1)


# --- Initialize Azure OpenAI client for embedding generation ---
try:
    openai_client = AzureOpenAI(
        azure_endpoint=config.azure_openai.embedding_endpoint,
        api_key=config.azure_openai.embedding_api_key,
        api_version=config.azure_openai.embedding_api_version
    )
    print(f"[INFO] Azure OpenAI client initialized for embedding (Retrieval Utility).")
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Retrieval): Error initializing Azure OpenAI client: {e}")
    sys.exit(1)


# --- Initialize ChromaDB client ---
try:
    chroma_client = chromadb.PersistentClient(
        path=config.vector_db.chromadb.persist_directory,
        settings=Settings(
            anonymized_telemetry=config.vector_db.chromadb.anonymized_telemetry 
        )
    )
    print(f"[INFO] ChromaDB client initialized for retrieval. Persistence directory: {config.vector_db.chromadb.persist_directory}")
except Exception as e:
    print(f"FATAL ERROR (ChromaDB Retrieval): Error initializing ChromaDB client: {e}")
    sys.exit(1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_logical_table_from_collection(collection_name: str, collection_prefix: str = None) -> str:
    """
    Extracts logical table name from collection name.
    Example: 'data_source_loan_Data' -> 'loan_Data'
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix
        
    if collection_name.startswith(collection_prefix + "_"):
        return collection_name[len(collection_prefix) + 1:]
    return collection_name


def map_collection_to_parquet(collection_name: str, catalog: Dict = None, collection_prefix: str = None) -> Optional[str]:
    """
    Maps a ChromaDB collection name back to its source Parquet file path.
    
    Args:
        collection_name: e.g., 'data_source_loan_Data'
        catalog: The global data catalog with logical table mappings
        collection_prefix: The prefix used in collection names
    
    Returns:
        Parquet file path or None
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix
        
    logical_table = extract_logical_table_from_collection(collection_name, collection_prefix)
    
    # If catalog provided, search for matching logical table
    if catalog:
        for table_name, table_info in catalog.items():
            if table_name == logical_table:
                # Return the parquet path from catalog
                if isinstance(table_info, dict) and 'parquet_path' in table_info:
                    return table_info['parquet_path']
                # If table_info is just a string path
                elif isinstance(table_info, str):
                    return table_info
    
    # Fallback: construct path based on naming convention
    # Adjust this based on your actual Azure storage structure
    parquet_path = f"azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/{logical_table}.parquet"
    return parquet_path


# ============================================================================
# CORE RETRIEVAL FUNCTIONS
# ============================================================================

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


def retrieve_chunks_from_chroma(collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Performs vector search in ChromaDB to find relevant chunks.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1.0 - distance

                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': score,
                    'collection': collection_name
                })

        return formatted_results

    except Exception as e:
        print(f"ERROR: ChromaDB Vector Search failed on collection '{collection_name}': {e}")
        return []


def get_relevant_collections_with_scores(
    query: str,
    collection_prefix: str = None,
    top_k: int = 10,
    score_threshold: float = 0.5,
    allowed_collections: List[str] = None
) -> List[Dict[str, any]]:
    """
    Identifies relevant collections for a query by checking relevance scores.

    Args:
        query: The search query
        collection_prefix: Prefix for collection names (uses config default if None)
        top_k: Maximum number of collections to return
        score_threshold: Minimum relevance score for a collection to be included
        allowed_collections: Optional list of collection names or logical table names
                            to filter. If provided, only these collections will be
                            checked instead of all collections in the database.
                            Can be either full collection names (e.g., 'data_source_loan_Data')
                            or logical table names (e.g., 'loan_Data').

    Returns:
        List of dicts with keys: collection_name, score, logical_table
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix

    try:
        query_vector = get_query_embedding(query)
        if not query_vector:
            return []

        all_collections = chroma_client.list_collections()
        collection_names = [col.name for col in all_collections if col.name.startswith(collection_prefix)]

        # Filter to only allowed collections if specified
        if allowed_collections:
            filtered_names = []
            for col_name in collection_names:
                logical_table = extract_logical_table_from_collection(col_name, collection_prefix)
                # Check if either the full collection name or logical table name is in allowed list
                if col_name in allowed_collections or logical_table in allowed_collections:
                    filtered_names.append(col_name)
            collection_names = filtered_names
            print(f"[INFO] Filtered to {len(collection_names)} collection(s) from allowed list: {collection_names}")

        if not collection_names:
            print(f"[WARNING] No collections found with prefix '{collection_prefix}'" +
                  (f" matching allowed list" if allowed_collections else ""))
            return []
        
        collection_scores = []
        
        for collection_name in collection_names:
            try:
                collection = chroma_client.get_collection(name=collection_name)
                
                # Quick relevance check with 1 result
                sample_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=1,
                    include=["distances"]
                )
                
                if sample_results and sample_results['distances'] and sample_results['distances'][0]:
                    distance = sample_results['distances'][0][0]
                    score = 1.0 - distance
                    
                    if score >= score_threshold:
                        logical_table = extract_logical_table_from_collection(collection_name, collection_prefix)
                        collection_scores.append({
                            'collection_name': collection_name,
                            'score': score,
                            'logical_table': logical_table
                        })
                        
            except Exception as e:
                print(f"[WARNING] Error checking collection {collection_name}: {e}")
                continue
        
        # Sort by score descending
        collection_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k
        relevant = collection_scores[:top_k]
        
        if relevant:
            print(f"[INFO] Relevant collections identified:")
            for item in relevant:
                print(f"  - {item['collection_name']} (table: {item['logical_table']}, score: {item['score']:.4f})")
        else:
            print(f"[WARNING] No collections found with score >= {score_threshold}")
        
        return relevant
        
    except Exception as e:
        print(f"[ERROR] Failed to identify relevant collections: {e}")
        return []


# MAIN RETRIEVAL FUNCTIONS
def get_semantic_context_for_query_chroma(
    query: str, 
    file_name: str, 
    collection_prefix: str = None, 
    limit: int = 10
) -> str:
    """
    LEGACY FUNCTION: Executes vector retrieval using a specific file name.
    Searches across all collections that match the base file name pattern.
    
    For new code, consider using get_semantic_context_and_files() instead.
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    base_pattern = f"{collection_prefix}_{base_name}"

    query_vector = get_query_embedding(query)
    if not query_vector:
        return ""

    try:
        all_collections = chroma_client.list_collections()
        collection_names = [col.name for col in all_collections]

        matching_collections = [
            name for name in collection_names
            if name.startswith(base_pattern)
        ]

        if not matching_collections:
            print(f"[WARNING] No ChromaDB collections found matching pattern '{base_pattern}*'")
            print(f"[INFO] Available collections: {collection_names}")
            return ""

        print(f"[INFO] Searching across {len(matching_collections)} collection(s): {matching_collections}")

        all_retrieved_docs = []
        for collection_name in matching_collections:
            docs = retrieve_chunks_from_chroma(collection_name, query_vector, limit=limit)
            all_retrieved_docs.extend(docs)

        if not all_retrieved_docs:
            print(f"[WARNING] No documents retrieved from any matching collections")
            return ""

        all_retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
        top_docs = all_retrieved_docs[:limit]

        context_list = [doc['text'] for doc in top_docs]
        full_context = "\n\n".join(context_list)

        print(f'-> SUCCESS: Semantic context retrieved from ChromaDB (Top score: {top_docs[0]["score"]:.4f})')
        return full_context

    except Exception as e:
        print(f"[ERROR] Failed to search ChromaDB collections: {e}")
        return ""


def get_semantic_context_and_files(
    query: str,
    file_name: str = None,
    catalog: Dict = None,
    collection_prefix: str = None,
    limit: int = 10,
    score_threshold: float = 0.5,
    allowed_collections: List[str] = None
) -> Tuple[str, List[str]]:
    """
    RECOMMENDED FUNCTION: Smart semantic retrieval that automatically identifies
    the most relevant collections and returns both context and parquet file paths.

    Args:
        query: The search query
        file_name: Optional specific file to search (if known)
        catalog: Global data catalog for mapping collections to parquet paths
        collection_prefix: Collection prefix (uses config default if None)
        limit: Number of results to return
        score_threshold: Minimum relevance score for collection selection
        allowed_collections: Optional list of collection names or logical table names
                            to filter. If provided, only these collections will be
                            checked instead of all collections in the database.

    Returns:
        Tuple of (context_string, list_of_parquet_paths)
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix
    
    # Get query embedding
    query_vector = get_query_embedding(query)
    if not query_vector:
        print(f"[ERROR] Failed to generate query embedding")
        return "", []
    
    # Determine which collections to search
    matching_collections = []
    
    # Strategy 1: If file_name is provided, try to use it
    if file_name:
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        base_pattern = f"{collection_prefix}_{base_name}"
        
        try:
            all_collections = chroma_client.list_collections()
            matching_collections = [
                col.name for col in all_collections 
                if col.name.startswith(base_pattern)
            ]
            
            if matching_collections:
                print(f"[INFO] Using provided file_name '{file_name}' -> collections: {matching_collections}")
            else:
                print(f"[WARNING] Provided file_name '{file_name}' not found. Using smart collection identification.")
                file_name = None  # Fall through to smart search
        except Exception as e:
            print(f"[ERROR] Failed to list collections: {e}")
            return "", []
    
    # Strategy 2: Smart collection identification based on query relevance
    if not matching_collections:
        relevant_collections_info = get_relevant_collections_with_scores(
            query=query,
            collection_prefix=collection_prefix,
            top_k=10,
            score_threshold=score_threshold,
            allowed_collections=allowed_collections
        )
        
        if not relevant_collections_info:
            # Fallback: search across all collections with the prefix (filtered if allowed_collections provided)
            try:
                all_collections = chroma_client.list_collections()
                matching_collections = [
                    col.name for col in all_collections if col.name.startswith(collection_prefix)
                ]
                # Apply allowed_collections filter in fallback too
                if allowed_collections:
                    filtered = []
                    for col_name in matching_collections:
                        logical_table = extract_logical_table_from_collection(col_name, collection_prefix)
                        if col_name in allowed_collections or logical_table in allowed_collections:
                            filtered.append(col_name)
                    matching_collections = filtered
                    print(f"[WARNING] No high-scoring collections; falling back to allowed collections: {matching_collections}")
                else:
                    print(f"[WARNING] No high-scoring collections; falling back to all collections with prefix '{collection_prefix}'")
            except Exception as e:
                print(f"[ERROR] No relevant collections found and listing failed: {e}")
                # Final fallback: legacy retrieval if a file hint exists
                if file_name:
                    legacy_ctx = get_semantic_context_for_query_chroma(
                        query=query,
                        file_name=file_name,
                        collection_prefix=collection_prefix,
                        limit=limit
                    )
                    return legacy_ctx, []
                return "", []
        else:
            matching_collections = [item['collection_name'] for item in relevant_collections_info]
            print(f"[INFO] Smart search identified {len(matching_collections)} relevant collection(s)")
    
    # Retrieve from all matching collections
    all_retrieved_docs = []
    
    for collection_name in matching_collections:
        docs = retrieve_chunks_from_chroma(collection_name, query_vector, limit=limit)
        all_retrieved_docs.extend(docs)
    
    if not all_retrieved_docs:
        print(f"[WARNING] No documents retrieved from any collection")
        # Fallback: legacy retrieval when a file hint is present
        if file_name:
            legacy_ctx = get_semantic_context_for_query_chroma(
                query=query,
                file_name=file_name,
                collection_prefix=collection_prefix,
                limit=limit
            )
            return legacy_ctx, []
        return "", []
    
    # Sort by score and take top N
    all_retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
    top_docs = all_retrieved_docs[:limit]
    
    # Build context with source attribution
    context_parts = []
    for doc in top_docs:
        source = doc['collection']
        logical_table = extract_logical_table_from_collection(source, collection_prefix)
        context_parts.append(f"[Source: {logical_table}]\n{doc['text']}")
    
    full_context = "\n\n".join(context_parts)
    
    # Map collections back to parquet paths
    parquet_paths = []
    unique_collections = list(set([doc['collection'] for doc in top_docs]))
    
    for coll_name in unique_collections:
        parquet_path = map_collection_to_parquet(coll_name, catalog, collection_prefix)
        if parquet_path and parquet_path not in parquet_paths:
            parquet_paths.append(parquet_path)
    
    print(f"[SUCCESS] Retrieved semantic context from {len(unique_collections)} collection(s)")
    print(f"  Top score: {top_docs[0]['score']:.4f} from collection: {top_docs[0]['collection']}")
    print(f"  Logical tables: {[extract_logical_table_from_collection(c, collection_prefix) for c in unique_collections]}")
    print(f"  Mapped to parquet files: {parquet_paths}")

    
    return full_context, parquet_paths

