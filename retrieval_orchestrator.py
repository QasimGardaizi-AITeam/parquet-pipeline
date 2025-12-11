"""
Orchestrates semantic retrieval with proper file identification
"""
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata"""
    context: str
    source_files: List[str]
    top_score: float
    collection_names: List[str]


def extract_logical_table_from_collection(collection_name: str, collection_prefix: str = "data_source") -> str:
    """
    Extracts logical table name from collection name.
    Example: 'data_source_loan_Data' -> 'loan_Data'
    """
    if collection_name.startswith(collection_prefix + "_"):
        return collection_name[len(collection_prefix) + 1:]
    return collection_name


def map_collection_to_parquet(collection_name: str, catalog: Dict[str, any], collection_prefix: str = "data_source") -> Optional[str]:
    """
    Maps a ChromaDB collection name back to its source Parquet file path.
    
    Args:
        collection_name: e.g., 'data_source_loan_Data'
        catalog: The global data catalog with logical table mappings
        collection_prefix: The prefix used in collection names
    
    Returns:
        Parquet file path or None
    """
    logical_table = extract_logical_table_from_collection(collection_name, collection_prefix)
    
    # Search in catalog for matching logical table
    for table_name, table_info in catalog.items():
        if table_name == logical_table:
            # Return the parquet path from catalog
            if 'parquet_path' in table_info:
                return table_info['parquet_path']
            # If catalog structure is different, adapt this
            
    return None


def get_relevant_collections_with_metadata(
    query: str,
    chroma_client,
    openai_client,
    config,
    collection_prefix: str = None,
    top_k: int = 3,
    score_threshold: float = 0.5
) -> List[Dict[str, any]]:
    """
    Identifies relevant collections for a query and returns metadata about each.
    
    Returns:
        List of dicts with keys: collection_name, score, logical_table
    """
    if collection_prefix is None:
        collection_prefix = config.vector_db.chromadb.collection_prefix
    
    try:
        # Get query embedding
        response = openai_client.embeddings.create(
            model=config.azure_openai.embedding_deployment_name,
            input=query,
        )
        query_vector = response.data[0].embedding
        
        # Get all collections
        all_collections = chroma_client.list_collections()
        collection_names = [col.name for col in all_collections if col.name.startswith(collection_prefix)]
        
        if not collection_names:
            print(f"[WARNING] No collections found with prefix '{collection_prefix}'")
            return []
        
        collection_scores = []
        
        for collection_name in collection_names:
            try:
                collection = chroma_client.get_collection(name=collection_name)
                
                # Query with small sample to check relevance
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
        
        return relevant
        
    except Exception as e:
        print(f"[ERROR] Failed to identify relevant collections: {e}")
        return []


def retrieve_semantic_context_smart(
    query: str,
    chroma_client,
    openai_client,
    config,
    catalog: Dict[str, any] = None,
    file_name: str = None,
    limit: int = 7
) -> Tuple[str, List[str]]:
    """
    Smart semantic retrieval that identifies correct collections and returns
    both context and the parquet file paths that should be queried.
    
    Returns:
        Tuple of (context_string, list_of_parquet_paths)
    """
    collection_prefix = config.vector_db.chromadb.collection_prefix
    
    # Get query embedding
    try:
        response = openai_client.embeddings.create(
            model=config.azure_openai.embedding_deployment_name,
            input=query,
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Failed to create query embedding: {e}")
        return "", []
    
    # Determine which collections to search
    matching_collections = []
    
    if file_name:
        # If file_name provided, try to use it
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        base_pattern = f"{collection_prefix}_{base_name}"
        
        all_collections = chroma_client.list_collections()
        matching_collections = [
            col.name for col in all_collections 
            if col.name.startswith(base_pattern)
        ]
        
        if matching_collections:
            print(f"[INFO] Using provided file_name '{file_name}' -> collections: {matching_collections}")
        else:
            print(f"[WARNING] Provided file_name '{file_name}' not found in collections")
            file_name = None  # Fall through to smart search
    
    if not matching_collections:
        # Smart collection identification
        relevant_collections_info = get_relevant_collections_with_metadata(
            query, chroma_client, openai_client, config, 
            collection_prefix=collection_prefix,
            top_k=3,
            score_threshold=0.5
        )
        
        if not relevant_collections_info:
            print(f"[ERROR] No relevant collections found for query")
            return "", []
        
        matching_collections = [item['collection_name'] for item in relevant_collections_info]
    
    # Retrieve from all matching collections
    all_retrieved_docs = []
    
    for collection_name in matching_collections:
        try:
            collection = chroma_client.get_collection(name=collection_name)
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1.0 - distance
                    
                    all_retrieved_docs.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': score,
                        'collection': collection_name
                    })
                    
        except Exception as e:
            print(f"[ERROR] Failed to retrieve from collection '{collection_name}': {e}")
            continue
    
    if not all_retrieved_docs:
        print(f"[WARNING] No documents retrieved from any collection")
        return "", []
    
    # Sort by score and take top N
    all_retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
    top_docs = all_retrieved_docs[:limit]
    
    # Build context
    context_parts = []
    for doc in top_docs:
        source = doc['collection']
        context_parts.append(f"[Source: {source}]\n{doc['text']}")
    
    full_context = "\n\n".join(context_parts)
    
    # Map collections back to parquet paths
    parquet_paths = []
    unique_collections = list(set([doc['collection'] for doc in top_docs]))
    
    if catalog:
        for coll_name in unique_collections:
            parquet_path = map_collection_to_parquet(coll_name, catalog, collection_prefix)
            if parquet_path and parquet_path not in parquet_paths:
                parquet_paths.append(parquet_path)
    else:
        # Fallback: construct parquet path from collection name
        for coll_name in unique_collections:
            logical_table = extract_logical_table_from_collection(coll_name, collection_prefix)
            # This assumes a naming convention - adjust based on your actual paths
            parquet_path = f"azure://auxeestorage.blob.core.windows.net/auxee-upload-files/parquet_files/{logical_table}.parquet"
            if parquet_path not in parquet_paths:
                parquet_paths.append(parquet_path)
    
    print(f"[SUCCESS] Retrieved semantic context from {len(unique_collections)} collection(s)")
    print(f"  Top score: {top_docs[0]['score']:.4f} from {top_docs[0]['collection']}")
    print(f"  Mapped to parquet files: {parquet_paths}")
    
    return full_context, parquet_paths