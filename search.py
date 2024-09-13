# search.py
import os
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility

from utils import (
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM,
    embed_text_to_vector, validate_embeddings
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Constants Configuration
BATCH_SIZE = 10
NPROBE = 16
MAX_QUERY_LIMIT = 16384

def search_vectorstore(query, limit=10, path_filter="", unique=False, collection_name=None, ip_address="localhost"):
    start_time = datetime.now()
    connections.connect("default", host=ip_address, port='19530')

    results = []
    try:
        if collection_name:
            collection_name = f"documents_{collection_name}"
        else:
            collection_name = f"documents_{DEFAULT_EMBEDDING_MODEL.replace('-', '_')}"

        logging.info(f"Searching in collection: {collection_name}")

        if not utility.has_collection(collection_name):
            logging.error(f"Collection {collection_name} does not exist.")
            return results

        collection = Collection(name=collection_name)
        collection.load()

        logging.info(f"Number of entities in collection: {collection.num_entities}")

        # Assume we're using the local model
        model = LOCAL_EMBEDDING_MODEL
        dim = LOCAL_EMBEDDING_DIM

        logging.info(f"Using embedding model: {model}")

        query_vectors = embed_text_to_vector([query], model)
        validated_query_vectors = validate_embeddings(query_vectors, dim)

        if not validated_query_vectors or validated_query_vectors[0] is None:
            logging.error("Failed to generate valid query vector.")
            return results

        # Build search expression if path_filter is provided
        expr = None
        if path_filter:
            expr = f'path == "{path_filter}"'

        search_params = {
            "data": [validated_query_vectors[0]],
            "anns_field": "vector",
            "param": {"metric_type": "L2", "params": {"nprobe": NPROBE}},
            "limit": min(limit, MAX_QUERY_LIMIT),
            "output_fields": ["text", "hash", "embedding_model", "creation_date"],
            "expr": expr  # Add the expr here
        }

        search_results = collection.search(**search_params)
        logging.info(f"Number of results returned by search: {len(search_results)}")

        for hits in search_results:
            for hit in hits:
                results.append({
                    "id": hit.id,
                    "text": hit.get('text') or '',
                    "hash": hit.get('hash') or '',
                    "embedding_model": hit.get('embedding_model') or '',
                    "distance": hit.distance,
                    "creation_date": datetime.fromtimestamp(int(hit.get('creation_date') or 0)).isoformat()
                })

        # Handle unique results if requested
        if unique:
            seen_hashes = set()
            unique_results = []
            for result in results:
                if result['hash'] not in seen_hashes:
                    seen_hashes.add(result['hash'])
                    unique_results.append(result)
            results = unique_results

    except Exception as e:
        logging.error(f"An error occurred during search: {str(e)}")
    finally:
        connections.disconnect("default")
        end_time = datetime.now()
        logging.info(f"Search operation completed in {end_time - start_time}.")

    return results

if __name__ == "__main__":
    # Example usage
    query = "I want a drink"
    collection_name = "amygdala"
    results = search_vectorstore(query, limit=5, collection_name=collection_name)

    if results:
        print(f"Found {len(results)} results:")
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Distance: {result['distance']}")
            print(f"Embedding Model: {result['embedding_model']}")
            print(f"Creation Date: {result['creation_date']}")
            print("---")
    else:
        print("No results found.")
