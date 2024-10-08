#!/usr/bin/env python3
import sys
import json
import logging
import os
from datetime import datetime
from openai import OpenAI
from pymilvus import connections, Collection, utility
import nltk
import argparse

# Constants Configuration
BATCH_SIZE = 10
NPROBE = 16
INITIAL_QUERY_LIMIT = 100
MAX_QUERY_LIMIT = 16384

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MODELS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072
}

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Downloading required NLP model
nltk.download('punkt', quiet=True)

def embed_text_to_vector(text_chunks, model):
    vectors = []
    for i in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[i:i + BATCH_SIZE]
        try:
            response = client.embeddings.create(model=model, input=batch)
            batch_vectors = [data.embedding for data in response.data]
            vectors.extend(batch_vectors)
        except Exception as e:
            logging.error(f"Embedding generation failed for batch index {i} using model {model}: {e}")
            vectors.extend([None] * len(batch))
    return vectors

def validate_embeddings(vectors, expected_dim):
    valid_vectors = []
    for v in vectors:
        if isinstance(v, list) and len(v) == expected_dim:
            valid_vectors.append(v)
        else:
            logging.error(f"Invalid vector dimension received. Expected: {expected_dim}, Received: {len(v) if isinstance(v, list) else 'None'}")
    return valid_vectors

def perform_search(collection, query_vector, limit):
    logging.debug(f"Searching with vector: {query_vector[:10]}... (truncated for log)")
    try:
        search_params = {
            "data": [query_vector],
            "anns_field": "vector",
            "param": {"metric_type": "L2", "params": {"nprobe": NPROBE}},
            "limit": min(limit, MAX_QUERY_LIMIT),
            "output_fields": ["path", "snippet", "embedding_model", "creation_date"]
        }
        
        results = collection.search(**search_params)
        return results[0] if results else []
    except Exception as e:
        logging.error(f"Search execution failed: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Query the vector store using Milvus.")
    parser.add_argument("query", type=str, nargs='?', help="Query to perform on the loaded documents")
    parser.add_argument("-l", "--limit", type=int, default=10, help="Number of results to return")
    parser.add_argument("-p", "--path", type=str, default="", help="Filter results by path containing this string")
    parser.add_argument("-u", "--unique", action="store_true", help="Return only unique file paths")
    parser.add_argument("--collection", type=str, help="Specify a single collection to search in")
    parser.add_argument("--ip-address", type=str, default="localhost", help="IP address of the Milvus server")
    parser.add_argument("--use-stdin", action="store_true", help="Read JSON input from stdin instead of using command-line arguments")
    parser.add_argument("--json", type=str, help="Pass JSON data as a command-line argument")

    args = parser.parse_args()

    # Determine the source of input data
    if args.json:
        try:
            input_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON input from --json argument: {e}")
            sys.exit(1)
    elif args.use_stdin or not args.query:
        try:
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON input from stdin: {e}")
            sys.exit(1)
    else:
        # Default to using command-line arguments
        input_data = {
            "query": args.query,
            "limit": args.limit,
            "path": args.path,
            "unique": args.unique,
            "collection": args.collection,
            "ip_address": args.ip_address
        }

    # Extract variables from input_data
    query = input_data.get('query')
    limit = input_data.get('limit', 10)
    path_filter = input_data.get('path', "")
    unique = input_data.get('unique', False)
    collection_name = input_data.get('collection')
    ip_address = input_data.get('ip_address', "localhost")

    start_time = datetime.now()
    connections.connect("default", host=ip_address, port='19530')
    
    results = []
    try:
        collections_to_search = [collection_name] if collection_name else [f"documents_{model.replace('-', '_')}" for model in EMBEDDING_MODELS.keys()]

        for collection_name in collections_to_search:
            if not utility.has_collection(collection_name):
                logging.error(f"Collection {collection_name} does not exist. Skipping.")
                continue
            
            collection = Collection(name=collection_name)
            collection.load()
            
            logging.info(f"Number of entities in collection: {collection.num_entities}")

            model = next((m for m in EMBEDDING_MODELS.keys() if m.replace('-', '_') in collection_name), DEFAULT_EMBEDDING_MODEL)
            dim = EMBEDDING_MODELS[model]
            
            logging.info(f"Using embedding model: {model} for collection: {collection_name}")

            query_vectors = embed_text_to_vector([query], model)
            validated_query_vectors = validate_embeddings(query_vectors, dim)

            model_results = perform_search(collection, validated_query_vectors[0], MAX_QUERY_LIMIT)
            logging.info(f"Number of results returned by search: {len(model_results)}")
            
            # Log the first result for debugging
            if model_results:
                logging.info(f"First result: {model_results[0]}")
                logging.info(f"First result entity: {model_results[0].entity}")
            
            # Filter results by path if specified
            filtered_results = [
                {
                    "id": result.id,
                    "path": getattr(result.entity, 'path', ''),
                    "snippet": getattr(result.entity, 'snippet', ''),
                    "embedding_model": getattr(result.entity, 'embedding_model', ''),
                    "distance": result.distance,
                    "creation_date": datetime.fromtimestamp(int(getattr(result.entity, 'creation_date', 0))).isoformat()
                }
                for result in model_results
                if path_filter in getattr(result.entity, 'path', '')
            ]
            
            results.extend(filtered_results)

        # Apply uniqueness filter if requested
        if unique:
            seen_paths = set()
            unique_results = []
            for result in results:
                if result["path"] not in seen_paths:
                    unique_results.append(result)
                    seen_paths.add(result["path"])
            results = unique_results
        
        # Sort results by distance and limit to the requested number
        results = sorted(results, key=lambda x: x['distance'])[:limit]
        
        # Output JSON results to stdout
        json.dump(results, sys.stdout, indent=4)
    finally:
        connections.disconnect("default")
        end_time = datetime.now()
        logging.info(f"Operation completed in {end_time - start_time}.")

if __name__ == "__main__":
    main()
