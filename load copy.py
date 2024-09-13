#!/usr/bin/env python3
import argparse
import os
import logging
from datetime import datetime
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm
import traceback

from utils import (
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIM,
    SNIPPET_LENGTH, INDEX_TYPE, METRIC_TYPE, NLIST,
    load_local_model, embed_text_to_vector, validate_embeddings, count_files, load_files,
    ensure_collection_exists, clear_collection, delete_old_entries, process_file, extract_snippet,
    file_hash, get_creation_date, process_and_insert_lines
)

# Logging Configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    start_time = datetime.now()
    connections.connect("default", host='localhost', port='19530')

    if args.clear:
        if not args.clear_collection:
            logging.error("When using --clear, you must specify a collection name with --clear-collection")
            return
        clear_collection(args.clear_collection)
        logging.info(f"Collection '{args.clear_collection}' has been cleared.")
        connections.disconnect("default")
        return

    if not args.path:
        logging.error("You must specify a path when not using --clear")
        return

    embedding_model = args.model if not args.local else LOCAL_EMBEDDING_MODEL
    embedding_dim = EMBEDDING_DIMENSIONS[embedding_model] if not args.local else LOCAL_EMBEDDING_DIM
    
    collection_name = args.collection if args.collection else f"documents_{embedding_model.replace('-', '_')}"

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=SNIPPET_LENGTH),
        FieldSchema(name="filehash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="creation_date", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, description="Document Collection")

    collection = ensure_collection_exists(collection_name, schema)
    
    logging.debug(f"Using collection: {collection_name}")
    logging.debug(f"Schema set with embedding dimension: {embedding_dim}")

    if not collection.has_index():
        index_params = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": {"nlist": NLIST}}
        collection.create_index(field_name="vector", index_params=index_params)
    
    collection.load()

    if args.local:
        local_tokenizer, local_model = load_local_model(LOCAL_EMBEDDING_MODEL)
    else:
        local_tokenizer, local_model = None, None

    path = os.path.abspath(args.path)

    if os.path.isdir(path):
        total_files = count_files(path)
        progress_bar = tqdm(total=total_files, desc="Processing files")
        for filepath, filehash, creation_date in load_files(path, args.recursive):
            try:
                expr = f"filehash == '{filehash}'"
                results = collection.query(expr, output_fields=["filehash"])
                if results:
                    logging.debug(f"Skipping {filepath} as it is unchanged.")
                    continue

                logging.debug(f"Processing {filepath} as it is new or modified.")
                if args.line_by_line:
                    process_and_insert_lines(filepath, collection, embedding_model, embedding_dim, args.local, local_tokenizer, local_model)
                else:
                    # Delete old entries before processing
                    delete_old_entries(collection, filepath)
                    chunks = process_file(filepath)
                    if chunks:
                        text_snippets = [extract_snippet(chunk) for chunk in chunks]
                        vectors = embed_text_to_vector(chunks, embedding_model, args.local, local_tokenizer, local_model)
                        validated_vectors = validate_embeddings(vectors, embedding_dim)
                        if validated_vectors:
                            logging.debug(f"Inserting {len(validated_vectors)} vectors into the collection.")
                            collection.insert([
                                validated_vectors,
                                [filepath] * len(validated_vectors),
                                text_snippets,
                                [filehash] * len(validated_vectors),
                                [embedding_model] * len(validated_vectors),
                                [creation_date] * len(validated_vectors)
                            ])
                            logging.info(f"Successfully inserted vectors and snippets for {filepath}")
            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}")
                logging.error(traceback.format_exc())
            finally:
                progress_bar.update(1)
        progress_bar.close()
    elif os.path.isfile(path):
        try:
            filehash = file_hash(path)
            creation_date = get_creation_date(path)
            expr = f"filehash == '{filehash}'"
            results = collection.query(expr, output_fields=["filehash"])
            if results:
                logging.debug(f"Skipping {path} as it is unchanged.")
            else:
                logging.debug(f"Processing {path} as it is new or modified.")
                if args.line_by_line:
                    process_and_insert_lines(path, collection, embedding_model, embedding_dim, args.local, local_tokenizer, local_model)
                else:
                    # Delete old entries before processing
                    delete_old_entries(collection, path)
                    chunks = process_file(path)
                    if chunks:
                        text_snippets = [extract_snippet(chunk) for chunk in chunks]
                        vectors = embed_text_to_vector(chunks, embedding_model, args.local, local_tokenizer, local_model)
                        validated_vectors = validate_embeddings(vectors, embedding_dim)
                        if validated_vectors:
                            logging.debug(f"Inserting {len(validated_vectors)} vectors into the collection.")
                            collection.insert([
                                validated_vectors,
                                [path] * len(validated_vectors),
                                text_snippets,
                                [filehash] * len(validated_vectors),
                                [embedding_model] * len(validated_vectors),
                                [creation_date] * len(validated_vectors)
                            ])
                            logging.info(f"Successfully inserted vectors and snippets for {path}")
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            logging.error(traceback.format_exc())

    connections.disconnect("default")
    end_time = datetime.now()
    logging.info(f"Operation completed in {end_time - start_time}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load documents into the vector store using Milvus.")
    parser.add_argument("path", type=str, nargs='?', help="File or directory to load documents from")
    parser.add_argument("-r", "--recursive", action="store_true", help="Load documents recursively if a directory is specified")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear the specified collection")
    parser.add_argument("--clear-collection", type=str, help="Specify the collection name to clear when using --clear")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_EMBEDDING_MODEL, choices=EMBEDDING_DIMENSIONS.keys(), help="Embedding model to use")
    parser.add_argument("--line-by-line", action="store_true", help="Process text files line by line")
    parser.add_argument("--collection", type=str, help="Specify the collection name to use")
    parser.add_argument("--local", action="store_true", help="Use local embedding model")
    args = parser.parse_args()
    main(args)