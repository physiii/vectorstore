import os
import hashlib
import logging
import webvtt
from nltk.tokenize import sent_tokenize
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# Constants Configuration
CHUNK_SIZE = 6
OVERLAP = 2
BATCH_SIZE = 10
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 1024
SNIPPET_LENGTH = 1500  # Length of text snippet to store with each document

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-large": 3072
}

# Local embedding constants
LOCAL_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
LOCAL_EMBEDDING_DIM = 1536  # Dimension for gte-Qwen2-1.5B-instruct

# OpenAI API Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_vtt_to_text(filepath):
    captions = webvtt.read(filepath)
    return " ".join(caption.text for caption in captions)

def chunk_text_with_overlap(text, initial_chunk_size=CHUNK_SIZE, max_length=SNIPPET_LENGTH):
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    
    while i < len(sentences):
        chunk_size = initial_chunk_size
        valid_chunk_formed = False
        while chunk_size > 0:
            chunk = ' '.join(sentences[i:i + chunk_size])
            if len(chunk) > max_length:
                logging.debug(f"Oversized chunk (size {len(chunk)}): {chunk[:500]}...")
                chunk_size -= 1
                if chunk_size == 0:
                    logging.error("Chunk size reduced to 0, unable to create valid chunks.")
                    i += 1  # Move to the next sentence to avoid infinite loop
                    break
            else:
                chunks.append(chunk)
                logging.debug(f"Final chunk (size {len(chunk)}): {chunk[:500]}...")
                valid_chunk_formed = True
                break  # Break once a valid chunk is formed
        
        if valid_chunk_formed:
            i += chunk_size  # Increment i by the size of the chunk that was added
        elif chunk_size == 0:
            i += 1  # Force increment if no valid chunk could be formed (e.g., a single sentence larger than max_length)

    return chunks

def extract_snippet(text):
    if (len(text) > SNIPPET_LENGTH):
        logging.warning(f"Snippet length ({len(text)}) exceeds max length ({SNIPPET_LENGTH}). Truncating.")
        return text[:SNIPPET_LENGTH]
    return text

def process_file(filepath):
    if filepath.endswith('.vtt'):
        text = load_vtt_to_text(filepath)
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        return []
    return chunk_text_with_overlap(text)

def file_hash(filepath):
    hash_obj = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    computed_hash = hash_obj.hexdigest()
    logging.debug(f"Hash for {filepath}: {computed_hash}")
    return computed_hash

def get_creation_date(filepath):
    stat_info = os.stat(filepath)
    return int(stat_info.st_ctime)

def load_local_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embed_text_to_vector(text_chunks, model, use_local=False, local_tokenizer=None, local_model=None):
    if not use_local:
        vectors = []
        for i in range(0, len(text_chunks), BATCH_SIZE):
            batch = text_chunks[i:i + BATCH_SIZE]
            try:
                responses = client.embeddings.create(model=model, input=batch)
                batch_vectors = [item.embedding for item in responses.data]
                if batch_vectors:
                    logging.debug(f"Embedding dimension: {len(batch_vectors[0])}")
                vectors.extend(batch_vectors)
            except Exception as e:
                logging.error(f"Failed to generate embeddings for batch starting at index {i}: {e}")
                # Extend the vectors list with None to keep the batch sizes consistent
                vectors.extend([None] * len(batch))
    else:
        vectors = []
        for i in range(0, len(text_chunks), BATCH_SIZE):
            batch = text_chunks[i:i + BATCH_SIZE]
            try:
                batch_dict = local_tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = local_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                vectors.extend(embeddings.tolist())
            except Exception as e:
                logging.error(f"Failed to generate local embeddings for batch starting at index {i}: {e}")
                vectors.extend([None] * len(batch))
    return vectors

def validate_embeddings(vectors, expected_dim):
    valid_vectors = []
    for v in vectors:
        if isinstance(v, list) and len(v) == expected_dim:
            valid_vectors.append(v)
        else:
            logging.error(f"Invalid vector: {v}, Expected dimension: {expected_dim}")
    return valid_vectors

def count_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

def load_files(directory, recursive):
    directory = os.path.abspath(directory)  # Ensure directory is an absolute path
    for root, dirs, files in os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]:
        for file in files:
            filepath = os.path.join(root, file)
            filepath = os.path.abspath(filepath)  # Ensure filepath is an absolute path
            if os.path.isfile(filepath):
                # Skip .mkv files
                if not (filepath.endswith('.txt') or filepath.endswith('.vtt')):
                    logging.debug(f"Skipping {filepath} as it is not a supported file type.")
                    continue

                filehash = file_hash(filepath)
                yield filepath, filehash, get_creation_date(filepath)

def ensure_collection_exists(collection_name, schema):
    from pymilvus import Collection, utility

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        # Check if the schema matches
        existing_fields = {field.name: field for field in collection.schema.fields}
        required_fields = {field.name: field for field in schema.fields}

        # Check if the vector field exists and has the correct dimension
        if 'vector' in existing_fields and 'vector' in required_fields:
            existing_dim = existing_fields['vector'].params['dim']
            required_dim = required_fields['vector'].params['dim']
            if existing_dim != required_dim:
                logging.debug(f"Collection exists but has incorrect dimension. Dropping and recreating collection.")
                collection.drop()
                collection = Collection(name=collection_name, schema=schema)
            else:
                logging.debug(f"Collection exists with correct dimension.")
        else:
            logging.debug(f"Collection exists but schema doesn't match. Dropping and recreating collection.")
            collection.drop()
            collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name, schema=schema)
        logging.debug(f"Created new collection with fields: {[field.name for field in schema.fields]}")
    
    return collection

def clear_collection(collection_name):
    from pymilvus import Collection, utility
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.drop()
        logging.info(f"Vector store {collection_name} cleared.")
    else:
        logging.info(f"Collection {collection_name} does not exist. Nothing to clear.")

def delete_old_entries(collection, filepath):
    expr = f'path == "{filepath}"'
    collection.delete(expr)
    logging.info(f"Deleted old entries for {filepath}")

def process_and_insert_lines(filepath, collection, embedding_model, embedding_dim, use_local=False, local_tokenizer=None, local_model=None):
    filehash = file_hash(filepath)
    creation_date = get_creation_date(filepath)
    
    # Delete old entries before inserting new ones
    delete_old_entries(collection, filepath)
    
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i + BATCH_SIZE]
        vectors = embed_text_to_vector(batch, embedding_model, use_local, local_tokenizer, local_model)
        
        for j, (line, vector) in enumerate(zip(batch, vectors), start=i):
            if vector is not None and len(vector) == embedding_dim:
                try:
                    collection.insert([
                        [vector],
                        [filepath],
                        [line[:SNIPPET_LENGTH]],
                        [filehash],
                        [embedding_model],
                        [creation_date]
                    ])
                    logging.info(f"Inserted vector for line {j+1}: {line[:50]}...")
                except Exception as e:
                    logging.error(f"Failed to insert vector for line {j+1}: {line[:50]}... Error: {e}")
            else:
                logging.error(f"Invalid vector for line {j+1}: {line[:50]}...")