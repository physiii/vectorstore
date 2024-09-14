# main.py
from flask import Flask, request, jsonify
from load import load_to_vectorstore, load_text_to_vectorstore, clear_vectorstore_collection
from search import search_vectorstore
import argparse
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/vectorstore', methods=['GET', 'POST'])
def handle_vectorstore_request():
    if request.method == 'GET':
        return jsonify({"message": "Vectorstore is running. Use POST for operations."}), 200

    data = request.json
    operation_type = data.get('type')

    logging.info(f"Received request: {data}")

    if operation_type == 'load':
        if 'text' in data:
            text = data.get('text')
            collection_name = data.get('collection')
            embedding_model = data.get('model')
            line_by_line = data.get('line_by_line', False)
            chunk_size = data.get('chunk_size', 1000)
            overlap = data.get('overlap', 0)

            result = load_text_to_vectorstore(
                text,
                collection_name=collection_name,
                embedding_model=embedding_model,
                line_by_line=line_by_line,
                chunk_size=chunk_size,
                overlap=overlap
            )
            return jsonify({"message": "Text loaded successfully", "details": result})
        elif 'path' in data:
            args = argparse.Namespace(**data)
            load_to_vectorstore(args)
            return jsonify({"message": "Documents loaded successfully"})
        else:
            return jsonify({"error": "No text or path provided for loading"}), 400

    elif operation_type == 'search':
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided for searching"}), 400

        # Extract additional search parameters
        limit = data.get('limit', 10)
        path_filter = data.get('path', "")
        unique = data.get('unique', False)
        collection_name = data.get('collection')
        ip_address = data.get('ip_address', "localhost")

        try:
            results = search_vectorstore(
                query,
                limit=limit,
                path_filter=path_filter,
                unique=unique,
                collection_name=collection_name,
                ip_address=ip_address
            )
            return jsonify({"results": results})
        except Exception as e:
            logging.error(f"Search operation failed: {str(e)}")
            return jsonify({"error": "An unexpected error occurred"}), 500

    elif operation_type == 'clear':
        collection_name = data.get('collection')
        if not collection_name:
            return jsonify({"error": "No collection name provided for clearing"}), 400

        try:
            clear_vectorstore_collection(collection_name)
            return jsonify({"message": f"Collection '{collection_name}' cleared successfully"})
        except Exception as e:
            logging.error(f"Clear operation failed: {str(e)}")
            return jsonify({"error": "An unexpected error occurred during clear operation"}), 500

    else:
        return jsonify({"error": "Invalid operation type"}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f"Unhandled exception: {str(e)}")
    # Return JSON instead of HTML for HTTP errors
    return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
