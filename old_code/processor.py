import os
from flask import Flask, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
UPLOAD_FOLDER = 'event_data'
INDEX_FILE = 'document_index.faiss'
MODEL = 'all-MiniLM-L6-v2'

# Load or initialize the FAISS index and the sentence transformer model
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    # Initialize the index if not present
    embedding_dim = 384  # Dimension for MiniLM embeddings
    index = faiss.IndexFlatL2(embedding_dim)

model = SentenceTransformer(MODEL)

# Function to embed text into vector space
def embed_text(text):
    return model.encode([text])[0]

# Function to index a document
def index_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    vec = embed_text(text)
    index.add(np.array([vec]))

# Endpoint to handle search
@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is missing"}), 400

    query_vec = np.array([embed_text(query)])
    # Search the index for the top 1 most similar vector
    D, I = index.search(query_vec, 1)
    if I.size > 0:
        return jsonify({"result": "Document found", "distance": float(D[0][0]), "index": int(I[0][0])}), 200
    else:
        return jsonify({"result": "No relevant documents found"}), 404

if __name__ == '__main__':
    # Index all documents in the directory on startup
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        index_document(file_path)
    # Write the index to disk for persistence
    faiss.write_index(index, INDEX_FILE)
    app.run(debug=True, port=5001)
