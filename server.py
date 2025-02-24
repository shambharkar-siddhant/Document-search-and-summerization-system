#!/usr/bin/env python3
import os
import json
import sqlite3
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, jsonify
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
import time

# Configuration
DATABASE_FILE = "database.db"
UPLOAD_FOLDER = "uploads"
PERSISTENT_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"
LENGTH_THRESHOLD = 300  # If text exceeds 300 words, we summarize it
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to cache the index and its metadata
global_index = None
global_metadata = None
global_index_mtime = None

def load_persistent_index_cached():
    """Load the persistent FAISS index and metadata from disk, but only if the file has changed."""
    global global_index, global_metadata, global_index_mtime
    try:
        mtime = os.path.getmtime(PERSISTENT_INDEX_FILE)
    except OSError:
        return None, None

    # If we already have a cached index and the file hasn't changed, return it.
    if global_index is not None and global_index_mtime == mtime:
        return global_index, global_metadata

    # Otherwise, load the index and metadata from disk.
    try:
        index = faiss.read_index(PERSISTENT_INDEX_FILE)
    except Exception as e:
        print(f"Error reading FAISS index: {e}")
        return None, None

    try:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None, None

    global_index = index
    global_metadata = metadata
    global_index_mtime = mtime
    return global_index, global_metadata


summary_cache = {}

def get_summary(text):
    if text in summary_cache:
        return summary_cache[text]
    
    summary = summarizer.summarize(text, "short")
    summary_cache[text] = summary
    return summary


# Summarizer class for generating a concise version of long text using OpenAI ChatCompletion
class Summarizer:
    def __init__(self):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")

    def summarize(self, text, summary_length="short"):
        prompt = (f"Summarize the following technical text in a {summary_length} summary, "
                  "preserving key details and technical accuracy:\n\n" + text)
        try:
            response = client.chat.completions.create(model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a concise and accurate technical summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3)
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            summary = f"Error during summarization: {e}"
        return summary

    def rephrase(self, text, query):
        prompt = (f"The following text might be unclear. Rephrase it so that it directly and clearly answers the query:\n"
                  f"Query: {query}\n\nText: {text}")
        try:
            response = client.chat.completions.create(model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who clarifies technical content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3)
            rephrased = response.choices[0].message.content.strip()
        except Exception as e:
            rephrased = f"Error during rephrasing: {e}"
        return rephrased

summarizer = Summarizer()

# Initialize the embedding model; ensure OPENAI_API_KEY is set.
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

# Create the Flask application
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO jobs (filename, filepath, status, created_at) VALUES (?, ?, ?, ?)",
              (file.filename, filepath, "pending", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

    return jsonify({"status": "File uploaded successfully."})

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    query = data["query"]

    # Load the persistent index and metadata from the cached version
    index, metadata = load_persistent_index_cached()
    if index is None:
        return jsonify({"error": "No persistent index available. Please try again later."}), 404

    # Compute the query embedding
    query_embedding = embedding_model.embed_query(query)
    query_embedding_np = np.array(query_embedding).astype("float32").reshape(1, -1)

    # Search for the best match (top_k = 1)
    top_k = 1
    distances, indices = index.search(query_embedding_np, top_k)

    best_result = None
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            meta = metadata[idx]
            score = 1 / (1 + distances[0][i])
            text = meta.get("text", "")

            # If the returned text is too long, summarize it
            if len(text.split()) > LENGTH_THRESHOLD:
                text = get_summary(text)
            best_result = {
                "doc_id": meta.get("doc_id"),
                "page_num": meta.get("page_num"),
                "score": score,
                "text": text
            }

    if best_result:
        return jsonify({"query": query, "result": best_result})
    else:
        return jsonify({"query": query, "result": None})


if __name__ == '__main__':
    app.run(debug=True)
