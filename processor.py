#!/usr/bin/env python3
import os
import time
import json
import sqlite3
import fitz  # PyMuPDF
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings

DATABASE_FILE = "database.db"
PERSISTENT_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"
api_key = os.getenv("OPENAI_API_KEY")


def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    filepath TEXT,
                    status TEXT,
                    created_at TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER,
                    doc_id TEXT,
                    page_num INTEGER,
                    text TEXT,
                    embedding TEXT
                 )''')
    conn.commit()
    conn.close()

init_db()

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path, chunk_size=500, overlap=50):
        doc = fitz.open(file_path)
        chunks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "doc_id": os.path.basename(file_path),
                    "page_num": page_num + 1,  # pages start at 1
                    "text": chunk_text
                })
                start += (chunk_size - overlap)
        return chunks

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

def update_job_status(job_id, status):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
    conn.commit()
    conn.close()

def update_persistent_index():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT id, doc_id, page_num, text, embedding FROM documents")
    rows = c.fetchall()
    conn.close()

    embeddings = []
    metadata = []
    for row in rows:
        try:
            embed = json.loads(row[4])
            embeddings.append(embed)
            metadata.append({
                "doc_id": row[1],
                "page_num": row[2],
                "text": row[3]
            })
        except Exception as e:
            print(f"Error loading embedding for row {row[0]}: {e}")

    if len(embeddings) == 0:
        print("No embeddings found; skipping persistent index update.")
        return

    embeddings_np = np.array(embeddings).astype("float32")
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)

    faiss.write_index(index, PERSISTENT_INDEX_FILE)

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"Persistent FAISS index updated with {len(embeddings)} embeddings.")

def process_job(job_id, filepath):
    print(f"Processing job {job_id} for file {filepath}")
    try:
        chunks = DocumentParser.parse_pdf(filepath)
    except Exception as e:
        print(f"Error parsing PDF {filepath}: {e}")
        update_job_status(job_id, "failed")
        return

    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    for chunk in chunks:
        text = chunk["text"]
        try:
            # Compute the embedding for this chunk (embedding_model.embed_documents returns a list)
            embedding = embedding_model.embed_documents([text])[0]
            embedding_json = json.dumps(embedding)
        except Exception as e:
            print(f"Error computing embedding for a chunk: {e}")
            continue

        c.execute("INSERT INTO documents (job_id, doc_id, page_num, text, embedding) VALUES (?, ?, ?, ?, ?)",
                  (job_id, chunk["doc_id"], chunk["page_num"], text, embedding_json))
    conn.commit()
    conn.close()
    update_job_status(job_id, "completed")
    print(f"Completed processing job {job_id}")
    
    # After processing, update the persistent FAISS index so that the API server can use it.
    update_persistent_index()

def poll_jobs():
    while True:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        # Retrieve jobs with status 'pending'
        c.execute("SELECT id, filepath FROM jobs WHERE status='pending'")
        jobs = c.fetchall()
        conn.close()

        if jobs:
            for job in jobs:
                job_id, filepath = job
                update_job_status(job_id, "processing")
                process_job(job_id, filepath)
        else:
           continue
        
        time.sleep(1)

if __name__ == '__main__':
    print("Starting processor...")
    poll_jobs()
