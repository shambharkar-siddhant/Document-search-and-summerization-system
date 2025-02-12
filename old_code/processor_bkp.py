import os
import fitz  # PyMuPDF for PDF parsing
from openai import OpenAI

client = OpenAI(api_key="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")
import faiss
import numpy as np
import time

# --- Setup and Utility Functions ---
document_store = {}
index = None

def initialize_index():
    global index
    index_path = os.path.join("event_data", "faiss_index.bin")
    index = load_index(index_path)

def load_index(path):
    """ Load or initialize the FAISS index. """
    if os.path.exists(path):
        print("Loading index from file...")
        return faiss.read_index(path)
    else:
        print("Creating new index...")
        idx = faiss.IndexFlatL2(1536)
        faiss.write_index(idx, path)
        return idx

def save_index(index, path):
    """ Save the FAISS index to disk. """
    faiss.write_index(index, path)

# --- Document Processing Functions ---

def create_embeddings(text):
    """ Create embeddings using OpenAI. """
    try:
        response = client.embeddings.create(model="text-embedding-ada-002",
        input=[text])
        return np.array(response.data[0].embedding).astype('float32')
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return None

def index_document(text, index, index_path):
    """ Index document text and store it in a dictionary for retrieval. """
    global document_store
    try:
        embedding = create_embeddings(text)
        if embedding is not None:
            doc_id = len(document_store)  # Unique ID for the document
            document_store[doc_id] = text
            index.add(np.array([embedding]))
            save_index(index, index_path)
            print(f"Document indexed successfully with ID {doc_id}.")
        else:
            print("Failed to create embedding, document not indexed.")
    except Exception as e:
        print("Failed to index document:", str(e))


def process_pdf_file(file_path, index, index_path):
    """ Process a single PDF file, extract text, create embeddings, and index it. """
    try:
        doc = fitz.open(file_path)
        text = ''.join(page.get_text() for page in doc)
        doc.close()
        print(f"Extracted text from {file_path}: {text[:100]}")  # Log extracted text snippet
        if text.strip():
            index_document(text, index, index_path)
            processed_path = file_path + ".processed"
            os.rename(file_path, processed_path)
            print(f"Processed and indexed: {processed_path}")
        else:
            print(f"No text found in {file_path}")
    except Exception as e:
        error_path = file_path + ".error"
        os.rename(file_path, error_path)
        print(f"Failed to process PDF {file_path}: {str(e)}")

# --- Search and Content Handling Functions ---

def fetch_document_content(document_id):
    """ Fetch document content from the document store. """
    try:
        return document_store[document_id]
    except KeyError:
        print(f"Document ID {document_id} not found in document store.")
        return "Document content not found."

def search_index(query):
    """ Perform a semantic search using the index. """
    directory_to_watch = "event_data"
    index_path = os.path.join(directory_to_watch, "faiss_index.bin")
    index = load_index(index_path)

    query_embedding = create_embeddings(query)
    D, I = index.search(np.array([query_embedding]), k=1)
    if I[0][0] == -1:
        return {"error": "No matching documents found"}, 404
    document_id = int(I[0][0])
    document_content = fetch_document_content(document_id)
    return {"document_id": document_id, "content": document_content}, 200

def rephrase_with_llm(text):
    """ Use an LLM to rephrase unclear text. """
    response = client.completions.create(model="gpt-4",  # Assuming GPT-4
    prompt=f"Rephrase this for clarity: {text}",
    max_tokens=100)
    return response.choices[0].text.strip()

def summarize_with_llm(text):
    """ Use an LLM to generate a concise summary. """
    response = client.completions.create(model="gpt-4",  # Assuming GPT-4
    prompt=f"Summarize this text: {text}",
    max_tokens=150)
    return response.choices[0].text.strip()

# --- Monitoring and Directory Watching ---

def monitor_directory(directory):
    """ Monitor directory and process new PDF files. """
    index_path = os.path.join(directory, "faiss_index.bin")
    initialize_index()  # Initialize index on startup
    while True:
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.endswith('.pdf') and not f.endswith('.processed')]
        for file in files:
            process_pdf_file(file, index_path)
        time.sleep(10)

if __name__ == "__main__":
    directory_to_watch = "event_data"
    monitor_directory(directory_to_watch)
