# Document Search and Summarization System

The Document Search and Summarization System is a backend solution that processes PDF files, indexes their contents using vector embeddings, and provides fast, semantic search with automatic summarization of long results. The system separates heavy processing from user interactions to ensure that search queries receive prompt responses.

---

## Overview

- **API Server:**  
  Provides REST API endpoints to accept PDF file uploads and perform semantic search. Uploaded files are stored and recorded as processing jobs, while search queries are answered using a persistent vector index.

- **Background Processor:**  
  Continuously monitors a database for new document processing jobs. It extracts text from PDFs, splits the text into overlapping chunks, computes vector embeddings, and builds a persistent FAISS index along with associated metadata.

- **Persistent Indexing:**  
  A FAISS index (with metadata) is maintained on disk. The API server loads these files into memory to serve search queries quickly without rebuilding the index on every request.

- **Process Management:**  
  A launch script sets the required environment variable, starts both the API server and the background processor, and ensures that all processes are terminated when the script is closed.

---

## Features

- **Efficient Document Ingestion:**  
  PDF files are parsed using PyMuPDF and divided into overlapping text chunks. Each chunk is converted into a vector embedding and stored for rapid retrieval.

- **Semantic Search and Summarization:**  
  User queries are transformed into vector embeddings and matched against the persistent FAISS index. If the retrieved text exceeds a specified word limit, it is automatically summarized using a large language model (LLM).

- **Background Processing:**  
  Intensive processing tasks (text extraction, embedding computation, and index updating) are handled asynchronously by a separate processor.

- **Robust Process Management:**  
  A dedicated launch script manages environment setup and the start/stop of the API server and processor processes.

---

## Setup and Launch

1. **Prerequisites:**  
   - Python 3.7 or later  
   - Required packages: Flask, faiss-cpu, PyMuPDF, langchain-openai, OpenAI, and SQLite (standard with Python)  
   - An OpenAI API key set in the `OPENAI_API_KEY` environment variable

2. **Installation:**  
   Install the necessary dependencies using pip.

3. **Launching the System:**  
   Use the provided launch script to set the OpenAI API key, start the API server and background processor, and ensure that all processes are terminated when the script is closed.

---

## API Endpoints

### 1. `/ingest`
**Purpose:**  
Accepts a PDF file upload. The file is saved locally and a processing job is recorded in the SQLite database. The background processor handles further processing.

**Example Request (using curl):**
```bash
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:5000/ingest
Example Response:
{
  "status": "success",
  "message": "File uploaded. Processing will be done in background."
}
```
### 2. `/search`
**Purpose:**  
Receives a user query, computes its embedding, and searches the persistent FAISS index for the best matching document section. If the returned text exceeds a predefined word threshold, it is automatically summarized.
**Example Request (using curl):**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "example search query"}' http://localhost:5000/search
Example Response:
{
  "query": "example search query",
  "result": {
    "doc_id": "document.pdf",
    "page_num": 2,
    "score": 0.95,
    "text": "The summarized content of the matching document..."
}
```

## Architecture Summary

- **Database:**
A SQLite database records file ingestion jobs and stores document chunks with computed embeddings.

- **Indexing:**
FAISS is used for fast semantic search. A persistent index and corresponding metadata are maintained on disk and loaded by the API server for efficient query handling.

- **Background Processing:**
A separate processor continuously polls the database, processes new PDF files, computes embeddings, and updates the persistent index.

- **Process Control:**
A launch script manages the setting of environment variables and the start/stop of the API server and processor processes.

This system is engineered for efficiency and scalability, with a clear separation between user-facing operations and intensive processing tasks. The design ensures that users receive prompt responses to their queries while heavy processing is handled asynchronously.

---