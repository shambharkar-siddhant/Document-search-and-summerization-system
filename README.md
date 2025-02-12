# Document Search and Summarization System

The Document Search and Summarization System processes PDF files, indexes their contents using vector embeddings, and provides fast, smart search with automatic summarization. 

---

## Overview

- **API Server:**  
  The API server provides REST endpoints for uploading PDF files and searching for content. Uploaded files are stored and marked for processing, while search queries use a pre-built index to quickly return the best match.

- **Background Processor:**  
  A background process watches for new file uploads. It extracts text from PDFs, divides the text into overlapping chunks, computes vector embeddings, and builds an efficient search index (using FAISS) along with metadata.

- **Persistent Indexing:**  
  The system maintains a persistent FAISS index and a metadata file on disk. The API server loads these files into memory to deliver fast search responses without rebuilding the index every time.

- **Process Management:**  
  A launch script sets up the required environment variables, starts both the API server and the background processor, and ensures that all processes are cleanly terminated when the script is closed.

---

## Features

- **Efficient Document Ingestion:**  
  PDF files are read using PyMuPDF and split into manageable, overlapping text chunks. Each chunk is converted into a vector embedding for rapid lookup.

- **Smart Search and Summarization:**  
  User queries are transformed into vector embeddings and matched against the stored index. If the returned text is too long, it is automatically summarized by a language model.

- **Asynchronous Background Processing:**  
  Intensive tasks like text extraction, embedding computation, and index updating are performed in the background so that the API responds quickly to users.

- **Robust Process Control:**  
  A dedicated launch script manages the entire system, ensuring smooth startup and shutdown of all processes.

---

## Setup and Launch

1. **Requirements:**
   - Flask, faiss-cpu, PyMuPDF, langchain-openai, OpenAI, and SQLite (built into Python)
   - An OpenAI API key stored in the `OPENAI_API_KEY` environment variable

2. **Launching the System:**  
   launch provided bash launch script to start the server and processor both programs. set OPENAI_API_KEY in the launch script to it will be accessible to programs.

---

## API Endpoints

### 1. Upload PDF File

- **Endpoint:** `/upload`
- **Purpose:**  
  Upload a PDF file to be processed.
- **Request Details:**  
  - **Method:** POST  
  - **Body:**  `file_to_upload.pdf` (submitted as form-data)
- **Example Response:**  
  ```json
  {
    "status": "success",
    "message": "File uploaded successfully."
  }
  ```

### 2. Search Documents

- **Endpoint:** `/search`
- **Purpose:**  
  Search for content within the processed documents.
- **Request Details:**  
  - **Method:** POST  
  - **Content-Type:** application/json
  - **Body:** 
  ```json
  {
  "query": "example search query"
  }
  ```

- **Example Response:**  
  ```json
  {
    "query": "example search query",
    "result": {
      "doc_id": "document.pdf",
      "page_num": 2,
      "score": 0.95,
      "text": "The summarized content of the matching document..."
    }
  }
  ```

---

## Architecture Summary

### Database:
A SQLite database records file upload jobs and stores document chunks along with their computed embeddings.

### Indexing:
FAISS is used for fast semantic search. A persistent index and its metadata are stored on disk and loaded into memory by the API server.

### Background Processing:
A separate processor continuously polls the database for new files, extracts text, computes embeddings, and updates the index.

### Process Control:
A launch script manages the environment setup and starts/stops both the API server and the background processor.

---