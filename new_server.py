import os
import numpy as np
import fitz  # PyMuPDF
import faiss
from openai import OpenAI

client = OpenAI(api_key="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")
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------------------------------------
# DOCUMENT PARSING MODULE
# -----------------------------------------------------------------------------

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path, chunk_size=500, overlap=50):
        """
        Parse a PDF file and return a list of text chunks with metadata.
        Splits each page into chunks of (approximately) `chunk_size` words,
        with an overlap to maintain context.
        """
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
                    "page_num": page_num + 1,  # pages numbered from 1
                    "text": chunk_text
                })
                start += (chunk_size - overlap)
        return chunks

# -----------------------------------------------------------------------------
# EMBEDDING & INDEXING MODULE
# -----------------------------------------------------------------------------

class EmbeddingIndexer:
    def __init__(self):
        # Use LangChain’s OpenAIEmbeddings (defaults to text-embedding-ada-002)
        self.embedding_model = OpenAIEmbeddings()
        # text-embedding-ada-002 returns vectors of dimension 1536
        self.dim = 1536  
        # Create a FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.dim)
        # To track metadata for each added vector
        self.metadata = []

    def add_documents(self, docs):
        """
        Given a list of document chunks (each with a 'text' field),
        compute embeddings and add them to the FAISS index.
        """
        texts = [doc["text"] for doc in docs]
        # Compute embeddings for all texts
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype("float32")
        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        # Append the associated metadata
        self.metadata.extend(docs)
        print(f"Added {len(docs)} chunks to the index.")

    def get_query_embedding(self, query):
        """
        Get embedding for a query string.
        """
        embedding = self.embedding_model.embed_query(query)
        return np.array(embedding).astype("float32")

# -----------------------------------------------------------------------------
# SEMANTIC SEARCH MODULE
# -----------------------------------------------------------------------------

class SearchEngine:
    def __init__(self, indexer: EmbeddingIndexer):
        self.indexer = indexer

    def search(self, query, top_k=3):
        """
        Search the FAISS index for the top_k similar chunks to the query.
        Returns a list of results with metadata and a computed "score".
        """
        query_embedding = self.indexer.get_query_embedding(query).reshape(1, -1)
        distances, indices = self.indexer.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            # Make sure we got a valid index
            if idx < len(self.indexer.metadata):
                meta = self.indexer.metadata[idx]
                # Convert distance to a simple score (the lower the distance, the higher the score)
                score = 1 / (1 + distances[0][i])
                results.append({
                    "doc_id": meta.get("doc_id"),
                    "page_num": meta.get("page_num"),
                    "text": meta.get("text"),
                    "score": score
                })
        return results

# -----------------------------------------------------------------------------
# SUMMARIZATION & REPHRASING MODULE
# -----------------------------------------------------------------------------
class Summarizer:
    def summarize(self, text, summary_length="short"):
        """
        Generate a concise summary of the text.
        summary_length can be 'short', 'medium', or 'detailed'.
        """
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
        """
        Rephrase the text so that it more directly and clearly answers the query.
        This is used if the search result is deemed “unclear”.
        """
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

# -----------------------------------------------------------------------------
# FLASK API & APPLICATION SETUP
# -----------------------------------------------------------------------------

app = Flask(__name__)

# Global objects for indexing, search, and summarization.
embedding_indexer = EmbeddingIndexer()
search_engine = SearchEngine(embedding_indexer)
summarizer = Summarizer()

# Simple in-memory cache for query responses.
query_cache = {}

@app.route('/ingest', methods=['POST'])
def ingest():
    """
    Endpoint to ingest a PDF document. The client should POST a file (multipart/form-data)
    with the key 'file'. The PDF is parsed, split into text chunks, and added to the FAISS index.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # Save the file temporarily
    temp_path = os.path.join("/tmp", file.filename)
    file.save(temp_path)

    try:
        chunks = DocumentParser.parse_pdf(temp_path)
        embedding_indexer.add_documents(chunks)
    except Exception as e:
        return jsonify({"error": f"Error during ingestion: {e}"}), 500

    return jsonify({
        "status": "success",
        "message": f"Document '{file.filename}' ingested and indexed.",
        "num_chunks": len(chunks)
    })

@app.route('/search', methods=['POST'])
def search():
    """
    Endpoint to perform semantic search. Expects a JSON body with:
      - query: the search string
      - summary_preference (optional): one of 'short', 'medium', 'detailed'
    The endpoint first performs a vector search and then (if needed) rephrases or summarizes
    the retrieved passages.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    query = data["query"]
    summary_pref = data.get("summary_preference", None)

    # Check if we have a cached result for this query
    if query in query_cache:
        return jsonify(query_cache[query])

    # Perform semantic search
    results = search_engine.search(query, top_k=3)
    final_results = []
    for result in results:
        score = result.get("score", 0)
        text = result.get("text", "")

        # Example heuristic: if score is low, try to rephrase the text.
        if score < 0.8:
            text = summarizer.rephrase(text, query)

        # If the passage is long and the user requested a summary, then summarize.
        if summary_pref and len(text.split()) > 200:
            text = summarizer.summarize(text, summary_pref)

        final_results.append({
            "doc_id": result.get("doc_id"),
            "page_num": result.get("page_num"),
            "text": text,
            "score": score
        })

    response = {"query": query, "results": final_results}
    # Cache the response
    query_cache[query] = response

    return jsonify(response)

@app.route('/summarize', methods=['POST'])
def summarize_route():
    """
    Endpoint to directly summarize a given text.
    Expects a JSON body with:
      - text: the text to be summarized
      - summary_preference (optional): one of 'short', 'medium', 'detailed'
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    text = data["text"]
    summary_pref = data.get("summary_preference", "short")
    summary_text = summarizer.summarize(text, summary_pref)
    return jsonify({"summary": summary_text})

if __name__ == '__main__':
    app.run(debug=True)
