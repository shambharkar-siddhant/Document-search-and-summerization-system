import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'event_data'

# Ensure the directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Endpoint for uploading documents
@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        # Save the file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "path": filepath}), 200

# Endpoint for searching documents
@app.route('/search', methods=['POST'])
def search_documents():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Send the query to the processor for searching
    response = requests.post("http://localhost:5001/search", json={"query": query})
    if response.status_code == 200:
        return jsonify(response.json()), 200
    else:
        return jsonify({"error": "Failed to process search"}), response.status_code

if __name__ == '__main__':
    app.run(debug=True, port=5000)
