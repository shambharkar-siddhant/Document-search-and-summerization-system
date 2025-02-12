import os
from flask import Flask, request, jsonify  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
from processor_bkp import search_index  # type: ignore

app = Flask(__name__)

# Directory to store uploaded PDFs
UPLOAD_FOLDER = 'event_data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

@app.route('/upload', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({"error": "File not provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200

    return jsonify({"error": "Unsupported file type"}), 400


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data['query']
    result, status_code = search_index(query)

    if status_code == 200:
        return jsonify(result), status_code
    else:
        return jsonify(result), status_code


if __name__ == '__main__':
    app.run(debug=True, port=5000)
