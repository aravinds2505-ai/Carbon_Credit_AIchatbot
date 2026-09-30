import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from query import query

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "doc"
DATA_DIR = BASE_DIR / "data"
VERCEL_DEPLOYMENT = bool(os.environ.get("VERCEL"))

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


def get_pdf_names():
    if not PDF_DIR.is_dir():
        return []
    return sorted(
        path.name for path in PDF_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/query")
@app.post("/query")
def handle_query():
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 2000:
        return jsonify({"error": "Questions must be 2,000 characters or fewer."}), 400

    joke = []
    try:
        answer = query(question.strip(), joke_callback=joke.append)
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Query failed")
        return jsonify({"error": "The assistant could not complete that request."}), 500

    return jsonify({"joke": joke[0] if joke else "", "answer": answer})


@app.get("/api/documents")
def documents():
    return jsonify({"documents": get_pdf_names()})


@app.post("/api/upload")
def upload_documents():
    if VERCEL_DEPLOYMENT:
        return jsonify({"error": "PDF uploads need persistent storage, which is not configured for this Vercel deployment."}), 503

    files = request.files.getlist("files")
    if not files or any(not file.filename for file in files):
        return jsonify({"error": "Choose one or more PDF files to upload."}), 400

    safe_files = []
    for file in files:
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files can be uploaded."}), 400
        safe_files.append((file, filename))

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for file, filename in safe_files:
        destination = PDF_DIR / filename
        if not destination.exists():
            file.save(destination)

    return jsonify({"documents": get_pdf_names()})


@app.post("/api/ingest")
def ingest_documents():
    if VERCEL_DEPLOYMENT:
        return jsonify({"error": "PDF ingestion needs persistent storage, which is not configured for this Vercel deployment."}), 503

    if not get_pdf_names():
        return jsonify({"error": "Upload at least one PDF before ingesting."}), 400

    try:
        from ingest import ingest as ingest_pdfs

        ingest_pdfs(str(PDF_DIR), out_dir=str(DATA_DIR))
    except Exception:
        app.logger.exception("PDF ingestion failed")
        return jsonify({"error": "The PDFs could not be ingested. Check the server log for details."}), 500

    return jsonify({"message": "PDF ingestion completed.", "documents": get_pdf_names()})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "ingestion_available": not VERCEL_DEPLOYMENT})


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "The upload exceeds the 100 MB request limit."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
