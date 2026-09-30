import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pickle

EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 500
OVERLAP = 50

# Default paths (Windows)
DEFAULT_PDF_DIR = r"C:\Users\Dell\Documents\Carbon_AI_Chatbot\doc"
DEFAULT_OUTDIR = r"C:\Users\Dell\Documents\Carbon_AI_Chatbot\data"


def load_pdf(path):
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks


def embed_texts(texts, model_name=EMBED_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings
    print(type(embeddings))  # Debugging line to check the type of embeddings


def ingest(pdf_path, out_dir="data", model_name=EMBED_MODEL, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    # use defaults when None passed
    model_name = model_name or EMBED_MODEL
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or OVERLAP

    if os.path.isdir(pdf_path):
        pdf_paths = sorted(
            os.path.join(pdf_path, filename)
            for filename in os.listdir(pdf_path)
            if filename.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(pdf_path, filename))
        )
    elif os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
        pdf_paths = [pdf_path]
    else:
        raise FileNotFoundError(f"PDF file or directory not found: {pdf_path}")

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {pdf_path}")

    os.makedirs(out_dir, exist_ok=True)
    for filename in ("chunks.pkl", "embeddings.npy", "faiss.index"):
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    model = SentenceTransformer(model_name)
    all_chunks = []
    embedding_batches = []

    for current_pdf in pdf_paths:
        text = load_pdf(current_pdf)
        chunks = chunk_text(text, size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        if chunks:
            embedding_batches.append(
                model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
            )

        if embedding_batches:
            embeddings = np.concatenate(embedding_batches, axis=0)
        else:
            embeddings = np.empty(
                (0, model.get_sentence_embedding_dimension()), dtype=np.float32
            )

        with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)
        np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
        print(f"Completed {os.path.basename(current_pdf)} ({len(chunks)} chunks)")

    print(f"Saved {len(all_chunks)} total chunks to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=DEFAULT_PDF_DIR, help='PDF file or directory to ingest')
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='Output directory for chunks/embeddings')
    parser.add_argument('--model', default=EMBED_MODEL, help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Chunk size (words)')
    parser.add_argument('--overlap', type=int, default=OVERLAP, help='Chunk overlap (words)')
    args = parser.parse_args()
    ingest(args.file, out_dir=args.outdir, model_name=args.model, chunk_size=args.chunk_size, overlap=args.overlap)
