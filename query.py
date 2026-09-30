import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if os.environ.get("VERCEL"):
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")

EMBED_MODEL = "all-mpnet-base-v2"
RAG_MODEL = "openai/gpt-oss-20b"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_QUESTION =f"what is the Indian carbon credit market?"
SYSTEM_PROMPT = """You are an AI assistant specializing in the Carbon Credit Market and Compliance domain, with a primary focus on the Indian Carbon Credit Trading Scheme (CCTS) and related Indian carbon-market regulations.

Your purpose is to help businesses understand carbon credits, carbon markets, participation requirements, compliance obligations, documentation, verification, authorities, procedures, and timelines.

You must provide answers based primarily on the retrieved source documents supplied to you through the RAG system.

Format every response in clear Markdown. Start with a concise "### Answer" section. Use short paragraphs, descriptive subheadings when useful, bullets for grouped facts, and numbered steps for procedures. Bold important terms sparingly. Avoid dense walls of text.

End every response with a "### Sources" section. For each source, list its document title, relevant authority, publication or effective date when available, page/section/clause reference, and source link. Only use details present in the retrieved sources. For unavailable details, write "Not available in retrieved sources"; never guess or invent citations."""


@lru_cache(maxsize=1)
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in .env or your deployment environment.")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


@lru_cache(maxsize=1)
def get_embedding_model(model_name):
    return SentenceTransformer(model_name)


def show_startup_joke(model_name):
    response = get_client().responses.create(
        model=model_name,
        input="Tell me one short, family-friendly joke about AI.",
        max_output_tokens=100,
        temperature=0.7,
    )
    return response.output_text


def load_embeddings(path="data/embeddings.npy"):
    emb = np.load(path)
    return emb


def ensure_store(data_dir="data"):
    emb_path = os.path.join(data_dir, "embeddings.npy")
    chunks_path = os.path.join(data_dir, "chunks.pkl")
    emb = load_embeddings(emb_path)
    dim = emb.shape[1]
    index_path = os.path.join(data_dir, "faiss.index")
    store = VectorStore(dim, index_path=index_path, chunks_path=chunks_path)
    if store.index.ntotal == 0:
        from pickle import load
        with open(store.chunks_path, "rb") as f:
            chunks = load(f)
        store.add(emb, chunks)
    return store


def query(
    question,
    k=5,
    embed_model=EMBED_MODEL,
    rag_model=RAG_MODEL,
    data_dir=DEFAULT_DATA_DIR,
    joke_callback=None,
):
    embed_model = embed_model or EMBED_MODEL
    rag_model = rag_model or RAG_MODEL
    joke = show_startup_joke(rag_model)
    if joke_callback:
        joke_callback(joke)
    else:
        print(f"AI joke: {joke}")

    model = get_embedding_model(embed_model)
    q_emb = model.encode([question], convert_to_numpy=True)[0]
    store = ensure_store(data_dir=data_dir)
    results = store.search(q_emb, k=k)
    context = "\n\n".join(results)
    prompt = f"Use the following context to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    response = get_client().responses.create(
        model=rag_model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=512,
        temperature=0.0,
    )
    return response.output_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to ask the ingested PDFs")
    parser.add_argument("--k", type=int, default=5, help="Number of context chunks to retrieve")
    parser.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model to use")
    parser.add_argument("--rag-model", default=RAG_MODEL, help="LLM model for synthesis")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory where embeddings/chunks live")
    args = parser.parse_args()
    print(query(args.question, k=args.k, embed_model=args.embed_model, rag_model=args.rag_model, data_dir=args.data_dir))
