import faiss
import numpy as np
import os
import pickle


class VectorStore:
    def __init__(self, dim, index_path="data/faiss.index", chunks_path="data/chunks.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.chunks_path = chunks_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.chunks = []

    def add(self, embeddings, chunks):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.save()

    def search(self, query_emb, k=5):
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)
        D, I = self.index.search(np.array([query_emb]), k)
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
