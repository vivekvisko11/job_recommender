# faiss_index_util.py
import numpy as np
import faiss
import os

def build_faiss_index(embeddings: np.ndarray, dim: int, index_path="faiss.index"):
    # embeddings shape: (n, dim)
    index = faiss.IndexFlatIP(dim)  # dot-product similarity (works with normalized vectors)
    # normalize to use cosine similarity equivalently
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def load_faiss_index(index_path="faiss.index"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    index = faiss.read_index(index_path)
    return index

def search_index(index, query_embedding, top_k=10):
    # query_embedding: 1-d numpy vector
    q = query_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q) 
    D, I = index.search(q, top_k)
    return I[0], D[0]  # indices, scores
