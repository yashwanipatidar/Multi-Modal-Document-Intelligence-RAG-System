# src/indexing/vector_store.py
from typing import List, Dict, Tuple
import numpy as np
import faiss
import pickle
from pathlib import Path

from ..config import FAISS_INDEX_PATH, METADATA_PATH
from .embeddings import TextEmbedder

def build_text_index(chunks: List[Dict]) -> None:
    """
    Build FAISS index over text chunks and save with metadata.
    """
    texts = [c["content"] for c in chunks]
    metadata = [{"page": c["page"], "source": c["source"]} for c in chunks]

    embedder = TextEmbedder()
    embeddings = embedder.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if embeddings are normalized
    index.add(embeddings.astype("float32"))

    # Save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    # Save metadata & texts
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"texts": texts, "metadata": metadata}, f)

    print(f"Index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


def load_text_index() -> Tuple[faiss.Index, Dict]:
    """
    Load FAISS index and metadata.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Index or metadata not found. Build index first.")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search_text_index(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search the FAISS index for a query and return top_k chunks with metadata.
    """
    index, meta = load_text_index()
    texts = meta["texts"]
    metadata = meta["metadata"]

    embedder = TextEmbedder()
    query_vec = embedder.encode([query])
    query_vec = query_vec.astype("float32")

    scores, indices = index.search(query_vec, top_k)
    scores = scores[0]
    indices = indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "text": texts[idx],
            "metadata": metadata[idx]
        })
    return results
