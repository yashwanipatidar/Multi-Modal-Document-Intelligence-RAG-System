# src/indexing/vector_store_tables.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, INDEX_DIR

TABLE_FAISS = INDEX_DIR / "tables_faiss.index"
TABLE_METADATA = INDEX_DIR / "tables_metadata.pkl"


def search_table_index(query, top_k=3):
    if not TABLE_FAISS.exists():
        return []

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    q_emb = model.encode([query]).astype("float32")

    index = faiss.read_index(str(TABLE_FAISS))
    scores, ids = index.search(q_emb, top_k)

    with open(TABLE_METADATA, "rb") as f:
        metadata = pickle.load(f)

    results = []
    for idx in ids[0]:
        results.append(metadata[idx])

    return results
