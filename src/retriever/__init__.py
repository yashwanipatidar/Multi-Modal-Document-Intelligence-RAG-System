# src/retriever/__init__.py
"""RAG pipeline modules."""

from .rag_pipeline import answer_query, answer_query_grouped_by_modality

__all__ = [
    "answer_query",
    "answer_query_grouped_by_modality"
]
