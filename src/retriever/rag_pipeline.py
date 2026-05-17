# src/retriever/rag_pipeline.py
"""
RAG Pipeline supporting multi-modal retrieval and generation.
Retrieves from text, images (OCR), and tables, then generates citations.
"""

import logging
import random
import time
from typing import List, Dict, Optional, Tuple
from groq import Groq
from ..indexing.multi_modal_store import MultiModalVectorStore
from ..config import GROQ_API_KEY, GROQ_MODEL


# Initialize Groq client (defer initialization until first use)
_client: Optional[Groq] = None
_logger = logging.getLogger(__name__)


def _is_retryable_groq_error(error: Exception) -> bool:
    """Return True for transient Groq/API failures worth retrying."""
    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    response = getattr(error, "response", None)
    if response is not None and getattr(response, "status_code", None) in {429, 500, 502, 503, 504}:
        return True

    message = str(error).lower()
    return any(token in message for token in ["rate limit", "timeout", "temporarily unavailable", "connection error"])


def _call_groq_chat_completion(
    *,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
    max_retries: int = 3,
    base_delay: float = 1.5,
    max_delay: float = 8.0,
) -> Tuple[str, int]:
    """Call Groq with exponential backoff and return the answer plus attempt count."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):
        try:
            response = _get_groq_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content, attempt
        except Exception as error:
            last_error = error
            if attempt > max_retries or not _is_retryable_groq_error(error):
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, 0.25 * delay)
            sleep_seconds = delay + jitter
            _logger.warning(
                "Groq request failed on attempt %s/%s; retrying in %.2fs: %s",
                attempt,
                max_retries + 1,
                sleep_seconds,
                error,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Groq completion failed without raising a concrete error")

def _get_groq_client() -> Groq:
    """Lazily initialize Groq client on first use."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def build_context_from_results(results: List[Dict]) -> str:
    """
    Build a context string from retrieval results with proper formatting.
    Includes source attribution and handles different modalities.
    """
    parts = []

    for i, result in enumerate(results, 1):
        source = result.get('source', 'unknown')
        page = result.get('page', -1)
        modality = result.get('modality', 'unknown')
        content = result.get('content', '')
        full_content = result.get('full_content', content)

        # Format based on modality
        if modality == 'text':
            source_str = f"[TEXT: {source}"
            if page > 0:
                source_str += f", page {page}"
            source_str += "]"
            parts.append(f"{source_str}\n{full_content}")

        elif modality == 'image':
            source_str = f"[IMAGE: {source}, page {page}]"
            ocr_text = full_content or "[Image content - OCR not available]"
            parts.append(f"{source_str}\nOCR Content:\n{ocr_text}")

        elif modality == 'table':
            source_str = f"[TABLE: {source}]"
            parts.append(f"{source_str}\n{full_content}")

        else:
            # Generic handling
            source_str = f"[{modality.upper()}: {source}]"
            parts.append(f"{source_str}\n{full_content}")

    return "\n\n---\n\n".join(parts)


def create_citations_summary(results: List[Dict]) -> str:
    """
    Create a summary of sources for citation purposes.

    Format:
    Citation [1]: source (page X) - modality
    """
    citations = []
    for i, result in enumerate(results, 1):
        source = result.get('source', 'unknown')
        page = result.get('page', -1)
        modality = result.get('modality', 'unknown')

        cite_str = f"[{i}] {source}"
        if page > 0:
            cite_str += f" (page {page})"
        cite_str += f" - {modality}"

        citations.append(cite_str)

    return "\n".join(citations)


def _normalize_text_only_results(results: List[Dict]) -> List[Dict]:
    """Normalize legacy text-only search results to the multi-modal result shape."""
    normalized = []

    for item in results:
        meta = item.get("metadata", {}) or {}
        text_content = item.get("text", "")
        normalized.append(
            {
                "score": float(item.get("score", 0.0)),
                "content": text_content,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", -1),
                "type": "text",
                "modality": "text",
                "full_content": text_content,
                "image_path": None,
                "metadata": meta,
            }
        )

    return normalized


def answer_query(
    query: str,
    top_k: int = 5,
    use_multi_modal: bool = True,
    temperature: float = 0.1,
    store: Optional[MultiModalVectorStore] = None,
) -> Dict:
    """
    Generate answer from multi-modal retrieval results.

    Args:
        query: User question
        top_k: Number of context items to retrieve
        use_multi_modal: Whether to use multi-modal index (True) or legacy text-only (False)

    Returns:
        {
            "answer": str,  # Generated answer
            "context": str,  # Full context used
            "citations": str,  # Citation list
            "retrieved": List[Dict],  # Raw retrieval results
            "retrieval_time": float,  # Time to retrieve
        }
    """
    import time

    retrieval_start = time.time()

    # ==================== RETRIEVAL ====================
    if use_multi_modal:
        try:
            active_store = store or MultiModalVectorStore()
            if active_store.index is None or active_store.metadata is None:
                active_store.load_index()
            retrieved = active_store.search(query, top_k=top_k)
        except FileNotFoundError:
            print("⚠ Multi-modal index not found. Falling back to text-only retrieval...")
            from ..indexing.vector_store import search_text_index
            retrieved = _normalize_text_only_results(search_text_index(query, top_k=top_k))
    else:
        from ..indexing.vector_store import search_text_index
        retrieved = _normalize_text_only_results(search_text_index(query, top_k=top_k))

    retrieval_time = time.time() - retrieval_start

    # ==================== NO RESULTS ====================
    if not retrieved:
        return {
            "answer": "No relevant information found in the document collection.",
            "context": "",
            "citations": "",
            "retrieved": [],
            "retrieval_time": retrieval_time
        }

    # ==================== BUILD CONTEXT & CITATIONS ====================
    context = build_context_from_results(retrieved)
    citations = create_citations_summary(retrieved)

    # ==================== LLM PROMPT ====================
    system_prompt = """You are a precise and factual assistant specialized in document analysis.
Your task is to answer questions STRICTLY based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. Do NOT hallucinate or add information not in the context
3. Be concise and clear
4. Always cite your sources using [citation number] format
5. If information is unclear or not in context, say "Not found in provided context"
6. When referencing tables, be specific about data points
7. When referencing images, describe what's shown and cite the image source"""

    user_message = f"""CONTEXT (numbered citations):
{context}

SOURCES:
{citations}

QUESTION:
{query}

ANSWER (cite using [1], [2], etc. format):"""

    # ==================== GROQ COMPLETION ====================
    answer_text, llm_attempts = _call_groq_chat_completion(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
    )

    # ==================== RETURN RESULT ====================
    return {
        "answer": answer_text,
        "context": context,
        "citations": citations,
        "retrieved": retrieved,
        "retrieval_time": retrieval_time,
        "query": query,
        "num_results": len(retrieved),
        "llm_attempts": llm_attempts,
        "llm_retries": max(0, llm_attempts - 1),
    }


def answer_query_grouped_by_modality(
    query: str,
    top_k: int = 3,
    temperature: float = 0.1,
    store: Optional[MultiModalVectorStore] = None,
) -> Dict:
    """
    Answer query with separate retrieval for each modality.
    Useful for understanding which modalities contributed to the answer.

    Returns:
        {
            "answer": str,
            "text_context": str,
            "image_context": str,
            "table_context": str,
            "full_context": str,
            "citations": str,
            "retrieved_by_modality": {
                "text": [...],
                "image": [...],
                "table": [...]
            }
        }
    """
    import time

    retrieval_start = time.time()

    try:
        active_store = store or MultiModalVectorStore()
        if active_store.index is None or active_store.metadata is None:
            active_store.load_index()
        results_by_modality = active_store.search_by_modality(query, top_k=top_k)
    except FileNotFoundError:
        print("⚠ Multi-modal index not found.")
        return {"answer": "Index not found", "retrieved_by_modality": {}}

    retrieval_time = time.time() - retrieval_start

    # Build separate contexts
    text_context = build_context_from_results(results_by_modality['text']) if results_by_modality['text'] else "[No text context]"
    image_context = build_context_from_results(results_by_modality['image']) if results_by_modality['image'] else "[No image context]"
    table_context = build_context_from_results(results_by_modality['table']) if results_by_modality['table'] else "[No table context]"

    all_results = results_by_modality['text'] + results_by_modality['image'] + results_by_modality['table']
    full_context = build_context_from_results(all_results)
    citations = create_citations_summary(all_results)

    # Generate answer using full context
    system_prompt = """You are a precise document analysis assistant.
Answer based ONLY on provided context. Cite sources. Be factual."""

    user_message = f"""CONTEXT:
{full_context}

SOURCES:
{citations}

QUESTION:
{query}

Answer (use [1], [2], etc. for citations):"""

    answer_text, llm_attempts = _call_groq_chat_completion(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
    )

    return {
        "answer": answer_text,
        "query": query,
        "text_context": text_context,
        "image_context": image_context,
        "table_context": table_context,
        "full_context": full_context,
        "citations": citations,
        "retrieved_by_modality": results_by_modality,
        "retrieval_time": retrieval_time,
        "llm_attempts": llm_attempts,
        "llm_retries": max(0, llm_attempts - 1),
    }
