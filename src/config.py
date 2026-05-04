# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env file
load_dotenv()


def _get_setting(name: str, default=None):
    """Resolve settings from environment first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return default

# -------------------------
# PATH CONFIGURATION
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

# Create directories if missing
for d in [DATA_DIR, RAW_DOCS_DIR, PROCESSED_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# EMBEDDINGS
# -------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# -------------------------
# FAISS INDEX PATHS
# -------------------------

FAISS_INDEX_PATH = INDEX_DIR / "text_faiss.index"
METADATA_PATH = INDEX_DIR / "text_metadata.pkl"

# -------------------------
# GROQ CONFIG
# -------------------------

GROQ_API_KEY = _get_setting("GROQ_API_KEY")
GROQ_MODEL = _get_setting("GROQ_MODEL", "llama-3.3-70b-versatile")

if GROQ_API_KEY is None:
    print(" WARNING: GROQ_API_KEY not found in environment or Streamlit secrets!")
