# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env file
load_dotenv()

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if GROQ_API_KEY is None:
    print(" WARNING: GROQ_API_KEY not found in .env!")
