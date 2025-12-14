# src/indexing/table_indexer.py

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from src.config import (
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    PROCESSED_DIR
)

TABLE_FAISS = INDEX_DIR / "tables_faiss.index"
TABLE_METADATA = INDEX_DIR / "tables_metadata.pkl"


def convert_table_to_text(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a readable text representation.
    Format: Each row as "col1: val1, col2: val2, ..."
    """
    lines = []
    columns = df.columns.tolist()

    for idx, row in df.iterrows():
        row_dict = {columns[i]: str(row[i]) for i in range(len(columns))}
        line = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
        lines.append(line)

    table_str = "\n".join(lines)
    return table_str


def chunk_large_table(df: pd.DataFrame, max_rows: int = 20) -> List[Tuple[pd.DataFrame, str]]:
    """
    Split large tables into smaller chunks to improve embedding quality.
    
    Args:
        df: DataFrame to chunk
        max_rows: Maximum rows per chunk
        
    Returns:
        List of (chunk_df, chunk_text) tuples
    """
    chunks = []
    num_rows = len(df)
    
    if num_rows <= max_rows:
        chunks.append((df, convert_table_to_text(df)))
    else:
        # Split table into max_rows-sized chunks
        for start_idx in range(0, num_rows, max_rows):
            end_idx = min(start_idx + max_rows, num_rows)
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_text = convert_table_to_text(chunk_df)
            chunks.append((chunk_df, chunk_text))
    
    return chunks


def build_table_index(table_paths: List[Path], chunk_large_tables: bool = True) -> None:
    """
    Build FAISS index for tables extracted from PDFs.
    
    Args:
        table_paths: List of CSV file paths containing extracted tables
        chunk_large_tables: Whether to split large tables into chunks
    """
    print("Building table index...")
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = []
    metadata = []

    for table_path in table_paths:
        try:
            df = pd.read_csv(table_path)
            
            # Optionally chunk large tables
            if chunk_large_tables:
                chunks = chunk_large_table(df, max_rows=20)
            else:
                chunks = [(df, convert_table_to_text(df))]
            
            for chunk_idx, (chunk_df, text_repr) in enumerate(chunks):
                # Encode table text
                emb = model.encode([text_repr], convert_to_numpy=True, normalize_embeddings=True)[0]
                embeddings.append(emb)
                
                # Store metadata
                metadata.append({
                    "source": table_path.name,
                    "path": str(table_path),
                    "table_text": text_repr,
                    "chunk_idx": chunk_idx,
                    "rows": len(chunk_df),
                    "modality": "table"
                })
            
            print(f"Indexed {table_path.name} ({len(chunks)} chunk(s))")
            
        except Exception as e:
            print(f"  Error indexing {table_path.name}: {e}")
            continue

    if len(embeddings) == 0:
        print("No table embeddings created. Check your table files.")
        return

    # Create and save FAISS index
    embeddings_array = np.array(embeddings).astype("float32")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use inner product for normalized embeddings
    index.add(embeddings_array)

    faiss.write_index(index, str(TABLE_FAISS))
    with open(TABLE_METADATA, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Table index created: {len(metadata)} table chunks indexed")
    print(f"   Saved to: {TABLE_FAISS}")
    print(f"   Metadata: {TABLE_METADATA}")


def load_table_index() -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS table index and metadata.
    
    Returns:
        (index, metadata_list) tuple
    """
    if not TABLE_FAISS.exists() or not TABLE_METADATA.exists():
        raise FileNotFoundError("Table index not found. Build it first with build_table_index().")
    
    index = faiss.read_index(str(TABLE_FAISS))
    with open(TABLE_METADATA, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata
