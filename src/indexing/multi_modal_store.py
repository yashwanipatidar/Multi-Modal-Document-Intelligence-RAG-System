# src/indexing/multi_modal_store.py
"""
Unified multi-modal vector store combining text, images, and tables.
Provides retrieval across all modalities with proper source attribution.
"""

from typing import List, Dict, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path

from ..config import INDEX_DIR
from .embeddings import MultiModalEmbedder


# Index file paths
MULTI_MODAL_FAISS = INDEX_DIR / "multi_modal_faiss.index"
MULTI_MODAL_METADATA = INDEX_DIR / "multi_modal_metadata.pkl"


class MultiModalVectorStore:
    """Unified vector store for multi-modal retrieval."""
    
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/clip-ViT-B-32",
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        """Initialize the multi-modal vector store."""
        self.embedder = MultiModalEmbedder(embedder_model)
        self.index = None
        self.metadata = None
        self.index_path = index_path or MULTI_MODAL_FAISS
        self.metadata_path = metadata_path or MULTI_MODAL_METADATA
    
    def build_index(self, chunks: List[Dict], table_paths: List[Path] = None) -> None:
        """
        Build unified FAISS index from text/image chunks and tables.
        
        Args:
            chunks: List of dicts with 'content', 'type', 'source', 'modality', etc.
            table_paths: Optional list of CSV table file paths
        """
        print(" Building unified multi-modal index...")
        
        all_embeddings = []
        all_metadata = []
        
        # ==================== HANDLE TEXT & IMAGE CHUNKS ====================
        if chunks:
            print(f" Processing {len(chunks)} chunks (text + images)...")
            
            # Use the multi-modal embedder for mixed content
            result = self.embedder.encode_mixed(chunks)
            embeddings = result['embeddings']
            valid_indices = result['indices']
            texts = result['texts']
            types = result['types']
            
            # Create metadata for each chunk
            for i, valid_idx in enumerate(valid_indices):
                chunk = chunks[valid_idx]
                metadata_entry = {
                    "source": chunk.get('source', 'unknown'),
                    "page": chunk.get('page', -1),
                    "content": texts[i],
                    "type": chunk.get('type', 'unknown'),
                    "modality": types[i],
                    "embedding_type": "multi-modal",
                }
                
                # Add image-specific metadata
                if chunk.get('type') == 'image':
                    metadata_entry['image_path'] = chunk.get('image_path')
                
                all_embeddings.append(embeddings[i])
                all_metadata.append(metadata_entry)
            
            print(f"    ✓ {len(embeddings)} embeddings created")
        
        # ==================== HANDLE TABLES ====================
        if table_paths and len(table_paths) > 0:
            print(f" Processing {len(table_paths)} tables...")
            
            import pandas as pd
            
            for table_path in table_paths:
                try:
                    df = pd.read_csv(table_path)
                    
                    # Convert table to text
                    from .table_indexer import convert_table_to_text
                    table_text = convert_table_to_text(df)
                    
                    # Encode table text
                    table_emb = self.embedder.encode_text([table_text])[0]
                    all_embeddings.append(table_emb)
                    
                    # Create metadata
                    all_metadata.append({
                        "source": table_path.name,
                        "content": table_text[:500] + "..." if len(table_text) > 500 else table_text,
                        "full_content": table_text,
                        "type": "table",
                        "modality": "table",
                        "embedding_type": "text",
                        "path": str(table_path)
                    })
                    
                    print(f"    ✓ Indexed {table_path.name}")
                    
                except Exception as e:
                    print(f"    ⚠ Error processing {table_path.name}: {e}")
        
        if len(all_embeddings) == 0:
            print("No embeddings created. Check your inputs.")
            return
        
        # ==================== CREATE FAISS INDEX ====================
        embeddings_array = np.array(all_embeddings).astype("float32")
        dim = embeddings_array.shape[1]
        
        # Use IndexFlatIP for cosine similarity with normalized embeddings
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_array)
        
        # Save index and metadata
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(all_metadata, f)
        
        self.index = index
        self.metadata = all_metadata
        
        print(f"\n Multi-modal index built successfully!")
        print(f"   Total items indexed: {len(all_metadata)}")
        print(f"   - Text chunks: {sum(1 for m in all_metadata if m['type'] == 'text')}")
        print(f"   - Images: {sum(1 for m in all_metadata if m['type'] == 'image')}")
        print(f"   - Tables: {sum(1 for m in all_metadata if m['type'] == 'table')}")
        print(f"   Embedding dimension: {dim}")
        print(f"   Saved to: {self.index_path}")
    
    def load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Multi-modal index not found. Build it first.")
        
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f" Loaded multi-modal index with {len(self.metadata)} items")
    
    def search(self, query: str, top_k: int = 5, modality_filter: str = None) -> List[Dict]:
        """
        Search the multi-modal index.
        
        Args:
            query: Search query (text)
            top_k: Number of results to return
            modality_filter: Optional filter ('text', 'image', 'table', or None for all)
            
        Returns:
            List of result dicts with score, content, metadata, etc.
        """
        if self.index is None or self.metadata is None:
            self.load_index()
        
        # Encode query
        query_vec = self.embedder.encode_text([query]).astype("float32")
        
        # Search (retrieve more to account for filtering)
        search_k = top_k * 3 if modality_filter else top_k
        scores, indices = self.index.search(query_vec, min(search_k, len(self.metadata)))
        
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            
            # Apply modality filter if specified
            if modality_filter and meta.get('modality') != modality_filter:
                continue
            
            results.append({
                "score": float(score),
                "content": meta.get('content', ''),
                "source": meta.get('source', 'unknown'),
                "page": meta.get('page', -1),
                "type": meta.get('type', 'unknown'),
                "modality": meta.get('modality', 'unknown'),
                "full_content": meta.get('full_content', meta.get('content', '')),
                "image_path": meta.get('image_path'),
                "metadata": meta
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_modality(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Search and return results grouped by modality.
        
        Returns:
            {
                'text': [...],
                'image': [...],
                'table': [...]
            }
        """
        results = {
            'text': self.search(query, top_k, modality_filter='text'),
            'image': self.search(query, top_k, modality_filter='image'),
            'table': self.search(query, top_k, modality_filter='table')
        }
        return results


# ==================== CONVENIENCE FUNCTIONS ====================

def build_multi_modal_index(chunks: List[Dict], table_paths: List[Path] = None,
                            embedder_model: str = "sentence-transformers/clip-ViT-B-32",
                            index_path: Optional[Path] = None,
                            metadata_path: Optional[Path] = None) -> MultiModalVectorStore:
    """
    Build a multi-modal index from chunks and tables.
    
    Args:
        chunks: List of text and image chunks with metadata
        table_paths: Optional list of CSV file paths
        embedder_model: CLIP or similar multi-modal model
        
    Returns:
        MultiModalVectorStore instance
    """
    store = MultiModalVectorStore(
        embedder_model=embedder_model,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    store.build_index(chunks, table_paths)
    return store


def search_multi_modal_index(query: str, top_k: int = 5, modality_filter: str = None) -> List[Dict]:
    """
    Search the multi-modal index.
    
    Args:
        query: Search query
        top_k: Number of results
        modality_filter: Optional modality filter
        
    Returns:
        List of search results
    """
    store = MultiModalVectorStore()
    store.load_index()
    return store.search(query, top_k, modality_filter)
