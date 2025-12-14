# src/indexing/embeddings.py
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path
from ..config import EMBEDDING_MODEL_NAME

class TextEmbedder:
    """Text-only embedder using sentence-transformers."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f" Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.modality = "text"

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


class MultiModalEmbedder:
    """Multi-modal embedder supporting text, images, and tables using CLIP-like models."""
    
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        """
        Initialize multi-modal embedder.
        Uses CLIP model for vision-text alignment.
        """
        print(f"Loading multi-modal embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.modality = "multi-modal"
        self.text_model = SentenceTransformer(EMBEDDING_MODEL_NAME)  # For pure text
    
    def encode_text(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode text using the multi-modal model for consistency."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
    
    def encode_images(self, image_paths: List[Union[str, Path]], batch_size: int = 8) -> np.ndarray:
        """
        Encode images using CLIP vision encoder.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings (normalized)
        """
        images = []
        valid_paths = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")
        
        if not images:
            return np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
        
        # CLIP encodes images directly
        embeddings = self.model.encode(
            images,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
    
    def encode_mixed(self, items: List[dict], batch_size: int = 16) -> dict:
        """
        Encode a mix of text and image chunks.
        Items should have 'content', 'type' (text/image), and optionally 'image_path'.
        
        Returns:
            {
                'embeddings': np.ndarray,
                'indices': List[int],  # indices of items with valid embeddings
                'texts': List[str],    # original content
                'types': List[str]     # 'text' or 'image'
            }
        """
        embeddings_list = []
        valid_indices = []
        texts = []
        types = []
        
        for i, item in enumerate(items):
            item_type = item.get('type', 'text')
            
            if item_type == 'text':
                # Encode text
                emb = self.encode_text([item['content']])[0]
                embeddings_list.append(emb)
                valid_indices.append(i)
                texts.append(item['content'])
                types.append('text')
                
            elif item_type == 'image':
                # Try to encode image
                img_path = item.get('image_path')
                if img_path and Path(img_path).exists():
                    try:
                        emb = self.encode_images([img_path])[0]
                        embeddings_list.append(emb)
                        valid_indices.append(i)
                        texts.append(item.get('content', f'Image: {Path(img_path).name}'))
                        types.append('image')
                    except Exception as e:
                        print(f"⚠ Failed to encode image {img_path}: {e}")
                        # Fall back to encoding OCR text if available
                        ocr_text = item.get('content', '')
                        if ocr_text:
                            emb = self.encode_text([ocr_text])[0]
                            embeddings_list.append(emb)
                            valid_indices.append(i)
                            texts.append(ocr_text)
                            types.append('image_ocr')
        
        embeddings = np.array(embeddings_list) if embeddings_list else np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
        
        return {
            'embeddings': embeddings.astype('float32'),
            'indices': valid_indices,
            'texts': texts,
            'types': types
        }
    
    def encode_tables(self, tables: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encode table content (as text strings) using the text model.
        Tables are better handled as semantic text rather than images.
        """
        embeddings = self.model.encode(
            tables,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
