# src/ingestion/pdf_text_extractor.py
from typing import List, Dict, Optional
import pdfplumber
from pathlib import Path
from ..config import RAW_DOCS_DIR, PROCESSED_DIR
import io
from PIL import Image
import pytesseract
import re
import unicodedata

# Handle headless environments
Image.LOAD_TRUNCATED_IMAGES = True


def clean_text_for_chunking(text: str) -> str:
    """Normalize and clean extracted text for better chunking.

    - Normalize unicode and common ligatures
    - Remove hyphenation across line breaks ("ex-\nample" -> "example")
    - Replace single newlines within paragraphs with spaces
    - Collapse multiple spaces/newlines
    """
    if not text:
        return text

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Fix common ligatures
    text = text.replace("\uFB01", "fi").replace("\uFB02", "fl")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # Remove carriage returns
    text = text.replace('\r', '')

    # Remove hyphenation at line breaks: '-\n' or '-  \n' -> ''
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace newlines that are inside paragraphs with spaces (but keep double-newlines as paragraph separators)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple newlines to exactly two (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Trim spaces at line starts/ends
    text = "\n".join([ln.strip() for ln in text.split('\n')])

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """A lightweight sentence splitter using punctuation heuristics.

    Falls back to simple regex splitting on punctuation followed by whitespace.
    """
    if not text:
        return []
    # Ensure single-line for splitting
    s = " ".join([ln.strip() for ln in text.split('\n') if ln.strip()])
    # Split on sentence enders followed by space and a capital letter/number (simple heuristic)
    parts = re.split(r'(?<=[\.\?!])\s+(?=[A-Z0-9])', s)
    # Fallback: if split produced a single long part, try a more permissive split
    if len(parts) == 1:
        parts = re.split(r'(?<=[\.\?!])\s+', s)
    return [p.strip() for p in parts if p.strip()]

def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extracts text page-wise from PDF and returns list of dicts:
    [{"page": int, "content": str, "source": "filename#page", "type": "text"}, ...]
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Clean OCR / PDF-extracted text to fix hyphenation, merged-lines, ligatures
            text = clean_text_for_chunking(text)
            if not text:
                continue

            chunks.append({
                "page": page_num,
                "content": text,
                "source": f"{pdf_path.name}#page_{page_num}",
                "type": "text",
                "modality": "text"
            })
    return chunks


def extract_images_from_pdf(
    pdf_path: Path,
    perform_ocr: bool = True,
    image_output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Extracts images from PDF and optionally performs OCR on them.
    Returns list of dicts with image metadata and OCR text.
    """
    image_chunks = []
    image_dir = image_output_dir or (PROCESSED_DIR / "images")
    image_dir.mkdir(parents=True, exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract images from page
            try:
                images = page.images
            except Exception as e:
                print(f" Could not extract images from page {page_num}: {e}")
                continue
            
            for img_idx, img_obj in enumerate(images):
                try:
                    # Get image from page
                    im = page.within_bbox(img_obj).crop(img_obj).get_image()
                    
                    # Convert to PIL Image - use safe mode
                    img = Image.open(io.BytesIO(im.tobytes()))
                    img = img.convert("RGB")
                    
                    # Save image file
                    img_filename = f"{pdf_path.stem}_p{page_num}_img{img_idx}.png"
                    img_path = image_dir / img_filename
                    img.save(img_path)
                    
                    # Perform OCR if requested
                    ocr_text = ""
                    if perform_ocr:
                        try:
                            ocr_text = pytesseract.image_to_string(img)
                            ocr_text = ocr_text.strip()
                        except Exception as e:
                            print(f"⚠ OCR failed for {img_filename}: {e}")
                    
                    image_chunks.append({
                        "page": page_num,
                        "image_path": str(img_path),
                        "content": ocr_text or f"[Image: {img_filename}]",
                        "source": f"{pdf_path.name}#page_{page_num}_image_{img_idx}",
                        "type": "image",
                        "modality": "image",
                        "image_description": ocr_text if ocr_text else "Visual content extracted from document"
                    })
                    
                except Exception as e:
                    print(f"⚠ Error processing image {img_idx} on page {page_num}: {e}")
                    continue
    
    return image_chunks


def simple_chunk_text(chunks: List[Dict], max_chars: int = 1200) -> List[Dict]:
    """
    Recursive, sentence-aware chunking with character-overlap.

    - Uses `clean_text_for_chunking` to normalize text first.
    - Packs sentences into chunks up to `max_chars`.
    - Emits overlapping chunks using a sliding window with `overlap_chars`.
    """
    def recursive_split_text(text: str, chunk_size: int) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        # Try splitting by paragraph (double newline)
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(parts) > 1:
            out = []
            for p in parts:
                out.extend(recursive_split_text(p, chunk_size))
            return out

        # Try sentence packing
        sentences = split_into_sentences(text)
        if len(sentences) > 1:
            out = []
            current = ""
            for s in sentences:
                if len(current) + len(s) + 1 > chunk_size:
                    if current:
                        out.append(current.strip())
                    current = s + " "
                else:
                    current += s + " "
            if current.strip():
                out.append(current.strip())
            return out

        # Fallback: hard split by whitespace preserving words
        words = text.split()
        out = []
        current = ""
        for w in words:
            if len(current) + len(w) + 1 > chunk_size:
                if current:
                    out.append(current.strip())
                current = w + " "
            else:
                current += w + " "
        if current.strip():
            out.append(current.strip())
        return out

    # Parameters tuned per recommendation
    CHUNK_SIZE = 800 if max_chars is None else max_chars
    OVERLAP = 150

    final_chunks = []
    for entry in chunks:
        if entry.get("type") != "text":
            final_chunks.append(entry)
            continue

        content = clean_text_for_chunking(entry["content"])
        page = entry["page"]
        source = entry["source"]

        pieces = recursive_split_text(content, CHUNK_SIZE)

        # Join pieces into a single string per page and produce sliding window char-overlap chunks
        full = "\n\n".join(pieces)
        if not full:
            continue

        start = 0
        text_len = len(full)
        while start < text_len:
            end = start + CHUNK_SIZE
            chunk_text = full[start:end].strip()
            if not chunk_text:
                break
            final_chunks.append({
                "page": page,
                "content": chunk_text,
                "source": source,
                "type": "text",
                "modality": "text"
            })
            if end >= text_len:
                break
            start = max(0, end - OVERLAP)

    # Post-process: merge very short chunks with neighbors to avoid tiny fragments
    merged = []
    for c in final_chunks:
        if c.get('type') != 'text':
            merged.append(c)
            continue
        if merged and len(c['content']) < int(CHUNK_SIZE * 0.35):
            # merge into previous text chunk
            if merged[-1].get('type') == 'text':
                merged[-1]['content'] = (merged[-1]['content'] + "\n\n" + c['content']).strip()
            else:
                merged.append(c)
        else:
            merged.append(c)

    return merged


def ingest_pdf_paths(
    pdf_paths: List[Path],
    include_images: bool = True,
    ocr_enabled: bool = True,
    image_output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Ingest a specific list of PDF files.

    Args:
        pdf_paths: List of absolute or relative PDF paths
        include_images: If True, extract images and perform OCR
        ocr_enabled: If True, perform OCR on extracted images
        image_output_dir: Optional directory for extracted images

    Returns:
        List of all chunks (text, images) with metadata
    """
    all_chunks = []

    if not pdf_paths:
        print("No PDF paths provided.")
        return []

    for pdf in pdf_paths:
        if not pdf.exists():
            print(f"Skipping missing file: {pdf}")
            continue

        print(f" Processing {pdf.name}...")

        # Extract text
        print(f"  -> Extracting text...")
        page_chunks = extract_text_from_pdf(pdf)
        chunked = simple_chunk_text(page_chunks)
        all_chunks.extend(chunked)
        print(f"{len(chunked)} text chunks extracted")

        # Extract images with OCR
        if include_images:
            print(f"  -> Extracting images with OCR...")
            image_chunks = extract_images_from_pdf(
                pdf,
                perform_ocr=ocr_enabled,
                image_output_dir=image_output_dir,
            )
            all_chunks.extend(image_chunks)
            print(f"{len(image_chunks)} images extracted")

    print(f"\n Total chunks created: {len(all_chunks)}")
    print(f" Text chunks: {sum(1 for c in all_chunks if c.get('modality') == 'text')}")
    print(f" Image chunks: {sum(1 for c in all_chunks if c.get('modality') == 'image')}")

    return all_chunks


def ingest_all_pdfs(include_images: bool = True, ocr_enabled: bool = True) -> List[Dict]:
    """
    Load all PDFs from RAW_DOCS_DIR and return chunk list with text, tables, and optionally images.
    
    Args:
        include_images: If True, extract images and perform OCR
        ocr_enabled: If True, perform OCR on extracted images
        
    Returns:
        List of all chunks (text, images) with metadata
    """
    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DOCS_DIR}. Put your docs there.")
        return []

    return ingest_pdf_paths(
        pdf_paths=pdf_files,
        include_images=include_images,
        ocr_enabled=ocr_enabled,
    )
