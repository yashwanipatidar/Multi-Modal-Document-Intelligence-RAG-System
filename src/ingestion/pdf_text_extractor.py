# src/ingestion/pdf_text_extractor.py
from typing import List, Dict, Optional
import pdfplumber
from pathlib import Path
from ..config import RAW_DOCS_DIR, PROCESSED_DIR
import io
from PIL import Image
import pytesseract

# Handle headless environments
Image.LOAD_TRUNCATED_IMAGES = True

def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extracts text page-wise from PDF and returns list of dicts:
    [{"page": int, "content": str, "source": "filename#page", "type": "text"}, ...]
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
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
    Further split page-level text into smaller semantic-ish chunks based on paragraphs.
    Preserves metadata like type and modality.
    """
    final_chunks = []
    for entry in chunks:
        # Skip non-text chunks in text chunking
        if entry.get("type") != "text":
            final_chunks.append(entry)
            continue
            
        content = entry["content"]
        page = entry["page"]
        source = entry["source"]

        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

        current = ""
        for para in paragraphs:
            # If adding this paragraph exceeds max_chars, flush current
            if len(current) + len(para) + 1 > max_chars:
                if current:
                    final_chunks.append({
                        "page": page,
                        "content": current.strip(),
                        "source": source,
                        "type": "text",
                        "modality": "text"
                    })
                current = para + "\n"
            else:
                current += para + "\n"

        # Flush remaining
        if current.strip():
            final_chunks.append({
                "page": page,
                "content": current.strip(),
                "source": source,
                "type": "text",
                "modality": "text"
            })

    return final_chunks


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
