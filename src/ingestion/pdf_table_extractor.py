# src/ingestion/pdf_table_extractor.py

import camelot
from pathlib import Path
from typing import List, Optional
from src.config import RAW_DOCS_DIR, PROCESSED_DIR

TABLE_DIR = PROCESSED_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_tables_from_paths(
    pdf_paths: List[Path],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Extract tables from a specific list of PDF files.

    Args:
        pdf_paths: List of PDF paths
        output_dir: Directory where extracted CSV files will be written

    Returns:
        List of extracted table CSV paths
    """
    table_files: List[Path] = []
    target_dir = output_dir or TABLE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"Skipping missing file: {pdf_path}")
            continue

        print(f"Extracting tables from: {pdf_path.name}")

        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        except Exception as e:
            print(f" Error reading {pdf_path.name}: {e}")
            continue

        print(f"   -> {len(tables)} tables found")

        for i, table in enumerate(tables):
            df = table.df

            output_path = target_dir / f"{pdf_path.stem}_table_{i}.csv"
            df.to_csv(output_path, index=False)
            table_files.append(output_path)

            print(f"Saved table: {output_path.name}")

    return table_files


def extract_all_tables():
    """
    Extracts tables from all PDFs inside data/raw_docs/
    Saves them as CSV files inside data/processed/tables/
    Returns a list of table file paths.
    """
    pdf_paths = list(RAW_DOCS_DIR.glob("*.pdf"))
    return extract_tables_from_paths(pdf_paths, output_dir=TABLE_DIR)
