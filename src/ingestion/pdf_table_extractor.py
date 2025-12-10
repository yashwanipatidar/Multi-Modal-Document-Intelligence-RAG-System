# src/ingestion/pdf_table_extractor.py

import camelot
import pandas as pd
from pathlib import Path
from src.config import RAW_DOCS_DIR, PROCESSED_DIR

TABLE_DIR = PROCESSED_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def extract_all_tables():
    """
    Extracts tables from all PDFs inside data/raw_docs/
    Saves them as CSV files inside data/processed/tables/
    Returns a list of table file paths.
    """
    table_files = []

    for pdf_path in RAW_DOCS_DIR.glob("*.pdf"):
        print(f"📄 Extracting tables from: {pdf_path.name}")

        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        except Exception as e:
            print(f"❌ Error reading {pdf_path.name}: {e}")
            continue

        print(f"   → {len(tables)} tables found")

        for i, table in enumerate(tables):
            df = table.df

            output_path = TABLE_DIR / f"{pdf_path.stem}_table_{i}.csv"
            df.to_csv(output_path, index=False)
            table_files.append(output_path)

            print(f"   ✓ Saved table: {output_path.name}")

    return table_files
