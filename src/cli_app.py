#!/usr/bin/env python3
# src/cli_app.py
"""
CLI Application for Multi-Modal RAG System
Supports: index building, interactive QA, batch querying, evaluation
"""

import sys
from pathlib import Path
from src.ingestion.pdf_text_extractor import ingest_all_pdfs
from src.ingestion.pdf_table_extractor import extract_all_tables
from src.indexing.multi_modal_store import build_multi_modal_index
from src.retriever.rag_pipeline import answer_query, answer_query_grouped_by_modality
from src.config import RAW_DOCS_DIR


def print_banner():
    """Print ASCII banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║   📄 Multi-Modal Document Intelligence RAG System         ║
║      Interactive Question Answering with Citations        ║
╚═══════════════════════════════════════════════════════════╝
    """)


def build_index_flow():
    """Build or rebuild the multi-modal index."""
    print("\n🔨 Building Multi-Modal Index...")
    
    try:
        # Step 1: Ingest PDFs
        print("\n📥 Step 1: Extracting text and images from PDFs...")
        chunks = ingest_all_pdfs(include_images=True, ocr_enabled=True)
        
        if not chunks:
            print("❌ No chunks extracted. Check your PDFs in data/raw_docs/")
            return
        
        # Step 2: Extract tables
        print("\n📊 Step 2: Extracting tables...")
        table_files = extract_all_tables()
        
        # Step 3: Build multi-modal index
        print("\n🔗 Step 3: Building unified multi-modal index...")
        store = build_multi_modal_index(chunks, table_files)
        
        print("\n✅ Index built successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during index building: {e}")
        import traceback
        traceback.print_exc()


def qa_flow():
    """Interactive QA mode."""
    print("\n💬 QA Mode (Interactive)")
    print("Type 'exit', 'quit', or 'q' to exit")
    print("Commands: 'modality' for per-modality results, 'verbose' for full context\n")
    
    verbose_mode = False
    modality_mode = False
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if not query:
                print("⚠️ Please enter a question")
                continue
            
            # Handle commands
            if query.lower() in {"exit", "quit", "q"}:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == "verbose":
                verbose_mode = not verbose_mode
                print(f"🔧 Verbose mode: {'ON' if verbose_mode else 'OFF'}")
                continue
            
            if query.lower() == "modality":
                modality_mode = not modality_mode
                print(f"🔧 Modality mode: {'ON' if modality_mode else 'OFF'}")
                continue
            
            # Process query
            print("\n🔍 Searching...", end="", flush=True)
            
            if modality_mode:
                result = answer_query_grouped_by_modality(query, top_k=5)
                
                print("\n\n✅ Answer:")
                print("-" * 60)
                print(result['answer'])
                print("-" * 60)
                
                # Summary by modality
                print("\n📊 Results by Modality:")
                text_count = len(result['retrieved_by_modality']['text'])
                image_count = len(result['retrieved_by_modality']['image'])
                table_count = len(result['retrieved_by_modality']['table'])
                
                print(f"  📝 Text:   {text_count} result(s)")
                print(f"  🖼️  Images: {image_count} result(s)")
                print(f"  📊 Tables: {table_count} result(s)")
                
                print(f"\n⏱️ Retrieval time: {result['retrieval_time']:.2f}s")
                
                if verbose_mode:
                    print("\n📚 Full Context:")
                    print("-" * 60)
                    print(result['full_context'])
                    print("-" * 60)
            
            else:
                result = answer_query(query, top_k=5, use_multi_modal=True)
                
                print("\n\n✅ Answer:")
                print("-" * 60)
                print(result['answer'])
                print("-" * 60)
                
                # Show sources
                print("\n📚 Sources:")
                for i, item in enumerate(result['retrieved'], 1):
                    print(f"\n  [{i}] {item['source']}")
                    print(f"      Modality: {item['modality']}")
                    if item['page'] > 0:
                        print(f"      Page: {item['page']}")
                    print(f"      Score: {item['score']:.4f}")
                
                print(f"\n⏱️ Retrieval time: {result['retrieval_time']:.2f}s")
                
                if verbose_mode:
                    print("\n📖 Full Context:")
                    print("-" * 60)
                    print(result['context'])
                    print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def batch_query_flow():
    """Process batch queries from a file."""
    print("\n📋 Batch Query Mode")
    
    query_file = input("Enter path to query file (one query per line): ").strip()
    
    try:
        with open(query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"\n🔄 Processing {len(queries)} queries...\n")
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query[:50]}...", end=" ", flush=True)
            
            result = answer_query(query, top_k=3)
            results.append({
                'query': query,
                'answer': result['answer'],
                'num_sources': result['num_results'],
                'retrieval_time': result['retrieval_time']
            })
            
            print("✓")
        
        # Save results
        output_file = Path("batch_results.txt")
        with open(output_file, 'w') as f:
            for r in results:
                f.write(f"Q: {r['query']}\n")
                f.write(f"A: {r['answer']}\n")
                f.write(f"Sources: {r['num_sources']} | Time: {r['retrieval_time']:.2f}s\n")
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"\n✅ Results saved to {output_file}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {query_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main CLI entry point."""
    print_banner()
    
    # Check for PDFs
    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n⚠️  No PDFs found in {RAW_DOCS_DIR}")
        print("Please add your PDF documents and run option 1 to build the index.\n")
    
    while True:
        print("\n" + "="*60)
        print("Options:")
        print("  1) Build/Rebuild Index")
        print("  2) Interactive QA")
        print("  3) Batch Query (from file)")
        print("  4) Exit")
        print("="*60)
        
        choice = input("\nChoose option (1/2/3/4): ").strip()
        
        if choice == "1":
            build_index_flow()
        elif choice == "2":
            qa_flow()
        elif choice == "3":
            batch_query_flow()
        elif choice == "4":
            print("\n👋 Thank you for using the Multi-Modal RAG System!")
            break
        else:
            print("⚠️ Invalid option. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
