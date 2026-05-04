#  Multi-Modal Document Intelligence RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to handle complex, information-rich documents with multiple modalities: **text**, **tables**, and **images (with OCR)**.

##  Features

### Core Capabilities
- **Multi-Modal Ingestion**: Extracts text, tables, and images from PDFs
- **OCR Integration**: Automated optical character recognition for scanned documents
- **Unified Vector Index**: CLIP-based embeddings for cross-modal semantic search
- **Smart Chunking**: Semantic and structural segmentation for LLM-friendly retrieval
- **Citation-Based Answers**: Generated responses include source attribution with page numbers
- **Interactive UI**: Both Streamlit web interface and CLI application

### Technical Highlights
- **FAISS Indexing**: Fast approximate nearest neighbor search
- **Sentence Transformers**: State-of-the-art embeddings (all-mpnet-base, CLIP ViT-B-32)
- **Groq LLaMA Integration**: Fast LLM inference for answer generation
- **Multi-Modal Retrieval**: Grouped results by modality for interpretability
- **Evaluation Suite**: Comprehensive benchmarking with metrics tracking

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/navigate to project directory
cd RAGASSIGNEMENT

# Create virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependency for OCR (Windows)
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use: choco install tesseract (if using Chocolatey)
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional: OpenAI fallback (not required)
OPENAI_API_KEY=your_openai_key_here
```

Get a free Groq API key from: [https://console.groq.com](https://console.groq.com)

### Streamlit Community Cloud Secrets

If you deploy on Streamlit Community Cloud, add the same values in the app's Secrets panel using valid TOML:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Do not paste log text, colons, or extra labels into Secrets.

### 3. Add Documents

Place your PDF documents in:
```
data/raw_docs/
```

Example: `data/raw_docs/report.pdf`

### 4. Build Index

**Via CLI:**
```bash
python -m src.cli_app
# Select option 1 to build index
```

**Programmatically:**
```python
from src.ingestion.pdf_text_extractor import ingest_all_pdfs
from src.ingestion.pdf_table_extractor import extract_all_tables
from src.indexing.multi_modal_store import build_multi_modal_index

chunks = ingest_all_pdfs(include_images=True, ocr_enabled=True)
tables = extract_all_tables()
store = build_multi_modal_index(chunks, tables)
```

### 5. Run Interactive QA

**Streamlit Web UI (Recommended):**
```bash
streamlit run src/demo_app.py
```
Then open browser to `http://localhost:8501`

**CLI Application:**
```bash
python -m src.cli_app
# Select option 2 for interactive QA
```

---

## 📁 Project Structure

```
RAGASSIGNEMENT/
├── data/
│   ├── raw_docs/              # Input PDFs
│   ├── processed/
│   │   ├── tables/            # Extracted CSV files
│   │   └── images/            # Extracted images
│   └── index/
│       ├── multi_modal_faiss.index      # FAISS index
│       ├── multi_modal_metadata.pkl     # Metadata
│       └── ...
├── src/
│   ├── ingestion/
│   │   ├── pdf_text_extractor.py        # Text & image extraction
│   │   └── pdf_table_extractor.py       # Table extraction
│   ├── indexing/
│   │   ├── embeddings.py                # Text & multi-modal embedders
│   │   ├── vector_store.py              # Text-only vector store
│   │   ├── multi_modal_store.py         # Unified multi-modal store
│   │   └── table_indexer.py             # Table indexing
│   ├── retriever/
│   │   └── rag_pipeline.py              # Answer generation pipeline
│   ├── evaluation/
│   │   └── evaluator.py                 # Evaluation suite
│   ├── cli_app.py                       # CLI interface
│   ├── demo_app.py                      # Streamlit web UI
│   └── config.py                        # Configuration
├── requirements.txt                     # Dependencies
├── .env                                 # API keys (create this)
└── README.md                            # This file
```

---

## 🔍 Usage Examples

### Interactive Question Answering

**Web UI (Streamlit):**
1. Open `http://localhost:8501`
2. Click "🔨 Build/Rebuild Index" in sidebar
3. Enter question in main panel
4. View answer with source attribution

**CLI:**
```bash
python -m src.cli_app
> 2
> What are the main findings?
```

### Programmatic Usage

```python
from src.retriever.rag_pipeline import answer_query

# Simple query
result = answer_query("What does the table show?", top_k=5)
print(result['answer'])
print(result['citations'])

# Results grouped by modality
from src.retriever.rag_pipeline import answer_query_grouped_by_modality

result = answer_query_grouped_by_modality("Describe the charts", top_k=3)
print(f"Text results: {len(result['retrieved_by_modality']['text'])}")
print(f"Image results: {len(result['retrieved_by_modality']['image'])}")
print(f"Table results: {len(result['retrieved_by_modality']['table'])}")
```

### Batch Evaluation

```python
from src.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator()

queries = [
    {"query": "What is the main conclusion?", "expected_modalities": ["text"]},
    {"query": "What data is in the tables?", "expected_modalities": ["table"]},
    {"query": "Describe the figures", "expected_modalities": ["image"]},
]

evaluator.evaluate_batch(queries)
evaluator.print_summary()
evaluator.save_results()
```

---

## 🏗️ Architecture

### Document Processing Pipeline

```
PDFs
  ├── Text Extraction (pdfplumber)
  ├── Image Extraction + OCR (pytesseract)
  └── Table Extraction (camelot)
        ↓
  Semantic Chunking (max 1200 chars)
        ↓
  Multi-Modal Embeddings (CLIP ViT-B-32)
        ↓
  FAISS Vector Index (IndexFlatIP)
```

### Retrieval & Answer Generation

```
User Query
  ↓
Embed Query (CLIP encoder)
  ↓
FAISS Similarity Search (top-k)
  ↓
Format Context (with citations)
  ↓
Groq LLaMA-3 Answer Generation
  ↓
Formatted Answer + Sources
```

### Multi-Modal Indexing

- **Text Chunks**: Direct semantic embedding
- **Images**: CLIP vision encoder + OCR fallback
- **Tables**: Semantic text representation with row-level structure

---

## 🎓 Key Design Decisions

### 1. **CLIP-Based Multi-Modal Embeddings**
- Unified embedding space for text and vision
- Robust cross-modal semantic matching
- Better than separate encoders for retrieval diversity

### 2. **Chunking Strategy**
- Page-level + semantic paragraph splitting
- Preserves document structure for citations
- Optimal for LLM context windows (1200 char chunks)

### 3. **Table Handling**
- Convert to semantic text representation
- Chunk large tables to preserve information density
- Maintains row relationships for accurate retrieval

### 4. **Citation Mechanism**
- Track source, page, modality for each chunk
- LLM prompted to cite sources in generated answers
- Enables fact-checking and traceability

### 5. **Evaluation Metrics**
- Retrieval latency and quality
- Modality distribution in results
- Answer faithfulness (via LLM grading)

---

## 📊 Evaluation & Benchmarking

Run the evaluation suite:

```bash
python -c "from src.evaluation.evaluator import run_sample_evaluation; run_sample_evaluation()"
```

Outputs:
- **Retrieval Time**: Average and percentiles
- **Result Count**: Documents retrieved per query
- **Modality Distribution**: How often each modality appears
- **Score Distribution**: Similarity scores of top results

Results saved to: `evaluation_results/evaluation_TIMESTAMP.json`

---

## 🔧 Troubleshooting

### OCR Not Working
```
Error: pytesseract.TesseractNotFoundError
```
**Solution**: Install Tesseract binary
- Windows: [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

### FAISS Index Not Found
```
Error: Multi-modal index not found
```
**Solution**: Build index first via CLI (option 1) or call `build_multi_modal_index()`

### Groq API Error
```
Error: Invalid API key
```
**Solution**: 
1. Get key from [console.groq.com](https://console.groq.com)
2. Add to `.env` file: `GROQ_API_KEY=your_key`
3. Restart application

### Out of Memory
**Solution**: Reduce batch size in embeddings:
```python
from src.indexing.embeddings import MultiModalEmbedder
embedder = MultiModalEmbedder()
embeddings = embedder.encode_text(texts, batch_size=8)  # Reduce from 16
```

---

## 🚀 Advanced Features

### Cross-Modal Reranking (Optional)

```python
from src.retriever.rag_pipeline import answer_query_grouped_by_modality

# Get results grouped by modality
result = answer_query_grouped_by_modality(query, top_k=3)

# Implement custom reranking
text_weight = 0.5
image_weight = 0.3
table_weight = 0.2

# Combine and rerank...
```

### Custom Embedding Models

```python
from src.indexing.embeddings import MultiModalEmbedder
from src.indexing.multi_modal_store import build_multi_modal_index

# Use different CLIP variant
embedder = "sentence-transformers/clip-ViT-L-14"
store = build_multi_modal_index(chunks, tables, embedder_model=embedder)
```

### Fine-tuning Retrieval

Adjust retrieval parameters in `config.py`:
```python
# Chunk size
CHUNK_SIZE = 1200  # characters

# Number of results
TOP_K = 5

# Temperature (lower = more factual)
TEMPERATURE = 0.1
```

---

## 📈 Performance Metrics

Typical performance on standard hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Text extraction | 2-5s per 100-page doc | With pdfplumber |
| Image extraction + OCR | 30-60s per 100 images | Depends on image quality |
| Embedding 1000 chunks | 5-10s | Batch size 16 |
| Similarity search | <100ms | FAISS with 10k items |
| Answer generation | 1-3s | Groq LLaMA-3 API |

---

## 📝 Output Examples

### Answer with Citations

**Q:** "What is the economic outlook?"

**A:**
> Based on the provided documents, the economic outlook shows moderate growth [1] with inflation pressures [2] and strong labor markets [3]. The recovery is expected to continue but at a slower pace [1].

**Sources:**
- [1] Executive Summary (page 2) - text
- [2] Economic Indicators Table (page 5) - table
- [3] Chart: Employment Trends (page 8) - image

---

## 🤝 Contributing

To extend this system:

1. **New Extractors**: Add to `src/ingestion/`
2. **Custom Embeddings**: Extend `src/indexing/embeddings.py`
3. **New Retrieval Methods**: Modify `src/indexing/multi_modal_store.py`
4. **Evaluation Metrics**: Update `src/evaluation/evaluator.py`

---

## 📚 References

- **Sentence Transformers**: [huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
- **CLIP Model**: [OpenAI CLIP](https://openai.com/research/learning-transferable-models-for-computational-pathology)
- **FAISS**: [Facebook Research](https://github.com/facebookresearch/faiss)
- **Groq**: [Groq.com](https://www.groq.com)
- **Streamlit**: [streamlit.io](https://streamlit.io)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Questions?** Check the code comments or run individual components in a Python REPL for debugging.
