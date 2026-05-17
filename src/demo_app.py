#!/usr/bin/env python3
# src/demo_app.py
"""
Multi-Modal RAG System - Streamlit Demo Application
Interactive interface for document Q&A with multi-modal retrieval
"""

import streamlit as st
import time
from pathlib import Path
from typing import List, Dict
import sys
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever.rag_pipeline import answer_query, answer_query_grouped_by_modality
from src.config import PROCESSED_DIR
from src.indexing.multi_modal_store import MultiModalVectorStore
from src.operation_tracker import get_operation_tracker, OperationStatus, OperationTimer


MAX_UPLOAD_MB = 25
MAX_FILES_PER_SESSION = 10
SESSIONS_ROOT = PROCESSED_DIR / "sessions"
LOG_DIR = PROCESSED_DIR / "session_logs"


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 12px;
        border-left: 4px solid #0066cc;
        margin: 8px 0;
        border-radius: 4px;
    }
    .answer-box {
        background-color: #f0f8f0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def _get_or_create_tracker():
    """Get or create operation tracker for current session"""
    if "tracker" not in st.session_state:
        st.session_state.tracker = get_operation_tracker(
            st.session_state.session_id,
            log_dir=LOG_DIR
        )
    return st.session_state.tracker


def visualize_table(table_path: Path, table_name: str):
    """Display table as formatted DataFrame with optional visualizations."""
    try:
        df = pd.read_csv(table_path)

        # Always show the formatted table first
        st.markdown("####  Table Data")
        st.dataframe(df, use_container_width=True, height=400)

        # Try to detect if table has numeric data for charts
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_cols) > 0 and len(df) > 1:
            st.markdown("#### Visualizations")

            # Create tabs for different chart types
            viz_tab1, viz_tab2 = st.tabs([" Line Chart", " Bar Chart"])

            with viz_tab1:
                # Line chart for time series data
                if len(numeric_cols) <= 10:  # Reasonable number of lines
                    fig = go.Figure()
                    for col in list(numeric_cols)[:5]:  # Limit to 5 series
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines+markers',
                            name=str(col)
                        ))
                    fig.update_layout(
                        title=f"{table_name} - Trends",
                        xaxis_title="Row Index",
                        yaxis_title="Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Too many numeric columns for line chart (showing first 5)")

            with viz_tab2:
                # Bar chart for first few rows
                if len(df) <= 20:  # Reasonable for bar chart
                    fig = px.bar(
                        df.head(10),
                        x=df.columns[0] if len(df.columns) > 0 else df.index,
                        y=numeric_cols[0] if len(numeric_cols) > 0 else None,
                        title=f"{table_name} - Distribution"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Showing first 10 rows")
                    fig = px.bar(df.head(10), x=df.columns[0], y=numeric_cols[0])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load table: {str(e)}")
        st.text("Table file may be missing or corrupted")


def _get_session_id() -> str:
    """Return a stable session id for the current browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        # Initialize tracker when session is created
        st.session_state.tracker = get_operation_tracker(
            st.session_state.session_id,
            log_dir=LOG_DIR
        )
    return st.session_state.session_id


def _get_session_workspace() -> Dict[str, Path]:
    """Create and return per-session directories."""
    session_id = _get_session_id()
    root = SESSIONS_ROOT / session_id
    raw_dir = root / "raw_docs"
    tables_dir = root / "processed" / "tables"
    images_dir = root / "processed" / "images"
    index_dir = root / "index"

    for path in [raw_dir, tables_dir, images_dir, index_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "raw_docs": raw_dir,
        "tables": tables_dir,
        "images": images_dir,
        "index": index_dir,
    }


def _session_index_paths(workspace: Dict[str, Path]) -> Dict[str, Path]:
    """Resolve index files used by the current session."""
    return {
        "faiss": workspace["index"] / "multi_modal_faiss.index",
        "metadata": workspace["index"] / "multi_modal_metadata.pkl",
    }


def _save_uploaded_files(uploaded_files, raw_docs_dir: Path) -> List[Path]:
    """Validate and persist uploaded PDF files for this session."""
    tracker = _get_or_create_tracker()
    
    if len(uploaded_files) > MAX_FILES_PER_SESSION:
        raise ValueError(f"Please upload up to {MAX_FILES_PER_SESSION} files per session.")

    # Replace files for deterministic indexing in this session
    for existing_pdf in raw_docs_dir.glob("*.pdf"):
        existing_pdf.unlink(missing_ok=True)

    saved_paths: List[Path] = []
    for upload in uploaded_files:
        if not upload.name.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file type: {upload.name}")

        file_size_mb = len(upload.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            raise ValueError(f"{upload.name} exceeds {MAX_UPLOAD_MB} MB")

        destination = raw_docs_dir / Path(upload.name).name
        destination.write_bytes(upload.getvalue())
        saved_paths.append(destination)
        
        # Track each file upload
        tracker.add_detail(f"file_{upload.name}", {
            "size_mb": round(file_size_mb, 2),
            "path": str(destination),
            "status": "saved"
        })

    return saved_paths


def _get_session_store(workspace: Dict[str, Path]) -> MultiModalVectorStore:
    """Instantiate and load session-specific vector store."""
    index_paths = _session_index_paths(workspace)
    store = MultiModalVectorStore(
        index_path=index_paths["faiss"],
        metadata_path=index_paths["metadata"],
    )
    store.load_index()
    return store


# ==================== SIDEBAR ====================
with st.sidebar:
    session_workspace = _get_session_workspace()
    session_index_paths = _session_index_paths(session_workspace)

    st.title(" Configuration")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

    # Upload section
    st.subheader(" Upload PDFs")
    uploaded_files = st.file_uploader(
        "Add one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {MAX_FILES_PER_SESSION} files, {MAX_UPLOAD_MB} MB each",
    )

    if st.button(" Save Uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF file.")
        else:
            tracker = _get_or_create_tracker()
            with OperationTimer(tracker, "PDF Upload", {"file_count": len(uploaded_files)}):
                try:
                    saved = _save_uploaded_files(uploaded_files, session_workspace["raw_docs"])
                    st.session_state.index_ready = False
                    tracker.add_detail("saved_files", len(saved))
                    st.success(f"✅ Saved {len(saved)} PDF file(s) to your private session workspace.")
                except Exception as e:
                    tracker.end_operation(status=OperationStatus.FAILED, error=e)
                    st.error(f"❌ Upload failed: {e}")

    # Index building section
    st.subheader(" Document Index")

    if st.button(" Build/Rebuild Index", use_container_width=True):
        with st.spinner("Building index... This may take a moment..."):
            tracker = _get_or_create_tracker()
            try:
                from src.ingestion.pdf_text_extractor import ingest_pdf_paths
                from src.indexing.multi_modal_store import build_multi_modal_index
                from src.ingestion.pdf_table_extractor import extract_tables_from_paths

                pdf_paths = list(session_workspace["raw_docs"].glob("*.pdf"))
                if not pdf_paths:
                    st.warning("Please upload and save PDFs first.")
                    st.session_state.index_ready = False
                else:
                    # Step 1: Ingest PDFs
                    st.status("Step 1: Extracting text and images from PDFs...")
                    with OperationTimer(tracker, "PDF Text & Image Extraction", {"pdf_count": len(pdf_paths)}):
                        chunks = ingest_pdf_paths(
                            pdf_paths=pdf_paths,
                            include_images=True,
                            ocr_enabled=True,
                            image_output_dir=session_workspace["images"],
                        )
                        tracker.add_detail("chunks_extracted", len(chunks))
                    st.success(f"✅ {len(chunks)} chunks extracted")

                    # Step 2: Extract tables
                    st.status("Step 2: Extracting tables...")
                    with OperationTimer(tracker, "Table Extraction", {"pdf_count": len(pdf_paths)}):
                        table_files = extract_tables_from_paths(
                            pdf_paths=pdf_paths,
                            output_dir=session_workspace["tables"],
                        )
                        tracker.add_detail("tables_extracted", len(table_files))
                    st.success(f"✅ {len(table_files)} tables found")

                    # Step 3: Build multi-modal index
                    st.status("Step 3: Building multi-modal index...")
                    with OperationTimer(tracker, "Multi-Modal Index Building", {"chunks": len(chunks), "tables": len(table_files)}):
                        build_multi_modal_index(
                            chunks,
                            table_files,
                            index_path=session_index_paths["faiss"],
                            metadata_path=session_index_paths["metadata"],
                        )
                        tracker.add_detail("index_file_size_mb", round(session_index_paths["faiss"].stat().st_size / (1024 * 1024), 2))
                    st.success("✅ Index built successfully!")

                    st.session_state.index_ready = True
                    st.balloons()

            except Exception as e:
                tracker.end_operation(status=OperationStatus.FAILED, error=e)
                st.error(f"❌ Error building index: {str(e)}")
                st.session_state.index_ready = False

    # Retrieval settings
    st.subheader(" Retrieval Settings")
    top_k = st.slider(
        "Number of results (top-k)",
        min_value=1,
        max_value=20,
        value=5,
        help="How many documents to retrieve for context"
    )

    retrieval_mode = st.radio(
        "Retrieval Mode",
        options=["Unified", "By Modality"],
        help="Unified: single ranked list | By Modality: separate results per modality"
    )

    temperature = st.slider(
        "Generation Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )

    st.divider()

    # Document info
    st.subheader(" Document Status")
    pdf_count = len(list(session_workspace["raw_docs"].glob("*.pdf")))
    table_count = len(list(session_workspace["tables"].glob("*.csv")))
    image_count = len(list(session_workspace["images"].glob("*.png")))
    index_ready = session_index_paths["faiss"].exists() and session_index_paths["metadata"].exists()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(" PDFs", pdf_count)
    with col2:
        st.metric(" Tables", table_count)

    col3, col4 = st.columns(2)
    with col3:
        st.metric(" Images", image_count)
    with col4:
        st.metric(" Index", "Ready" if index_ready else "Not ready")

    st.divider()

    # Operation tracking
    st.subheader(" Operation Logs")
    tracker = _get_or_create_tracker()
    summary = tracker.get_operations_summary()
    
    if summary["total_operations"] > 0:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Ops", summary["total_operations"])
        with col2:
            st.metric("Success Rate", f"{summary['success_rate']:.0f}%")
        with col3:
            st.metric("Total Time", f"{summary['total_duration']:.1f}s")
        
        # Detailed logs
        if st.checkbox("📋 Show Detailed Logs"):
            operations = tracker.get_operation_details()
            for op in reversed(operations[-10:]):  # Show last 10 operations
                with st.expander(f"[{op['status'].upper()}] {op['operation_name']} ({op['duration_seconds']:.2f}s)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"**Started:** {op['start_time']}")
                        st.caption(f"**Duration:** {op['duration_seconds']:.2f}s")
                    with col2:
                        st.caption(f"**Status:** {op['status']}")
                        if op['error_message']:
                            st.error(f"Error: {op['error_message']}")
                    
                    # Details
                    if op['details']:
                        st.caption("**Details:**")
                        for key, value in op['details'].items():
                            st.text(f"  {key}: {value}")
                    
                    # Warnings
                    if op['warning_messages']:
                        st.caption("**Warnings:**")
                        for warning in op['warning_messages']:
                            st.warning(warning)
    else:
        st.info("No operations logged yet")


# ==================== MAIN CONTENT ====================
st.title(" Multi-Modal Document Intelligence System")
st.markdown("""
**Interactive RAG-based Question Answering**
- Retrieves from text, tables, and images (OCR)
- Source attribution with page numbers
- Multi-modal relevance ranking
""")

st.divider()

# ==================== QUERY INTERFACE ====================
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "❓ Ask a question about your documents:",
        placeholder="e.g., What are the key findings from the report?",
        help="Ask anything about text, tables, or image content"
    )
with col2:
    search_button = st.button("🔍 Search", use_container_width=True, type="primary")

# ==================== RESULTS DISPLAY ====================
if search_button and query:
    if pdf_count == 0:
        st.warning("⚠️ No uploaded PDFs found for this session. Upload and save PDFs first.")
    elif not index_ready:
        st.warning("⚠️ Index is not ready for this session. Click 'Build/Rebuild Index' first.")
    else:
        with st.spinner("Searching and generating answer..."):
            tracker = _get_or_create_tracker()
            try:
                session_store = _get_session_store(session_workspace)

                if retrieval_mode == "Unified":
                    with OperationTimer(tracker, "Query Retrieval (Unified)", {"query": query[:100], "top_k": top_k}):
                        result = answer_query(
                            query,
                            top_k=top_k,
                            use_multi_modal=True,
                            temperature=temperature,
                            store=session_store,
                        )
                        tracker.add_detail("results_found", len(result.get('retrieved', [])))
                        tracker.add_detail("retrieval_time_seconds", result.get('retrieval_time', 0))
                    # Display answer
                    st.markdown("### 📝 Answer")
                    with st.container():
                        st.markdown(result['answer'])
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Retrieval Time", f"{result['retrieval_time']:.2f}s")
                    with col2:
                        st.metric("📍 Sources Used", result['num_results'])
                    with col3:
                        st.metric("🎯 Top Score", f"{result['retrieved'][0]['score']:.3f}" if result['retrieved'] else "N/A")
                    # Display sources
                    st.markdown("### 📚 Sources & Evidence")

                    for i, item in enumerate(result['retrieved'], 1):
                        with st.expander(f"[{i}] {item['source']} ({item['modality'].upper()})", expanded=i<=2):
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                st.caption(f"**Modality:** {item['modality']}")
                            with col2:
                                if item['page'] > 0:
                                    st.caption(f"**Page:** {item['page']}")
                            with col3:
                                st.caption(f"**Score:** {item['score']:.4f}")

                            # If it's a table, try to visualize it
                            if item['modality'] == 'table':
                                table_path_str = item.get("metadata", {}).get("path")
                                table_path = Path(table_path_str) if table_path_str else None
                                table_filename = table_path.name if table_path else item['source']

                                if table_path and table_path.exists():
                                    visualize_table(table_path, table_filename)
                                else:
                                    st.markdown("**Content Preview:**")
                                    st.markdown(f"```\n{item['full_content'][:500]}{'...' if len(item['full_content']) > 500 else ''}\n```")
                            else:
                                st.markdown("**Content Preview:**")
                                st.markdown(f"```\n{item['full_content'][:500]}{'...' if len(item['full_content']) > 500 else ''}\n```")

                    # Full context (collapsible)
                    with st.expander("🔍 Full Context Used", expanded=False):
                        st.markdown(result['context'])

                else:  # By Modality
                    with OperationTimer(tracker, "Query Retrieval (By Modality)", {"query": query[:100], "top_k": top_k}):
                        result = answer_query_grouped_by_modality(
                            query,
                            top_k=top_k,
                            temperature=temperature,
                            store=session_store,
                        )
                        total_results = sum(len(r) for r in result['retrieved_by_modality'].values())
                        tracker.add_detail("results_found", total_results)
                        tracker.add_detail("retrieval_time_seconds", result.get('retrieval_time', 0))

                    # Display answer
                    st.markdown("### 📝 Answer")
                    with st.container():
                        st.markdown(result['answer'])

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("⏱️ Retrieval Time", f"{result['retrieval_time']:.2f}s")
                    with col2:
                        total_results = sum(len(r) for r in result['retrieved_by_modality'].values())
                        st.metric("📍 Total Sources", total_results)

                    # Display results by modality
                    st.markdown("### 🎯 Results by Modality")

                    tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Images", "📊 Tables"])

                    with tab1:
                        if result['retrieved_by_modality']['text']:
                            for i, item in enumerate(result['retrieved_by_modality']['text'], 1):
                                with st.expander(f"Text [{i}] {item['source']}", expanded=i==1):
                                    st.caption(f"Score: {item['score']:.4f}")
                                    st.markdown(item['full_content'][:800])
                        else:
                            st.info("No text results found")

                    with tab2:
                        if result['retrieved_by_modality']['image']:
                            for i, item in enumerate(result['retrieved_by_modality']['image'], 1):
                                with st.expander(f"Image [{i}] {item['source']}", expanded=i==1):
                                    st.caption(f"Page: {item['page']} | Score: {item['score']:.4f}")
                                    if item.get('image_path'):
                                        try:
                                            st.image(item['image_path'], width=400)
                                        except:
                                            st.warning("Could not load image")
                                    st.markdown("**OCR Content:**")
                                    st.markdown(item['full_content'][:800])
                        else:
                            st.info("No image results found")

                    with tab3:
                        if result['retrieved_by_modality']['table']:
                            for i, item in enumerate(result['retrieved_by_modality']['table'], 1):
                                with st.expander(f"📊 Table [{i}] {item['source']}", expanded=i==1):
                                    st.caption(f"**Modality:** table  |  **Score:** {item['score']:.4f}")

                                    # Try to visualize the table
                                    table_path_str = item.get("metadata", {}).get("path")
                                    table_path = Path(table_path_str) if table_path_str else None
                                    table_filename = table_path.name if table_path else item['source']

                                    if table_path and table_path.exists():
                                        visualize_table(table_path, table_filename)
                                    else:
                                        # Fallback to text preview
                                        st.markdown("**Content Preview:**")
                                        st.markdown(item['full_content'][:1000])
                        else:
                            st.info("No table results found")

                    # Full context
                    with st.expander("🔍 Full Context", expanded=False):
                        st.markdown(result['full_context'])

                    # Citations
                    with st.expander("📖 Citation List", expanded=False):
                        st.code(result['citations'], language="text")

            except Exception as e:
                tracker.end_operation(status=OperationStatus.FAILED, error=e)
                st.error(f"❌ Error processing query: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

elif search_button:
    st.warning("⚠️ Please enter a question")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
---
**Multi-Modal RAG System** | Built with Streamlit, FAISS, and Groq
- 📝 Text extraction and chunking
- 🖼️ Image extraction with OCR
- 📊 Table detection and indexing
- 🔍 Multi-modal vector search
- 🤖 LLM-powered answer generation
""")
