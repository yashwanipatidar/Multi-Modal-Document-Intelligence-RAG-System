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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever.rag_pipeline import answer_query, answer_query_grouped_by_modality
from src.config import RAW_DOCS_DIR, PROCESSED_DIR


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


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title(" Configuration")

    # Index building section
    st.subheader(" Document Index")

    if st.button(" Build/Rebuild Index", use_container_width=True):
        with st.spinner("Building index... This may take a moment..."):
            try:
                from src.ingestion.pdf_text_extractor import ingest_all_pdfs
                from src.indexing.multi_modal_store import build_multi_modal_index
                from src.ingestion.pdf_table_extractor import extract_all_tables

                # Step 1: Ingest PDFs
                st.status("Step 1: Extracting text and images from PDFs...")
                chunks = ingest_all_pdfs(include_images=True, ocr_enabled=True)
                st.success(f" {len(chunks)} chunks extracted")

                # Step 2: Extract tables
                st.status("Step 2: Extracting tables...")
                table_files = extract_all_tables()
                st.success(f" {len(table_files)} tables found")

                # Step 3: Build multi-modal index
                st.status("Step 3: Building multi-modal index...")
                store = build_multi_modal_index(chunks, table_files)
                st.success(" Index built successfully!")

                st.session_state.index_ready = True
                st.balloons()

            except Exception as e:
                st.error(f" Error building index: {str(e)}")
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
    pdf_count = len(list(RAW_DOCS_DIR.glob("*.pdf")))
    table_dir = PROCESSED_DIR / "tables"
    table_count = len(list(table_dir.glob("*.csv"))) if table_dir.exists() else 0
    image_dir = PROCESSED_DIR / "images"
    image_count = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(" PDFs", pdf_count)
    with col2:
        st.metric(" Tables", table_count)

    col3, col4 = st.columns(2)
    with col3:
        st.metric(" Images", image_count)
    with col4:
        st.metric(" Chunks", "N/A")


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
        st.warning("⚠️ No PDFs found. Please add documents to `data/raw_docs/` and build an index.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                if retrieval_mode == "Unified":
                    result = answer_query(query, top_k=top_k, use_multi_modal=True)
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
                                table_filename = item['source'].split('/')[-1] if '/' in item['source'] else item['source']
                                table_path = PROCESSED_DIR / "tables" / table_filename

                                if table_path.exists():
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
                    result = answer_query_grouped_by_modality(query, top_k=top_k)

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
                                    table_filename = item['source'].split('/')[-1] if '/' in item['source'] else item['source']
                                    table_path = PROCESSED_DIR / "tables" / table_filename

                                    if table_path.exists():
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
