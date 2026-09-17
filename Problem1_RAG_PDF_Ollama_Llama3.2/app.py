"""Streamlit chat UI for the PDF RAG assistant.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    CHROMA_DIR,
    DOCS_DIR,
    OLLAMA_LLM_MODEL,
    answer_question,
    ingest_documents,
)

st.set_page_config(page_title="PDF RAG Assistant (Ollama + Llama 3.2)", page_icon="📄")
st.title("📄 PDF RAG Assistant")
st.caption(f"Local RAG over your PDFs, powered by Ollama model `{OLLAMA_LLM_MODEL}` + ChromaDB")

with st.sidebar:
    st.header("Knowledge base")
    pdf_files = sorted(p.name for p in DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
    if pdf_files:
        st.write(f"{len(pdf_files)} PDF(s) found in `documents/`:")
        for name in pdf_files:
            st.write(f"- {name}")
    else:
        st.warning("No PDFs found in `documents/`. Run `create_sample_pdfs.py` or add your own.")

    if st.button("🔄 (Re)build index", use_container_width=True):
        with st.spinner("Chunking PDFs and indexing into ChromaDB..."):
            try:
                n_chunks = ingest_documents(reset=True)
                st.success(f"Indexed {n_chunks} chunks.")
            except FileNotFoundError as exc:
                st.error(str(exc))

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not CHROMA_DIR.exists():
    st.info("No index yet — click **(Re)build index** in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s.source}** (chunk {s.chunk_index}, distance={s.distance:.3f})")

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, chunks = answer_question(prompt, top_k=top_k)
            except Exception as exc:  # e.g. Ollama not running / model missing
                answer, chunks = f"Error calling Ollama: {exc}", []
            st.markdown(answer)
            if chunks:
                with st.expander("Sources"):
                    for c in chunks:
                        st.write(f"- **{c.source}** (chunk {c.chunk_index}, distance={c.distance:.3f})")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})
