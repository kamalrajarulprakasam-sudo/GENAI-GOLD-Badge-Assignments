"""Streamlit UI for the GPT4All + Gemini RAG assistant.

Shows the local GPT4All answer and the Gemini answer side by side (both
grounded in the same ChromaDB-retrieved context), plus an optional
Gemini-reconciled final answer.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline import (
    CHROMA_DIR,
    DOCS_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    GPT4ALL_MODEL_NAME,
    answer_dual,
    ingest_documents,
)

st.set_page_config(page_title="GPT4All + Gemini RAG (ChromaDB)", page_icon="🧠")
st.title("🧠 GPT4All + Gemini RAG Assistant")
st.caption(
    f"Same retrieved context, two LLMs: local GPT4All `{GPT4ALL_MODEL_NAME}` "
    f"vs cloud Gemini `{GEMINI_MODEL_NAME}` — backed by ChromaDB."
)

with st.sidebar:
    st.header("Knowledge base")
    doc_files = (
        sorted(p.name for p in DOCS_DIR.glob("*.txt")) + sorted(p.name for p in DOCS_DIR.glob("*.pdf"))
        if DOCS_DIR.exists()
        else []
    )
    if doc_files:
        st.write(f"{len(doc_files)} document(s) found in `documents/`:")
        for name in doc_files:
            st.write(f"- {name}")
    else:
        st.warning("No documents found in `documents/`. Run `create_sample_docs.py` or add your own .txt/.pdf files.")

    if st.button("🔄 (Re)build index", use_container_width=True):
        with st.spinner("Chunking documents and indexing into ChromaDB..."):
            try:
                n_chunks = ingest_documents(reset=True)
                st.success(f"Indexed {n_chunks} chunks.")
            except FileNotFoundError as exc:
                st.error(str(exc))

    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)
    synthesize = st.checkbox("Ask Gemini to reconcile both answers", value=True)

    if not GEMINI_API_KEY:
        st.info("Set GEMINI_API_KEY to enable the Gemini answer + reconciliation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not CHROMA_DIR.exists():
    st.info("No index yet — click **(Re)build index** in the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            result = msg["result"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🖥️ GPT4All (local)**")
                st.markdown(result.gpt4all_answer)
            with col2:
                st.markdown("**☁️ Gemini**")
                st.markdown(result.gemini_answer)
            if result.synthesized_answer:
                st.markdown("**✅ Synthesized (Gemini reconciles both)**")
                st.markdown(result.synthesized_answer)
            if result.chunks:
                with st.expander("Sources"):
                    for c in result.chunks:
                        st.write(f"- **{c.source}** (chunk {c.chunk_index}, distance={c.distance:.3f})")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and asking both LLMs..."):
            result = answer_dual(prompt, top_k=top_k, synthesize=synthesize)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🖥️ GPT4All (local)**")
            st.markdown(result.gpt4all_answer)
        with col2:
            st.markdown("**☁️ Gemini**")
            st.markdown(result.gemini_answer)
        if result.synthesized_answer:
            st.markdown("**✅ Synthesized (Gemini reconciles both)**")
            st.markdown(result.synthesized_answer)
        if result.chunks:
            with st.expander("Sources"):
                for c in result.chunks:
                    st.write(f"- **{c.source}** (chunk {c.chunk_index}, distance={c.distance:.3f})")

    st.session_state.messages.append({"role": "assistant", "result": result})
