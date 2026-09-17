"""Core RAG pipeline: PDF loading, chunking, vector store, and generation.

Shared by `ingest.py`, `app.py` (Streamlit UI) and `cli.py`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_rag_collection"

# The chat/generation model. Must already be pulled: `ollama pull llama3.2`
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")

CHUNK_SIZE = 1200        # characters per chunk
CHUNK_OVERLAP = 200      # characters shared between consecutive chunks
TOP_K = 4                # number of chunks retrieved per query


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker with overlap."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_chroma_collection():
    """Return (creating if necessary) the persistent Chroma collection.

    Uses Chroma's built-in default embedding function (ONNX MiniLM), so no
    extra embedding model needs to be pulled through Ollama.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(docs_dir: Path = DOCS_DIR, reset: bool = True) -> int:
    """Chunk every PDF under `docs_dir` and (re)populate the Chroma collection.

    Returns the number of chunks indexed.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_paths = sorted(glob.glob(str(docs_dir / "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {docs_dir}. Run create_sample_pdfs.py first, "
            "or drop your own PDFs into that folder."
        )

    ids, documents, metadatas = [], [], []
    for pdf_path in pdf_paths:
        source_name = Path(pdf_path).name
        text = extract_pdf_text(Path(pdf_path))
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{source_name}::{i}")
            documents.append(chunk)
            metadatas.append({"source": source_name, "chunk_index": i})

    if documents:
        # Chroma has a per-call batch size limit; chunk the insert.
        batch_size = 128
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    return len(documents)


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    retrieved: list[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", -1),
                distance=dist,
            )
        )
    return retrieved


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided "
    "context excerpts from the user's PDF documents. If the answer is not "
    "contained in the context, say you don't know instead of guessing. "
    "Always mention which source document(s) you used."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above."
    )


def answer_question(query: str, top_k: int = TOP_K, model: str = OLLAMA_LLM_MODEL) -> tuple[str, list[RetrievedChunk]]:
    """Full RAG turn: retrieve relevant chunks, then ask the Ollama LLM."""
    chunks = retrieve(query, top_k=top_k)
    user_prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response["message"]["content"]
    return answer, chunks


def stream_answer(query: str, top_k: int = TOP_K, model: str = OLLAMA_LLM_MODEL):
    """Generator yielding answer text tokens as they stream from Ollama.

    Call `retrieve(query, top_k)` separately if you also need the source
    chunks (e.g. to display citations alongside the streamed answer).
    """
    chunks = retrieve(query, top_k=top_k)
    user_prompt = build_prompt(query, chunks)

    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]
