"""Core RAG pipeline: document loading, chunking, ChromaDB store, and dual
generation with the local **GPT4All** desktop LLM and the cloud **Gemini**
API.

Shared by `ingest.py`, `app.py` (Streamlit UI) and `cli.py`.

Design
------
1. Documents (`.txt` and `.pdf`) under `documents/` are chunked and embedded
   into a persistent ChromaDB collection (using Chroma's built-in default
   embedding function, so no extra embedding model/API is required).
2. A query is embedded and the top-k most similar chunks are retrieved from
   Chroma -- this part is identical regardless of which LLM answers.
3. The same retrieved context is sent to *two* generators:
     - `answer_with_gpt4all`: a local GPT4All model (fully offline).
     - `answer_with_gemini`:  Google's Gemini API (needs GEMINI_API_KEY).
   `answer_dual` runs both and optionally asks Gemini to reconcile the two
   answers into one final, cited response.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from pypdf import PdfReader

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "gpt4all_gemini_rag_collection"

# Local GPT4All model. Auto-downloaded on first use into ~/.cache/gpt4all.
# Swap for any model name from https://gpt4all.io/models/models3.json
GPT4ALL_MODEL_NAME = os.environ.get("GPT4ALL_MODEL_NAME", "orca-mini-3b-gguf2-q4_0.gguf")

# Gemini model used for the cloud answer + optional reconciliation step.
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CHUNK_SIZE = 1200        # characters per chunk
CHUNK_OVERLAP = 200      # characters shared between consecutive chunks
TOP_K = 4                # number of chunks retrieved per query


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


@dataclass
class DualAnswer:
    query: str
    chunks: list[RetrievedChunk]
    gpt4all_answer: str
    gemini_answer: str
    synthesized_answer: str | None = None


# --------------------------------------------------------------------------
# Chunking + ingestion
# --------------------------------------------------------------------------

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


def extract_text(doc_path: Path) -> str:
    if doc_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(doc_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return doc_path.read_text(encoding="utf-8", errors="ignore")


def get_chroma_collection():
    """Return (creating if necessary) the persistent Chroma collection.

    Uses Chroma's default embedding function (ONNX MiniLM) so retrieval
    works even without GPT4All or a Gemini API key configured.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(docs_dir: Path = DOCS_DIR, reset: bool = True) -> int:
    """Chunk every .txt/.pdf under `docs_dir` and (re)populate ChromaDB.

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

    doc_paths = sorted(
        glob.glob(str(docs_dir / "*.txt")) + glob.glob(str(docs_dir / "*.pdf"))
    )
    if not doc_paths:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {docs_dir}. Run create_sample_docs.py "
            "first, or drop your own documents into that folder."
        )

    ids, documents, metadatas = [], [], []
    for doc_path in doc_paths:
        source_name = Path(doc_path).name
        text = extract_text(Path(doc_path))
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
    "context excerpts from the user's documents. If the answer is not "
    "contained in the context, say you don't know instead of guessing. "
    "Always mention which source document(s) you used."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above."
    )


# --------------------------------------------------------------------------
# Generator 1: local GPT4All desktop LLM
# --------------------------------------------------------------------------

_gpt4all_model = None  # lazy-loaded singleton, avoids reloading weights per call


def _get_gpt4all_model():
    global _gpt4all_model
    if _gpt4all_model is None:
        from gpt4all import GPT4All

        _gpt4all_model = GPT4All(GPT4ALL_MODEL_NAME, allow_download=True)
    return _gpt4all_model


def answer_with_gpt4all(query: str, chunks: list[RetrievedChunk] | None = None, top_k: int = TOP_K) -> tuple[str, list[RetrievedChunk]]:
    """RAG turn answered by the local GPT4All model."""
    if chunks is None:
        chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    model = _get_gpt4all_model()
    with model.chat_session():
        answer = model.generate(prompt, max_tokens=512, temp=0.2)
    return answer.strip(), chunks


# --------------------------------------------------------------------------
# Generator 2: Gemini API
# --------------------------------------------------------------------------

_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a key from https://aistudio.google.com/apikey "
                "and set it as an environment variable (or in a local .env file)."
            )
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _gemini_model


def answer_with_gemini(query: str, chunks: list[RetrievedChunk] | None = None, top_k: int = TOP_K) -> tuple[str, list[RetrievedChunk]]:
    """RAG turn answered by the Gemini API."""
    if chunks is None:
        chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    model = _get_gemini_model()
    response = model.generate_content(prompt)
    return response.text.strip(), chunks


# --------------------------------------------------------------------------
# Combined: retrieve once, ask both LLMs, optionally reconcile with Gemini
# --------------------------------------------------------------------------

def answer_dual(query: str, top_k: int = TOP_K, synthesize: bool = True) -> DualAnswer:
    """Retrieve context once and get answers from both GPT4All and Gemini.

    If `synthesize` is True (and Gemini is configured), Gemini is asked a
    second time to reconcile the two answers into a single best response
    grounded in the same retrieved context.
    """
    chunks = retrieve(query, top_k=top_k)

    try:
        gpt4all_answer, _ = answer_with_gpt4all(query, chunks=chunks)
    except Exception as exc:  # model not installed / download failed / etc.
        gpt4all_answer = f"[GPT4All error] {exc}"

    try:
        gemini_answer, _ = answer_with_gemini(query, chunks=chunks)
    except Exception as exc:  # missing API key / network error / etc.
        gemini_answer = f"[Gemini error] {exc}"

    synthesized_answer = None
    if synthesize and GEMINI_API_KEY:
        try:
            synthesized_answer = _reconcile_with_gemini(query, chunks, gpt4all_answer, gemini_answer)
        except Exception as exc:
            synthesized_answer = f"[Synthesis error] {exc}"

    return DualAnswer(
        query=query,
        chunks=chunks,
        gpt4all_answer=gpt4all_answer,
        gemini_answer=gemini_answer,
        synthesized_answer=synthesized_answer,
    )


def _reconcile_with_gemini(
    query: str,
    chunks: list[RetrievedChunk],
    gpt4all_answer: str,
    gemini_answer: str,
) -> str:
    """Ask Gemini to act as a judge/reconciler over the two candidate answers."""
    context_blocks = "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)
    prompt = (
        "You are reviewing two candidate answers to the same question, both "
        "grounded in the same document context. Produce one final answer that "
        "keeps whatever is correct and well-supported, corrects any "
        "hallucinated or unsupported claims, and cites the source document(s). "
        "If both answers already agree, just clean up the wording.\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {query}\n\n"
        f"Candidate answer A (GPT4All, local model):\n{gpt4all_answer}\n\n"
        f"Candidate answer B (Gemini):\n{gemini_answer}\n\n"
        "Final reconciled answer:"
    )
    model = _get_gemini_model()
    response = model.generate_content(prompt)
    return response.text.strip()
