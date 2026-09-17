# Problem 1 — Full-Fledged RAG over PDFs with Ollama + Llama 3.2

A complete Retrieval-Augmented-Generation application:

- **Ingestion**: extracts text from ≥3 PDF files, chunks it (sliding window
  with overlap), and stores embeddings in a persistent **ChromaDB**
  collection (using Chroma's built-in ONNX MiniLM embedding function — fully
  offline, no extra model to pull).
- **Retrieval**: top-k semantic search over the chunks for each user
  question.
- **Generation**: the retrieved chunks are stuffed into a prompt and sent to
  a **local Ollama server running `llama3.2`** for the final answer.
- **UI**: a Streamlit chat app (`app.py`) with source citations, plus a
  plain terminal REPL (`cli.py`) for a no-browser option.

> **Run it in Google Colab instead:** open [Problem1_Colab.ipynb](Problem1_Colab.ipynb) —
> it installs Ollama, pulls `llama3.2`, generates the sample PDFs, builds the
> ChromaDB index, and runs a sample query, all with no local setup.

## 1. Prerequisites

1. Install [Ollama](https://ollama.com/download) and make sure it's running
   (it starts a local server on `http://localhost:11434`).
2. Pull the Llama 3.2 model:

   ```powershell
   ollama pull llama3.2
   ```

## 2. Set up the Python environment

```powershell
cd "Problem1_RAG_PDF_Ollama_Llama3.2"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 3. Generate the sample PDFs (or use your own)

```powershell
.venv\Scripts\python create_sample_pdfs.py
```

This writes 3 PDFs to `documents/`: an employee handbook, a coffee-machine
product manual, and an IT security policy — deliberately different topics so
retrieval quality is easy to verify. You can also drop any of your own PDFs
into `documents/` instead/in addition; `ingest.py` picks up every `*.pdf` file
in that folder.

## 4. Build the vector index

```powershell
.venv\Scripts\python ingest.py
```

## 5. Run the app

**Streamlit chat UI:**

```powershell
.venv\Scripts\streamlit run app.py
```

**Or the terminal REPL:**

```powershell
.venv\Scripts\python cli.py
```

Ask questions such as:

- "How many PTO days do employees get per year?"
- "How often should I descale the NovaBrew coffee machine?"
- "What MFA methods are approved for admin accounts?"

Each answer cites which source PDF/chunk it came from.

## Configuration

- `OLLAMA_LLM_MODEL` env var overrides the generation model (default
  `llama3.2`).
- `rag_pipeline.py` exposes `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K`
  constants if you want to tune retrieval behavior.

## File overview

| File | Purpose |
|------|---------|
| `create_sample_pdfs.py` | Generates 3 sample PDFs into `documents/` |
| `rag_pipeline.py` | Shared core: PDF loading, chunking, Chroma index, Ollama chat |
| `ingest.py` | CLI to (re)build the ChromaDB index from `documents/` |
| `app.py` | Streamlit chat UI |
| `cli.py` | Plain terminal chat REPL |
| `requirements.txt` | Python dependencies |
