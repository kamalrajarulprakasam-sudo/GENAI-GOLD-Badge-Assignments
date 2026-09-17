# Problem 3 — RAG with GPT4All + Gemini + ChromaDB

A Python Retrieval-Augmented-Generation pipeline that:

1. Chunks and embeds documents (`.txt` and/or `.pdf`) into a persistent
   **ChromaDB** collection (using Chroma's built-in ONNX MiniLM embedding
   function — no extra embedding API needed).
2. Retrieves the top-k most relevant chunks for a user question.
3. Sends the **same** retrieved context to two different LLMs:
   - **GPT4All** — a local, fully offline desktop LLM (via the `gpt4all`
     Python bindings).
   - **Gemini** — Google's cloud LLM (via `google-generativeai`).
4. Optionally asks Gemini a second time to **reconcile** the two answers
   into one final, source-cited response (a simple "LLM-as-judge" step).

This lets you directly compare a fully local model against a frontier cloud
model answering from identical retrieved context.

> **Run it in Google Colab instead:** open [Problem3_Colab.ipynb](Problem3_Colab.ipynb) —
> it installs all dependencies, prompts for an optional `GEMINI_API_KEY`,
> generates the sample documents, builds the ChromaDB index, and runs a
> sample dual-answer query.

## Project layout

```
Problem3_GPT4All_Gemini_RAG_ChromaDB/
├── create_sample_docs.py   # generates 3 sample .txt documents
├── rag_pipeline.py         # chunking, ChromaDB, GPT4All + Gemini generation
├── ingest.py                # CLI: (re)build the ChromaDB index
├── cli.py                    # CLI chat REPL (no Streamlit needed)
├── app.py                    # Streamlit side-by-side chat UI
├── requirements.txt
├── documents/                # sample_docs.py output / your own .txt or .pdf files
└── chroma_db/                 # persistent vector store (created on first ingest)
```

## Prerequisites

- Python 3.10+
- The [GPT4All](https://www.nomic.ai/gpt4all) desktop app is optional — the
  `gpt4all` **pip package** used here downloads and runs the same `.gguf`
  models directly, no separate desktop install required.
- A **Gemini API key** — free tier available at
  <https://aistudio.google.com/apikey>.

## Setup

```powershell
cd "Problem3_GPT4All_Gemini_RAG_ChromaDB"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Gemini API key (either as an environment variable, or in a local
`.env` file which `rag_pipeline.py` auto-loads via `python-dotenv`):

```powershell
# Option A: environment variable for this session
$env:GEMINI_API_KEY = "your-api-key-here"

# Option B: create a .env file next to this README
"GEMINI_API_KEY=your-api-key-here" | Out-File -Encoding utf8 .env
```

Optional overrides (env vars):

| Variable | Default | Notes |
|---|---|---|
| `GPT4ALL_MODEL_NAME` | `orca-mini-3b-gguf2-q4_0.gguf` | Small (~1.9 GB), CPU-friendly. Swap for e.g. `Meta-Llama-3-8B-Instruct.Q4_0.gguf` for higher quality (needs more RAM/VRAM). Auto-downloaded on first use into `~/.cache/gpt4all`. |
| `GEMINI_MODEL_NAME` | `gemini-1.5-flash` | Any Gemini model name available to your API key. |

## Usage

```powershell
# 1. Generate 3 sample .txt documents (skip if you already added your own to documents/)
python create_sample_docs.py

# 2. Build the ChromaDB index
python ingest.py

# 3a. Chat via CLI (prints GPT4All answer, Gemini answer, and reconciled answer)
python cli.py

# 3b. ...or launch the Streamlit UI
streamlit run app.py
```

The first GPT4All call downloads the configured `.gguf` model (a few GB);
subsequent calls reuse the cached weights. If `GEMINI_API_KEY` isn't set,
the Gemini answer/reconciliation steps report a clear error but the
GPT4All answer still works fully offline.

## How the reconciliation step works

`answer_dual()` in [rag_pipeline.py](rag_pipeline.py) retrieves the context
chunks **once**, then:

1. Asks GPT4All to answer using only that context.
2. Asks Gemini to answer using the same context.
3. If both succeeded and `GEMINI_API_KEY` is set, asks Gemini again — this
   time given the question, the context, **and** both candidate answers —
   to produce a single reconciled answer that keeps well-supported claims,
   drops any hallucinations, and cites sources.

This mirrors a common Gold-Badge pattern: combine a private/offline model
with a stronger cloud model for higher-quality, still-grounded answers.
