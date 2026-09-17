"""Command-line chat REPL for the PDF RAG assistant (no Streamlit needed).

Usage:
    python cli.py
"""

from __future__ import annotations

from rag_pipeline import CHROMA_DIR, OLLAMA_LLM_MODEL, answer_question


def main() -> None:
    if not CHROMA_DIR.exists():
        print("No index found yet. Run `python ingest.py` first.")
        return

    print(f"PDF RAG CLI (model: {OLLAMA_LLM_MODEL}). Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        answer, chunks = answer_question(query)
        print(f"\nAssistant: {answer}\n")
        print("Sources:")
        for c in chunks:
            print(f"  - {c.source} (chunk {c.chunk_index}, distance={c.distance:.3f})")
        print()


if __name__ == "__main__":
    main()
