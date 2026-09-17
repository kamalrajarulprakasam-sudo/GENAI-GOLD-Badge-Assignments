"""Command-line chat REPL for the GPT4All + Gemini RAG assistant.

Prints the local GPT4All answer, the Gemini answer, and (if GEMINI_API_KEY
is set) a Gemini-reconciled final answer -- all grounded in the same
ChromaDB-retrieved context.

Usage:
    python cli.py
"""

from __future__ import annotations

from rag_pipeline import CHROMA_DIR, GEMINI_API_KEY, GEMINI_MODEL_NAME, GPT4ALL_MODEL_NAME, answer_dual


def main() -> None:
    if not CHROMA_DIR.exists():
        print("No index found yet. Run `python ingest.py` first.")
        return

    print(f"GPT4All + Gemini RAG CLI")
    print(f"  GPT4All model: {GPT4ALL_MODEL_NAME}")
    print(f"  Gemini model:  {GEMINI_MODEL_NAME} ({'configured' if GEMINI_API_KEY else 'NO API KEY SET'})")
    print("Type 'exit' to quit.\n")

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

        result = answer_dual(query)

        print(f"\n[GPT4All]: {result.gpt4all_answer}\n")
        print(f"[Gemini]: {result.gemini_answer}\n")
        if result.synthesized_answer:
            print(f"[Synthesized]: {result.synthesized_answer}\n")

        print("Sources:")
        for c in result.chunks:
            print(f"  - {c.source} (chunk {c.chunk_index}, distance={c.distance:.3f})")
        print()


if __name__ == "__main__":
    main()
