"""Ingest all .txt/.pdf files in documents/ into the persistent ChromaDB
collection used by the GPT4All + Gemini RAG pipeline.

Usage:
    python ingest.py            # (re)build the index from scratch
    python ingest.py --no-reset # add to the existing index instead
"""

from __future__ import annotations

import argparse

from rag_pipeline import ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not wipe the existing collection before ingesting.",
    )
    args = parser.parse_args()

    n_chunks = ingest_documents(reset=not args.no_reset)
    print(f"Indexed {n_chunks} chunks into ChromaDB.")


if __name__ == "__main__":
    main()
