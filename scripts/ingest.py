"""Ingest documents into the Chroma knowledge base.

Usage:
    python scripts/ingest.py data/
    python scripts/ingest.py data/ --clear   # wipe collection first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.observability import setup_observability  # noqa: E402
from src.tools.rag import ingest_documents  # noqa: E402

setup_observability()


def clear_collection() -> None:
    """Delete the persisted Chroma collection directory."""
    import shutil

    persist_dir = Path(settings.rag_persist_dir)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"Cleared {persist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument("source", type=str, help="Directory containing .txt/.md/.pdf files.")
    parser.add_argument(
        "--clear", action="store_true", help="Wipe the collection before ingesting."
    )
    args = parser.parse_args()

    if args.clear:
        clear_collection()

    # Force RAG on for this script run.
    settings.rag_enabled = True

    count = ingest_documents(args.source)
    print(f"\nIngested {count} chunks from {args.source} into {settings.rag_persist_dir}")


if __name__ == "__main__":
    main()