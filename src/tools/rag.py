"""RAG knowledge-base tooling (P1.7).

Provides:
- `get_vector_store()` — lazily instantiates a Chroma persistent collection.
- `retrieve_documents(query, k)` — semantic search over local documents.
- `tavily_rag_search` — a LangChain `@tool` exposing retrieval to agents.
- `ingest_documents(path)` — chunk + embed files from a directory.

Chroma runs fully embedded (no separate server), so this works on any laptop
and inside the Docker stack without extra services. Embeddings default to
OpenAI's `text-embedding-3-small`; override via `EMBEDDING_MODEL` env var.

This module degrades gracefully: if chromadb/embeddings are unavailable or the
collection is empty, retrieval returns an explicit "no local docs indexed"
string instead of crashing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

_embedding_model: Any = None
_vector_store: Any = None


class RetrievalQuery(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    k: int = Field(4, ge=1, le=10, description="Top-k documents to return.")


def _build_embedding():
    """Construct the embedding model (OpenAI by default)."""
    from langchain_openai import OpenAIEmbeddings

    embedding_model_name = "text-embedding-3-small"
    return OpenAIEmbeddings(
        model=embedding_model_name, api_key=settings.openai_api_key
    )


def get_embedding():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _build_embedding()
    return _embedding_model


def get_vector_store():
    """Return the singleton Chroma vector store, or None if unavailable."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        from langchain_chroma import Chroma

        persist_dir = Path(settings.rag_persist_dir)
        if not persist_dir.exists():
            logger.warning(
                "RAG persist dir %s does not exist; retrieval will return empty",
                persist_dir,
            )

        _vector_store = Chroma(
            collection_name=settings.rag_collection,
            embedding_function=get_embedding(),
            persist_directory=str(persist_dir),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Chroma unavailable, RAG disabled: %s", e)
        _vector_store = None

    return _vector_store


def retrieve_documents(query: str, k: int = 4) -> list[dict]:
    """Semantic search over the local knowledge base."""
    vs = get_vector_store()
    if vs is None:
        return []

    try:
        results = vs.similarity_search_with_score(query, k=k)
    except Exception as e:  # noqa: BLE001
        logger.warning("RAG retrieval failed: %s", e)
        return []

    docs = []
    for doc, score in results:
        docs.append(
            {
                "content": getattr(doc, "page_content", str(doc)),
                "source": getattr(doc, "metadata", {}).get("source", "unknown"),
                "score": float(score),
            }
        )
    return docs


@tool
def tavily_rag_search(query: str, k: int = 4) -> str:
    """Search the local research knowledge base for relevant documents.

    Use this when the task asks about material that may have been ingested into
    the project's local document store (PDFs, notes, papers under data/).
    Returns up to k matching passages with source attribution.

    Args:
        query: Natural-language search query.
        k: Number of passages to return (1-10).
    """
    if not settings.rag_enabled:
        return (
            "RAG knowledge base is disabled. Ask the user to enable RAG_ENABLED "
            "and run `python scripts/ingest.py data/` to index documents."
        )

    hits = retrieve_documents(query, k=k)
    if not hits:
        return "No matching documents found in the local knowledge base."

    lines = [f"Found {len(hits)} passage(s) from local KB:"]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"\n### Passage {i} (source: {h['source']}, score: {h['score']:.3f})\n"
            f"{h['content']}"
        )
    return "\n".join(lines)


def ingest_documents(source_dir: str | Path) -> int:
    """Chunk + embed all supported files under source_dir into Chroma.

    Returns the number of chunks added. Supports .txt, .md, .pdf (if pypdf is
    installed). Safe to call from CLI: `python scripts/ingest.py data/`.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    vs = get_vector_store()
    if vs is None:
        raise RuntimeError(
            "Vector store unavailable — install `langchain-chroma` and chromadb."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks_added = 0

    for ext in ("*.txt", "*.md"):
        for fp in source_path.rglob(ext):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            chunks = splitter.split_text(text)
            metadatas = [{"source": str(fp)} for _ in chunks]
            vs.add_texts(chunks, metadatas=metadatas)
            chunks_added += len(chunks)
            logger.info("Ingested %s (%d chunks)", fp, len(chunks))

    try:
        from langchain_community.document_loaders import PyPDFLoader

        for fp in source_path.rglob("*.pdf"):
            loader = PyPDFLoader(str(fp))
            pages = loader.load_and_split(splitter)
            for p in pages:
                if not p.metadata.get("source"):
                    p.metadata["source"] = str(fp)
            vs.add_documents(pages)
            chunks_added += len(pages)
            logger.info("Ingested PDF %s (%d chunks)", fp, len(pages))
    except ImportError:
        logger.info("pypdf not installed, skipping PDF ingestion")

    return chunks_added
