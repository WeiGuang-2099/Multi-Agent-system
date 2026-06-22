"""LangSmith tracing initialization (P0.4).

LangSmith gives a hosted, per-node view of every graph run: which node ran,
what the LLM was called with, how long each step took, and where errors
occurred. This module configures it from `settings` and is a no-op when the
API key is absent (so local dev without LangSmith still works).
"""
from __future__ import annotations

import os

from src.config import settings


def init_tracing() -> bool:
    """Enable LangSmith tracing if credentials are present.

    Returns True if tracing is active, False otherwise. Safe to call multiple
    times — LangSmith reads these env vars lazily, so setting them here before
    any LLM call is sufficient.
    """
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault(
            "LANGCHAIN_API_KEY", settings.langsmith_api_key
        )  # backwards-compat alias
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        return True

    # Explicitly disable to avoid partial-config warnings in some SDK versions.
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    return False
