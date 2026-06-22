"""Observability utilities for the Multi-Agent Research Assistant (P0.4).

This package provides three complementary observability primitives:

1. `logging_config` — configures structured JSON logging across all modules,
   so logs can be ingested by any log aggregator (ELK, Loki, Datadog).

2. `metrics` — a lightweight token-usage & cost counter that hooks into
   LangChain callbacks and appends JSONL records to `logs/metrics.jsonl`.
   Deliberately dependency-free (no Prometheus) to keep the project portable;
   the README documents how to extend it to Prometheus + Grafana.

3. `tracing` — initializes LangSmith tracing from settings, so every graph run
   appears in the LangSmith UI with full node/tool/LLM spans.

Usage:

    from src.observability import setup_observability
    setup_observability()  # call once at startup (main.py / streamlit_app.py)
"""
from src.observability.logging_config import setup_logging
from src.observability.metrics import CostTracker, UsageEventHandler
from src.observability.tracing import init_tracing


def setup_observability() -> None:
    """Configure logging, metrics, and tracing. Safe to call multiple times."""
    setup_logging()
    init_tracing()


__all__ = [
    "setup_observability",
    "setup_logging",
    "CostTracker",
    "UsageEventHandler",
    "init_tracing",
]
