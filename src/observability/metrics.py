"""Token usage & cost tracking (P0.4).

- `CostTracker` — thread-safe accumulator that records per-model token usage
  and estimates USD cost, persisting each record as a JSONL line in
  `logs/metrics.jsonl`.

- `UsageEventHandler` — a LangChain `BaseCallbackHandler` that LLMs invoke
  automatically on every `on_llm_end`; it pulls `usage_metadata` from the
  response and feeds it to a `CostTracker`.

Deliberately dependency-free (no Prometheus) to keep the project portable;
the README documents the one-line swap to a Prometheus Counter.

Per-1M-token pricing is a rough public snapshot (2025-Q2). Override per model
via env vars `PRICE_INPUT_<MODEL_UPPER>` / `PRICE_OUTPUT_<MODEL_UPPER>`.
"""
from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # model_lower_prefix -> (input_per_1m, output_per_1m)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5": (0.50, 1.50),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-haiku": (0.25, 1.25),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-1.5": (1.25, 5.00),
}


def _price_for(model: str) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m), with env override."""
    suffix = model.upper().replace("-", "_")
    env_in = os.environ.get(f"PRICE_INPUT_{suffix}")
    env_out = os.environ.get(f"PRICE_OUTPUT_{suffix}")
    if env_in and env_out:
        return float(env_in), float(env_out)

    low = model.lower()
    for prefix, price in DEFAULT_PRICING.items():
        if low.startswith(prefix):
            return price
    return (0.0, 0.0)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CostTracker:
    """Thread-safe token + cost accumulator with JSONL persistence."""

    def __init__(self, log_path: str | Path | None = None):
        self._lock = threading.Lock()
        self._log_path = Path(
            log_path or os.environ.get("METRICS_LOG", "logs/metrics.jsonl")
        )
        self._totals: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "cost_usd": 0.0,
                "calls": 0.0,
            }
        )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        in_price, out_price = _price_for(model)
        cost = (input_tokens / 1_000_000) * in_price + (
            output_tokens / 1_000_000
        ) * out_price

        entry = {
            "ts": _now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        }

        with self._lock:
            bucket = self._totals[model]
            bucket["input_tokens"] += input_tokens
            bucket["output_tokens"] += output_tokens
            bucket["cost_usd"] += cost
            bucket["calls"] += 1

        self._append(entry)

    def _append(self, entry: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break the app

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {k: dict(v) for k, v in self._totals.items()}

    def total_cost(self) -> float:
        with self._lock:
            return sum(v["cost_usd"] for v in self._totals.values())

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()


_global_tracker = CostTracker()


def get_global_tracker() -> CostTracker:
    return _global_tracker


class UsageEventHandler(BaseCallbackHandler):
    """LangChain callback that feeds usage data into a CostTracker."""

    def __init__(self, tracker: CostTracker | None = None):
        self.tracker = tracker or _global_tracker
        self._model_by_run: dict[str, str] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        inv = kwargs.get("invocation_params") or {}
        model = str(
            inv.get("model")
            or inv.get("model_name")
            or inv.get("model_id")
            or serialized.get("name", "")
        )
        if run_id is not None:
            self._model_by_run[str(run_id)] = model

    def on_llm_end(self, response, *, run_id: Any = None, **kwargs: Any) -> None:
        run_key = str(run_id) if run_id is not None else ""
        fallback = (
            getattr(response, "llm_output", {}).get("model_name", "unknown")
            if getattr(response, "llm_output", None)
            else "unknown"
        )
        model = self._model_by_run.pop(run_key, "") or fallback

        usage = _extract_usage(response)
        if usage is None:
            return

        self.tracker.record(
            model=model,
            input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
        )


def _extract_usage(response) -> dict[str, int] | None:
    """Pull a usage dict out of various LangChain LLMResult/generations shapes."""
    try:
        generations = getattr(response, "generations", [])
        for batch in generations:
            for gen in batch:
                msg = getattr(gen, "message", None) or gen
                um = getattr(msg, "usage_metadata", None)
                if isinstance(um, dict):
                    return um
    except Exception:  # noqa: BLE001
        pass

    try:
        llm_output = getattr(response, "llm_output", {}) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict):
            return token_usage
    except Exception:  # noqa: BLE001
        pass

    return None
