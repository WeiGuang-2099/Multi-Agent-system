"""Structured JSON logging configuration (P0.4).

We emit one JSON object per log line so logs can be shipped to any aggregator
(ELK, Loki, Datadog, CloudWatch) without re-parsing. Falls back to a readable
console format when `LOG_JSON=false` is set (handy for local development).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render every LogRecord as a single-line JSON object."""

    _RESERVED = {"message", "asctime", "levelname", "name", "created", "msecs"}
    _STD_KEYS = {
        "args", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "filename", "module", "pathname", "process", "processName",
        "thread", "threadName", "relativeCreated", "levelno", "msg",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in self._RESERVED or key in self._STD_KEYS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def _ensure_log_dir() -> Path:
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logger with JSON stdout + file handlers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = level or os.environ.get("LOG_LEVEL", "INFO")
    use_json = os.environ.get("LOG_JSON", "true").lower() != "false"

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    if use_json:
        stream_handler.setFormatter(JsonFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
    root.addHandler(stream_handler)

    try:
        log_dir = _ensure_log_dir()
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError:
        root.warning(
            "Could not open logs/app.log for writing; file logging disabled"
        )

    _CONFIGURED = True
