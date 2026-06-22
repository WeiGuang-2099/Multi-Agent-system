"""Pytest configuration shared across the whole test suite.

Sets dummy API keys so tests that construct LLMs (via mocked classes) can run
in CI without real credentials. Real key validation is exercised by the
`validate_required_keys` unit test instead.
"""
from __future__ import annotations

import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    """Provide fake API keys for every test; real keys are not needed for unit tests."""
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-anthropic-key")
    monkeypatch.setattr(settings, "openai_api_key", "fake-openai-key")
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "tavily_api_key", "fake-tavily-key")
    yield
