import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver


def _make_mock_checkpointer():
    """Create a mock checkpointer that is a valid BaseCheckpointSaver instance."""
    mock = MagicMock(spec=BaseCheckpointSaver)
    mock.setup = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_build_workflow_falls_back_to_memory():
    """When PostgreSQL is unavailable, workflow falls back to MemorySaver."""
    # Inject a broken postgres module so the import triggers the fallback.
    mock_pg_aio = types.ModuleType("langgraph.checkpoint.postgres.aio")
    mock_pg_aio.AsyncPostgresSaver = MagicMock()
    mock_pg_aio.AsyncPostgresSaver.from_conn_string = MagicMock(
        side_effect=Exception("PG down")
    )

    # Remove any cached imports of the workflow module so it re-imports.
    sys.modules.pop("src.graph.workflow", None)
    sys.modules["langgraph.checkpoint.postgres.aio"] = mock_pg_aio

    try:
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None
    finally:
        sys.modules.pop("langgraph.checkpoint.postgres.aio", None)
        sys.modules.pop("src.graph.workflow", None)


@pytest.mark.asyncio
async def test_build_workflow_uses_pg_when_available():
    """When PostgreSQL is available, workflow uses AsyncPostgresSaver."""
    mock_checkpointer = _make_mock_checkpointer()

    mock_pg_aio = types.ModuleType("langgraph.checkpoint.postgres.aio")
    mock_pg_aio.AsyncPostgresSaver = MagicMock()
    mock_pg_aio.AsyncPostgresSaver.from_conn_string = MagicMock(
        return_value=mock_checkpointer
    )

    sys.modules.pop("src.graph.workflow", None)
    sys.modules["langgraph.checkpoint.postgres.aio"] = mock_pg_aio

    try:
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None
    finally:
        sys.modules.pop("langgraph.checkpoint.postgres.aio", None)
        sys.modules.pop("src.graph.workflow", None)


def test_build_workflow_sync_falls_back_to_memory():
    """Sync variant also falls back gracefully."""
    mock_pg = types.ModuleType("langgraph.checkpoint.postgres")
    mock_pg.PostgresSaver = MagicMock()
    mock_pg.PostgresSaver.from_conn_string = MagicMock(
        side_effect=Exception("PG down")
    )

    sys.modules.pop("src.graph.workflow", None)
    sys.modules["langgraph.checkpoint.postgres"] = mock_pg

    try:
        from src.graph.workflow import build_workflow_sync

        graph = build_workflow_sync()
        assert graph is not None
    finally:
        sys.modules.pop("langgraph.checkpoint.postgres", None)
        sys.modules.pop("src.graph.workflow", None)
