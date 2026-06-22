from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_workflow_builds():
    pytest.importorskip("langgraph.checkpoint.postgres")

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    with patch.object(AsyncPostgresSaver, "from_conn_string") as mock_from_conn:
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()
        mock_from_conn.return_value = mock_checkpointer

        with (
            patch("src.graph.workflow.search_agent_node", MagicMock()),
            patch("src.graph.workflow.code_agent_node", MagicMock()),
            patch("src.graph.workflow.writer_agent_node", MagicMock()),
            patch("src.graph.workflow.supervisor_node", MagicMock()),
        ):
            from src.graph.workflow import build_workflow

            graph = await build_workflow()
            assert graph is not None


@pytest.mark.asyncio
async def test_agent_state_structure():
    from src.graph.state import AgentState

    state: AgentState = {
        "messages": [],
        "task_description": "Research Python asyncio",
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
    }

    assert state["task_description"] == "Research Python asyncio"
