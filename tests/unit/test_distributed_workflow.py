import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_distributed_workflow_builds():
    with patch("src.graph.distributed_workflow.A2AClient") as mock_cls:
        mock_cls.return_value = AsyncMock()
        from src.graph.distributed_workflow import build_distributed_workflow

        graph = await build_distributed_workflow()
        assert graph is not None


@pytest.mark.asyncio
async def test_distributed_search_node_calls_client():
    mock_client = AsyncMock()
    mock_client.send_task = AsyncMock(return_value="search results here")

    with patch("src.graph.distributed_workflow._client", mock_client):
        from src.graph.distributed_workflow import _make_remote_agent_node

        node = _make_remote_agent_node("search_agent")
        state = {
            "messages": [],
            "task_description": "test task",
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = await node(state)
        assert mock_client.send_task.called
