import pytest
from unittest.mock import AsyncMock, patch


def test_interactive_mode_exists():
    from src.main import _run_interactive
    assert callable(_run_interactive)


@pytest.mark.asyncio
async def test_execute_task_returns_report():
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "# Test Report\nContent here",
        "errors": [],
    })

    with patch("src.graph.workflow.build_workflow", AsyncMock(return_value=mock_graph)):
        from src.main import _execute_task

        result = await _execute_task("test task")
        assert result == "# Test Report\nContent here"


@pytest.mark.asyncio
async def test_execute_task_handles_no_report():
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "",
        "errors": ["something failed"],
    })

    with patch("src.graph.workflow.build_workflow", AsyncMock(return_value=mock_graph)):
        from src.main import _execute_task

        result = await _execute_task("test task")
        assert result is None
