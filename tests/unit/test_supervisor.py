import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import AgentState


def test_supervisor_routes_to_search():
    from src.graph.supervisor import supervisor_node

    mock_decision = MagicMock()
    mock_decision.agent_name = "search_agent"
    mock_decision.reasoning = "Need info"
    mock_decision.subtask_description = "Search for patterns"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Research Python asyncio",
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "search_agent"


def test_supervisor_routes_to_code():
    from src.graph.supervisor import supervisor_node

    mock_decision = MagicMock()
    mock_decision.agent_name = "code_agent"
    mock_decision.reasoning = "Need computation"
    mock_decision.subtask_description = "Run analysis"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Analyze data",
            "search_results": ["some results"],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "code_agent"


def test_supervisor_finishes():
    from src.graph.supervisor import supervisor_node

    mock_decision = MagicMock()
    mock_decision.agent_name = "FINISH"
    mock_decision.reasoning = "All done"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Done task",
            "search_results": ["results"],
            "code_results": ["output"],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "__end__"


def test_route_after_worker_returns_supervisor():
    from src.graph.supervisor import route_after_worker

    state: AgentState = {
        "messages": [],
        "task_description": "test",
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
        "report_ready": False,
    }
    assert route_after_worker(state) == "supervisor"


def test_route_after_worker_ends_when_report_ready():
    from src.graph.supervisor import route_after_worker

    state: AgentState = {
        "messages": [],
        "task_description": "test",
        "search_results": [],
        "code_results": [],
        "final_report": "# Report",
        "retry_count": {},
        "errors": [],
        "report_ready": True,
    }
    assert route_after_worker(state) == "__end__"
