from unittest.mock import MagicMock

from src.graph.state import AgentState, RouteDecision
from src.graph.workflow import _route_from_supervisor


def test_route_to_search_agent():
    state: AgentState = {
        "messages": [MagicMock(content="Routing to search_agent: find info about Python")],
    }
    result = _route_from_supervisor(state)
    assert result == "search_agent"


def test_route_to_code_agent():
    state: AgentState = {
        "messages": [MagicMock(content="Routing to code_agent: compute statistics")],
    }
    result = _route_from_supervisor(state)
    assert result == "code_agent"


def test_route_to_writer_agent():
    state: AgentState = {
        "messages": [MagicMock(content="Routing to writer_agent: write final report")],
    }
    result = _route_from_supervisor(state)
    assert result == "writer_agent"


def test_route_to_end():
    state: AgentState = {
        "messages": [MagicMock(content="All tasks completed.")],
    }
    from langgraph.graph import END
    result = _route_from_supervisor(state)
    assert result == END


def test_route_default_end():
    state: AgentState = {
        "messages": [MagicMock(content="Unknown message")],
    }
    from langgraph.graph import END
    result = _route_from_supervisor(state)
    assert result == END


def test_route_decision_model():
    decision = RouteDecision(
        agent_name="search_agent",
        reasoning="Need to find information",
        subtask_description="Search for Python asyncio patterns",
    )
    assert decision.agent_name == "search_agent"
    assert decision.reasoning == "Need to find information"
    assert decision.subtask_description == "Search for Python asyncio patterns"


def test_route_empty_messages():
    state: AgentState = {"messages": []}
    from langgraph.graph import END
    result = _route_from_supervisor(state)
    assert result == END


def test_route_multiple_messages_picks_last():

    state: AgentState = {
        "messages": [
            MagicMock(content="Routing to search_agent: first search"),
            MagicMock(content="Routing to code_agent: then code"),
        ]
    }
    result = _route_from_supervisor(state)
    assert result == "code_agent"
