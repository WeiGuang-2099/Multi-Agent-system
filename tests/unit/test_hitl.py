"""Unit tests for the Human-in-the-Loop (HITL) supervisor logic (P0.2).

These tests verify that:
- With hitl_enabled=False, the supervisor behaves exactly as before.
- With hitl_enabled=True and the user approves, the original routing decision
  is preserved and recorded in `approved_plans`.
- With hitl_enabled=True and the user rejects, the workflow terminates.
- With hitl_enabled=True and the user edits the plan, the edited decision wins.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import AgentState


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "task_description": "Research Python asyncio",
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _mock_router_decision(agent_name, reasoning="r", subtask="s"):
    """Patch create_llm so the router returns a fixed RouteDecision-like obj."""
    decision = MagicMock()
    decision.agent_name = agent_name
    decision.reasoning = reasoning
    decision.subtask_description = subtask

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = decision
    mock_llm.with_structured_output.return_value = mock_router
    return patch("src.graph.supervisor.create_llm", return_value=mock_llm)


def test_hitl_disabled_preserves_original_behavior():
    """Default behavior: no interrupt, routes directly."""
    from src.graph.supervisor import supervisor_node

    with _mock_router_decision("search_agent"), patch(
        "src.config.settings.hitl_enabled", False
    ):
        result = supervisor_node(_make_state())
        assert result.goto == "search_agent"


def test_hitl_approved_keeps_routing_and_records_plan():
    """User approves -> same routing + plan recorded in approved_plans."""
    from src.graph.supervisor import supervisor_node

    with _mock_router_decision("code_agent"), patch(
        "src.config.settings.hitl_enabled", True
    ), patch("src.graph.supervisor.interrupt", return_value={"approved": True}):
        result = supervisor_node(_make_state())
        assert result.goto == "code_agent"
        # The plan should be recorded.
        approved = result.update.get("approved_plans", [])
        assert any(p["agent_name"] == "code_agent" for p in approved)


def test_hitl_rejected_terminates_workflow():
    """User rejects -> workflow ends."""
    from src.graph.supervisor import supervisor_node

    with _mock_router_decision("search_agent"), patch(
        "src.config.settings.hitl_enabled", True
    ), patch("src.graph.supervisor.interrupt", return_value={"approved": False}):
        result = supervisor_node(_make_state())
        assert result.goto == "__end__"


def test_hitl_edit_changes_routing():
    """User edits the plan -> edited agent_name is used."""
    from src.graph.supervisor import supervisor_node

    edit = {
        "approved": True,
        "agent_name": "writer_agent",
        "reasoning": "skip to writing",
        "subtask_description": "write now",
    }
    with _mock_router_decision("search_agent"), patch(
        "src.config.settings.hitl_enabled", True
    ), patch("src.graph.supervisor.interrupt", return_value=edit):
        result = supervisor_node(_make_state())
        assert result.goto == "writer_agent"
        msg = result.update["messages"][0].content
        assert "Routing to writer_agent" in msg


def test_hitl_finish_decision_skips_interrupt():
    """FINISH decisions should not trigger interrupt even when HITL is on."""
    from src.graph.supervisor import supervisor_node

    interrupt_mock = MagicMock(side_effect=AssertionError("should not be called"))
    with _mock_router_decision("FINISH"), patch(
        "src.config.settings.hitl_enabled", True
    ), patch("src.graph.supervisor.interrupt", interrupt_mock):
        result = supervisor_node(_make_state())
        assert result.goto == "__end__"
        interrupt_mock.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
