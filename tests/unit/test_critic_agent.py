"""Unit tests for the Reflection/Critic agent (P1.8)."""
from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import AgentState


def _state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "task_description": "Explain RAG",
        "search_results": ["RAG retrieves docs"],
        "code_results": [],
        "final_report": "# Draft\nA short report.",
        "retry_count": {},
        "errors": [],
        "critic_feedback": "",
        "critic_rounds": 0,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _patch_verdict(score, passes, feedback=""):
    verdict = MagicMock()
    verdict.score = score
    verdict.passes = passes
    verdict.feedback = feedback

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = verdict
    mock_llm.with_structured_output.return_value = mock_structured
    return patch("src.agents.critic_agent.create_agent_llm", return_value=mock_llm)


@pytest.mark.asyncio
async def test_critic_accepts_passing_report():
    from src.agents.critic_agent import critic_agent_node

    with _patch_verdict(score=4.5, passes=True, feedback="LGTM"), patch(
        "src.agents.critic_agent.settings"
    ) as mock_settings:
        mock_settings.critic_max_rounds = 2
        result = await critic_agent_node(_state())

    assert result.goto == "__end__"
    assert result.update.get("report_ready") is True
    assert result.update.get("critic_rounds") == 1


@pytest.mark.asyncio
async def test_critic_rejects_and_routes_to_writer():
    from src.agents.critic_agent import critic_agent_node

    with _patch_verdict(
        score=2.0, passes=False, feedback="missing citations"
    ), patch("src.agents.critic_agent.settings") as mock_settings:
        mock_settings.critic_max_rounds = 3
        result = await critic_agent_node(_state(critic_rounds=0))

    assert result.goto == "writer_agent"
    assert result.update.get("critic_feedback") == "missing citations"
    assert result.update.get("report_ready") is None


@pytest.mark.asyncio
async def test_critic_forces_accept_after_max_rounds():
    """Even a failing report is accepted once max_rounds is reached."""
    from src.agents.critic_agent import critic_agent_node

    with _patch_verdict(score=2.0, passes=False, feedback="still bad"), patch(
        "src.agents.critic_agent.settings"
    ) as mock_settings:
        mock_settings.critic_max_rounds = 2
        # Already at round 1, this will be round 2 -> force-accept.
        result = await critic_agent_node(_state(critic_rounds=1))

    assert result.goto == "__end__"
    assert result.update.get("report_ready") is True


def test_build_revision_prompt_returns_empty_without_feedback():
    from src.agents.critic_agent import build_revision_prompt

    assert build_revision_prompt(_state(critic_feedback="")) == ""


def test_build_revision_prompt_includes_feedback():
    from src.agents.critic_agent import build_revision_prompt

    prompt = build_revision_prompt(_state(critic_feedback="add more detail"))
    assert "add more detail" in prompt
    assert "Critic Feedback" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
