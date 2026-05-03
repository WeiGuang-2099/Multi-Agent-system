import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.graph.state import AgentState


def test_writer_agent_produces_report():
    from src.agents.writer_agent import WriterAgent

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# Research Report\n\n## Summary\nFindings here."
    mock_llm.invoke.return_value = mock_response

    with patch("src.agents.writer_agent.create_agent_llm", return_value=mock_llm):
        agent = WriterAgent()
        state: AgentState = {
            "messages": [],
            "task_description": "Research Python",
            "search_results": ["Search result 1"],
            "code_results": ["Code output 1"],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = agent._run(state)
        assert "Research Report" in result


@pytest.mark.asyncio
async def test_writer_agent_node_routes_to_end():
    from src.agents.writer_agent import writer_agent_node

    # Mock WriterAgent.execute to return a Command simulating successful execution
    with patch(
        "src.agents.writer_agent.WriterAgent.execute",
        new_callable=AsyncMock,
    ) as mock_execute:
        mock_execute.return_value = Command(
            update={
                "messages": [
                    AIMessage(content="# Final Report")
                ]
            },
            goto="supervisor",
        )

        state: AgentState = {
            "messages": [],
            "task_description": "Test task",
            "search_results": ["data"],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = await writer_agent_node(state)
        assert result.goto == "__end__"
        assert result.update.get("report_ready") is True
        assert "# Final Report" in result.update.get("final_report", "")
