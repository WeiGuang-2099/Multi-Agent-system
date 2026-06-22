"""Retrieval agent (P1.7).

Performs semantic search over the local knowledge base (Chroma) using the
`tavily_rag_search` tool. Designed to complement the web Search Agent: when a
task references materials that may live in `data/`, the supervisor can route
here to ground the answer in local documents (PDFs, notes, papers).

Only active when `settings.rag_enabled` is True; otherwise the supervisor
never routes to this node.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.graph.state import AgentState
from src.llm.prompts import SEARCH_AGENT_SYSTEM_PROMPT
from src.llm.providers import create_agent_llm
from src.tools.rag import tavily_rag_search

RETRIEVAL_AGENT_SYSTEM_PROMPT = (
    SEARCH_AGENT_SYSTEM_PROMPT
    + "\n\nYou are specifically retrieving from the LOCAL knowledge base "
    "(ingested PDFs/notes/papers), not the open web. Prefer `tavily_rag_search`."
)


class RetrievalAgent(BaseAgent):
    agent_name = "retrieval_agent"

    def _run(self, state: AgentState) -> str:
        llm = create_agent_llm("retrieval_agent")
        llm_with_tools = llm.bind_tools([tavily_rag_search])

        messages = [SystemMessage(content=RETRIEVAL_AGENT_SYSTEM_PROMPT)]

        task_desc = state.get("task_description", "")
        last_msg = state["messages"][-1].content if state.get("messages") else ""

        messages.append(
            HumanMessage(
                content=(
                    f"Task: {task_desc}\n\nSpecific instruction: {last_msg}\n\n"
                    "Retrieve relevant passages from the local knowledge base. "
                    "Use the tavily_rag_search tool."
                )
            )
        )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        max_iterations = 5
        iteration = 0
        while response.tool_calls and iteration < max_iterations:
            for tool_call in response.tool_calls:
                tool_result = tavily_rag_search.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=str(tool_result), tool_call_id=tool_call["id"]
                    )
                )
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            iteration += 1

        return json.dumps(
            {"status": "completed", "summary": response.content},
            ensure_ascii=False,
        )


async def retrieval_agent_node(state: AgentState) -> Command:
    """Retrieve from local KB and accumulate results into search_results."""
    agent = RetrievalAgent()
    result = await agent.execute(state)

    if result.goto == "retrieval_agent":
        return result

    messages = result.update.get("messages", [])
    summary = ""
    if messages and hasattr(messages[0], "content"):
        summary = (
            messages[0].content if isinstance(messages[0].content, str) else ""
        )

    search_results = state.get("search_results", [])
    return Command(
        update={
            "messages": messages,
            "search_results": search_results + [summary] if summary else search_results,
            "retry_count": result.update.get(
                "retry_count", state.get("retry_count", {})
            ),
            "errors": result.update.get("errors", state.get("errors", [])),
        },
        goto=result.goto,
    )
