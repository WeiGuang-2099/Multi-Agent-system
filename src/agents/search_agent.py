import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.graph.state import AgentState
from src.llm.prompts import SEARCH_AGENT_SYSTEM_PROMPT
from src.llm.providers import create_agent_llm
from src.tools.search import tavily_search


class SearchAgent(BaseAgent):
    agent_name = "search_agent"

    def _run(self, state: AgentState) -> str:
        llm = create_agent_llm("search_agent")
        llm_with_tools = llm.bind_tools([tavily_search])

        messages = [SystemMessage(content=SEARCH_AGENT_SYSTEM_PROMPT)]

        task_desc = state.get("task_description", "")
        last_msg = state["messages"][-1].content if state.get("messages") else ""

        messages.append(
            HumanMessage(
                content=f"Task: {task_desc}\n\nSpecific instruction: {last_msg}\n\n"
                "Please search for relevant information. Use the tavily_search tool."
            )
        )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        max_iterations = 5
        iteration = 0
        while response.tool_calls and iteration < max_iterations:
            for tool_call in response.tool_calls:
                tool_result = tavily_search.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            iteration += 1

        results_count = len(state.get("search_results", [])) + 1
        return json.dumps(
            {
                "status": "completed",
                "summary": response.content,
                "results_count": results_count,
            },
            ensure_ascii=False,
        )


async def search_agent_node(state: AgentState) -> Command:
    agent = SearchAgent()
    result = await agent.execute(state)

    if result.goto == "search_agent":
        return result

    search_results = state.get("search_results", [])

    return Command(
        update={
            "messages": result.update.get("messages", []),
            "search_results": search_results + [_extract_summary(result)],
            "retry_count": result.update.get("retry_count", state.get("retry_count", {})),
            "errors": result.update.get("errors", state.get("errors", [])),
        },
        goto=result.goto,
    )


def _extract_summary(command: Command) -> str:
    messages = command.update.get("messages", [])
    if messages:
        content = messages[0].content
        if isinstance(content, str):
            return content
    return ""
