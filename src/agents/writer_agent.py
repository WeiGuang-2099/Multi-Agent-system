from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.graph.state import AgentState
from src.llm.prompts import WRITER_AGENT_SYSTEM_PROMPT
from src.llm.providers import create_agent_llm


class WriterAgent(BaseAgent):
    agent_name = "writer_agent"

    def _run(self, state: AgentState) -> str:
        llm = create_agent_llm("writer_agent")

        messages = [SystemMessage(content=WRITER_AGENT_SYSTEM_PROMPT)]

        task_desc = state.get("task_description", "")
        search_results = state.get("search_results", [])
        code_results = state.get("code_results", [])

        context_parts = [f"Original task: {task_desc}\n"]

        if search_results:
            context_parts.append("## Search Results\n")
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"### Source {i}\n{result}\n")

        if code_results:
            context_parts.append("## Code Execution Results\n")
            for i, result in enumerate(code_results, 1):
                context_parts.append(f"### Result {i}\n{result}\n")

        messages.append(
            HumanMessage(
                content="\n".join(context_parts)
                + "\n\nPlease write a comprehensive Markdown research report based on all the above information."
            )
        )

        response = llm.invoke(messages)
        return response.content


def writer_agent_node(state: AgentState) -> Command:
    agent = WriterAgent()
    result = agent.execute(state)

    if result.goto == "writer_agent":
        return result

    messages = result.update.get("messages", [])
    report = ""
    if messages and hasattr(messages[0], "content"):
        report = messages[0].content if isinstance(messages[0].content, str) else ""

    return Command(
        update={
            "messages": [
                AIMessage(content="FINAL_REPORT_READY"),
                AIMessage(content=report),
            ],
            "final_report": report,
        },
        goto=result.goto,
    )
