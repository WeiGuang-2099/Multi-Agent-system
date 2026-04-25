import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.graph.state import AgentState
from src.llm.prompts import CODE_AGENT_SYSTEM_PROMPT
from src.llm.providers import create_agent_llm
from src.tools.code_executor import execute_python_code


class CodeAgent(BaseAgent):
    agent_name = "code_agent"

    def _run(self, state: AgentState) -> str:
        llm = create_agent_llm("code_agent")
        llm_with_tools = llm.bind_tools([execute_python_code])

        messages = [SystemMessage(content=CODE_AGENT_SYSTEM_PROMPT)]

        task_desc = state.get("task_description", "")
        last_msg = state["messages"][-1].content if state.get("messages") else ""

        search_context = ""
        if state.get("search_results"):
            search_context = (
                "\n\nPrevious search results for context:\n"
                + "\n".join(state["search_results"])
            )

        messages.append(
            HumanMessage(
                content=f"Task: {task_desc}\n\nSpecific instruction: {last_msg}"
                f"{search_context}\n\n"
                "Generate and execute Python code for this task. "
                "Use the execute_python_code tool."
            )
        )

        response = llm_with_tools.invoke(messages)

        return json.dumps(
            {
                "status": "completed",
                "output": response.content,
            },
            ensure_ascii=False,
        )


def code_agent_node(state: AgentState) -> Command:
    agent = CodeAgent()
    result = agent.execute(state)

    code_results = state.get("code_results", [])
    messages = result.update.get("messages", [])

    summary = ""
    if messages and hasattr(messages[0], "content"):
        summary = messages[0].content if isinstance(messages[0].content, str) else ""

    return Command(
        update={
            "messages": messages,
            "code_results": code_results + [summary] if summary else code_results,
            "retry_count": result.update.get("retry_count", state.get("retry_count", {})),
            "errors": result.update.get("errors", state.get("errors", [])),
        },
        goto=result.goto,
    )
