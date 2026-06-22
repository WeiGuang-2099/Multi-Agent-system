from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from src.agents.base import BaseAgent
from src.config import settings
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
        existing_report = state.get("final_report", "")
        critic_feedback = state.get("critic_feedback", "")

        context_parts = [f"Original task: {task_desc}\n"]

        if search_results:
            context_parts.append("## Search Results\n")
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"### Source {i}\n{result}\n")

        if code_results:
            context_parts.append("## Code Execution Results\n")
            for i, result in enumerate(code_results, 1):
                context_parts.append(f"### Result {i}\n{result}\n")

        # P1.8: if a previous draft exists, treat this as a revision pass.
        if existing_report and critic_feedback:
            context_parts.append(
                f"## Previous Draft (revise this)\n{existing_report}\n\n"
                f"## Critic Feedback (must address)\n{critic_feedback}\n"
            )
            instruction = (
                "Please REVISE the previous draft to address the critic's feedback. "
                "Output the full updated Markdown report."
            )
        elif existing_report:
            instruction = (
                "A draft already exists. Improve it into a final comprehensive "
                "Markdown research report."
            )
        else:
            instruction = (
                "Please write a comprehensive Markdown research report based on "
                "all the above information."
            )

        messages.append(
            HumanMessage(content="\n".join(context_parts) + "\n\n" + instruction)
        )

        response = llm.invoke(messages)
        return response.content


async def writer_agent_node(state: AgentState) -> Command:
    agent = WriterAgent()
    result = await agent.execute(state)

    if result.goto == "writer_agent":
        return result

    messages = result.update.get("messages", [])
    report = ""
    if messages and hasattr(messages[0], "content"):
        report = messages[0].content if isinstance(messages[0].content, str) else ""

    # P1.8: when the critic loop is active, the writer does NOT set report_ready
    # (the critic decides acceptance). Otherwise we end immediately.
    if getattr(settings, "critic_enabled", False):
        return Command(
            update={
                "messages": [AIMessage(content=report)],
                "final_report": report,
            },
            goto="critic_agent",
        )

    return Command(
        update={
            "messages": [AIMessage(content=report)],
            "final_report": report,
            "report_ready": True,
        },
        goto="__end__",
    )
