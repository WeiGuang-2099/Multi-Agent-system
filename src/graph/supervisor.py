from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.config import settings
from src.graph.state import AgentState, RouteDecision
from src.llm.prompts import SUPERVISOR_SYSTEM_PROMPT
from src.llm.providers import create_llm


def supervisor_node(state: AgentState) -> Command:
    llm = create_llm()
    router = llm.with_structured_output(RouteDecision)

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

    if state.get("messages"):
        state_messages = state["messages"][-20:]  # Message windowing: keep last 20
        messages.extend(state_messages)

    context_parts = []
    if state.get("task_description"):
        context_parts.append(f"Original task: {state['task_description']}")
    if state.get("search_results"):
        context_parts.append(
            f"Search results collected: {len(state['search_results'])} items"
        )
    if state.get("code_results"):
        context_parts.append(
            f"Code results collected: {len(state['code_results'])} items"
        )
    if state.get("errors"):
        context_parts.append(f"Errors encountered: {state['errors']}")
    if context_parts:
        messages.append(
            HumanMessage(content="Current state:\n" + "\n".join(context_parts))
        )

    decision = router.invoke(messages)

    if decision.agent_name == "FINISH":
        return Command(
            update={"messages": [HumanMessage(content="All tasks completed.")]},
            goto="__end__",
        )

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=f"Routing to {decision.agent_name}: {decision.subtask_description}"
                )
            ]
        },
        goto=decision.agent_name,
    )


def route_after_worker(state: AgentState) -> str:
    """Route after a worker agent completes. Check if report is ready."""
    if state.get("report_ready"):
        return "__end__"
    return "supervisor"
