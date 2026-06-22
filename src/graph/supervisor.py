"""Supervisor node: analyzes task and routes to the next agent.

P0.2 adds optional Human-in-the-Loop (HITL) approval. When
`settings.hitl_enabled` is True, the supervisor pauses (via LangGraph's
`interrupt()`) after producing a routing plan, exposing the plan to the UI
for user approval. The user can approve (resume with the same decision) or
reject (resume with a FINISH override).

The HITL path is fully opt-in; with `hitl_enabled=False` (default), behavior
is identical to the original supervisor and existing tests remain valid.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

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

    decision: RouteDecision = router.invoke(messages)

    # --- P0.2 Human-in-the-Loop approval ---------------------------------
    if settings.hitl_enabled and decision.agent_name != "FINISH":
        plan = {
            "agent_name": decision.agent_name,
            "reasoning": decision.reasoning,
            "subtask_description": decision.subtask_description,
        }

        # `interrupt()` suspends the graph here and surfaces `plan` to the
        # caller (UI). When resumed, its return value is the user's response.
        # Expected shapes:
        #   {"approved": True}            -> proceed with the plan
        #   {"approved": False}           -> override to FINISH
        #   {"agent_name": "...", ...}    -> edited plan
        user_response = interrupt(plan) or {}

        if isinstance(user_response, dict):
            if not user_response.get("approved", True):
                # User rejected: terminate the workflow gracefully.
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content="All tasks completed (rejected by user)."
                            )
                        ],
                        "pending_plan": None,
                    },
                    goto="__end__",
                )

            # Allow the user to edit the routing decision.
            edited_agent = user_response.get("agent_name")
            if edited_agent in {
                "search_agent",
                "code_agent",
                "writer_agent",
                "retrieval_agent",
                "FINISH",
            }:
                decision = RouteDecision(
                    agent_name=edited_agent,
                    reasoning=user_response.get("reasoning", decision.reasoning),
                    subtask_description=user_response.get(
                        "subtask_description", decision.subtask_description
                    ),
                )

        # Record the approved plan for auditability.
        approved = list(state.get("approved_plans", []))
        approved.append(
            {
                "agent_name": decision.agent_name,
                "reasoning": decision.reasoning,
                "subtask_description": decision.subtask_description,
            }
        )
        state_update = {"pending_plan": None, "approved_plans": approved}
    else:
        state_update = {}

    # --- Routing ----------------------------------------------------------
    if decision.agent_name == "FINISH":
        return Command(
            update={
                **state_update,
                "messages": [HumanMessage(content="All tasks completed.")],
            },
            goto="__end__",
        )

    return Command(
        update={
            **state_update,
            "messages": [
                HumanMessage(
                    content=f"Routing to {decision.agent_name}: {decision.subtask_description}"
                )
            ],
        },
        goto=decision.agent_name,
    )


def route_after_worker(state: AgentState) -> str:
    """Route after a worker agent completes. Check if report is ready."""
    if state.get("report_ready"):
        return "__end__"
    return "supervisor"
