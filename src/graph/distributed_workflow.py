import json
import logging

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.a2a.client import A2AClient
from src.graph.state import AgentState
from src.graph.supervisor import route_after_worker, supervisor_node

logger = logging.getLogger(__name__)

_client = A2AClient()


def _make_remote_agent_node(agent_name: str):
    async def remote_node(state: AgentState) -> Command:
        task_desc = state.get("task_description", "")
        last_msg = state["messages"][-1].content if state.get("messages") else ""

        context = json.dumps({
            "task_description": task_desc,
            "search_results": state.get("search_results", []),
            "code_results": state.get("code_results", []),
        }, ensure_ascii=False)

        message = f"{last_msg}\n\nContext: {context}" if last_msg else context

        try:
            result_text = await _client.send_task(agent_name, message)
        except Exception as e:
            logger.error(f"A2A call to {agent_name} failed: {e}")
            errors = state.get("errors", []).copy()
            errors.append(f"{agent_name} remote call failed: {str(e)}")
            return Command(
                update={"errors": errors},
                goto="supervisor",
            )

        update = {
            "messages": [AIMessage(content=f"[{agent_name}] {result_text}")],
        }

        if agent_name == "search_agent":
            search_results = state.get("search_results", [])
            update["search_results"] = search_results + [result_text]
        elif agent_name == "code_agent":
            code_results = state.get("code_results", [])
            update["code_results"] = code_results + [result_text]
        elif agent_name == "writer_agent":
            update["final_report"] = result_text
            update["report_ready"] = True
            return Command(update=update, goto="__end__")

        return Command(update=update, goto="supervisor")

    return remote_node


async def build_distributed_workflow():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search_agent", _make_remote_agent_node("search_agent"))
    builder.add_node("code_agent", _make_remote_agent_node("code_agent"))
    builder.add_node("writer_agent", _make_remote_agent_node("writer_agent"))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor_dist,
        ["search_agent", "code_agent", "writer_agent", "__end__"],
    )
    builder.add_conditional_edges("search_agent", route_after_worker)
    builder.add_conditional_edges("code_agent", route_after_worker)
    builder.add_conditional_edges("writer_agent", route_after_worker)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _route_from_supervisor_dist(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if isinstance(content, str):
            for agent in ["search_agent", "code_agent", "writer_agent"]:
                if f"Routing to {agent}" in content:
                    return agent
            if "All tasks completed" in content:
                return END
    return END
