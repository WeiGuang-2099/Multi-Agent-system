from langgraph.graph import END, START, StateGraph

from src.agents.code_agent import code_agent_node
from src.agents.search_agent import search_agent_node
from src.agents.writer_agent import writer_agent_node
from src.config import settings
from src.graph.state import AgentState
from src.graph.supervisor import route_after_worker, supervisor_node


async def build_workflow() -> StateGraph:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search_agent", search_agent_node)
    builder.add_node("code_agent", code_agent_node)
    builder.add_node("writer_agent", writer_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        ["search_agent", "code_agent", "writer_agent", "__end__"],
    )

    builder.add_conditional_edges("search_agent", route_after_worker)
    builder.add_conditional_edges("code_agent", route_after_worker)
    builder.add_conditional_edges("writer_agent", route_after_worker)

    checkpointer = AsyncPostgresSaver.from_conn_string(settings.postgres_uri)
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


def build_workflow_sync() -> StateGraph:
    from langgraph.checkpoint.postgres import PostgresSaver

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search_agent", search_agent_node)
    builder.add_node("code_agent", code_agent_node)
    builder.add_node("writer_agent", writer_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        ["search_agent", "code_agent", "writer_agent", "__end__"],
    )

    builder.add_conditional_edges("search_agent", route_after_worker)
    builder.add_conditional_edges("code_agent", route_after_worker)
    builder.add_conditional_edges("writer_agent", route_after_worker)

    checkpointer = PostgresSaver.from_conn_string(settings.postgres_uri)
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


def _route_from_supervisor(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        if isinstance(content, str):
            for agent in ["search_agent", "code_agent", "writer_agent"]:
                if f"Routing to {agent}" in content:
                    return agent
            if "All tasks completed" in content:
                return END
    return END
