import logging

from langgraph.graph import END, START, StateGraph

from src.agents.code_agent import code_agent_node
from src.agents.search_agent import search_agent_node
from src.agents.writer_agent import writer_agent_node
from src.config import settings
from src.graph.state import AgentState
from src.graph.supervisor import route_after_worker, supervisor_node

logger = logging.getLogger(__name__)


def _build_graph_builder() -> StateGraph:
    """Create and configure the workflow graph builder with all nodes and edges.

    When `settings.critic_enabled` is True, a Reflection/Critic loop is added:
    writer -> critic -> (writer | end). Otherwise the graph keeps the original
    writer -> supervisor -> end structure.
    """
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

    if getattr(settings, "critic_enabled", False):
        # P1.8: insert the critic reflection loop.
        from src.agents.critic_agent import critic_agent_node

        builder.add_node("critic_agent", critic_agent_node)
        # writer hands off to critic (not supervisor) for self-review.
        builder.add_edge("writer_agent", "critic_agent")
        # critic_agent's Command(goto=...) decides writer_agent (revise) or __end__.
    else:
        builder.add_conditional_edges("writer_agent", route_after_worker)

    # P1.7: optional RAG retrieval agent (only registered when enabled).
    if getattr(settings, "rag_enabled", False):
        from src.agents.retrieval_agent import retrieval_agent_node

        builder.add_node("retrieval_agent", retrieval_agent_node)
        builder.add_conditional_edges("retrieval_agent", route_after_worker)

    return builder


async def build_workflow():
    """Build the async workflow with PostgreSQL checkpointing, falling back to MemorySaver."""
    builder = _build_graph_builder()

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpointer = AsyncPostgresSaver.from_conn_string(settings.postgres_uri)
        await checkpointer.setup()
        logger.info("Using PostgreSQL checkpointer")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}), falling back to MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def build_workflow_sync():
    """Build the sync workflow with PostgreSQL checkpointing, falling back to MemorySaver."""
    builder = _build_graph_builder()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver.from_conn_string(settings.postgres_uri)
        checkpointer.setup()
        logger.info("Using PostgreSQL checkpointer (sync)")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}), falling back to MemorySaver (sync)")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def _route_from_supervisor(state: AgentState) -> str:
    """Fallback routing based on the supervisor's last routing message.
    In practice, Command.goto from supervisor_node handles routing directly.
    This serves as a compatible fallback if Command.goto is not honored."""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if isinstance(content, str):
            for agent in [
                "search_agent",
                "code_agent",
                "writer_agent",
                "retrieval_agent",
            ]:
                if f"Routing to {agent}" in content:
                    return agent
            if "All tasks completed" in content:
                return END
    return END
