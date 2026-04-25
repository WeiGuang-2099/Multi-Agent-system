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
    """Create and configure the workflow graph builder with all nodes and edges."""
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

    return builder


async def build_workflow():
    """Build the async workflow with PostgreSQL checkpointing."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    builder = _build_graph_builder()

    try:
        checkpointer = AsyncPostgresSaver.from_conn_string(settings.postgres_uri)
        await checkpointer.setup()
    except Exception as e:
        logger.error(f"Failed to setup async PostgreSQL checkpointer: {e}")
        raise ConnectionError(f"Cannot connect to PostgreSQL: {e}") from e

    return builder.compile(checkpointer=checkpointer)


def build_workflow_sync():
    """Build the sync workflow with PostgreSQL checkpointing."""
    from langgraph.checkpoint.postgres import PostgresSaver

    builder = _build_graph_builder()

    try:
        checkpointer = PostgresSaver.from_conn_string(settings.postgres_uri)
        checkpointer.setup()
    except Exception as e:
        logger.error(f"Failed to setup sync PostgreSQL checkpointer: {e}")
        raise ConnectionError(f"Cannot connect to PostgreSQL: {e}") from e

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
            for agent in ["search_agent", "code_agent", "writer_agent"]:
                if f"Routing to {agent}" in content:
                    return agent
            if "All tasks completed" in content:
                return END
    return END
