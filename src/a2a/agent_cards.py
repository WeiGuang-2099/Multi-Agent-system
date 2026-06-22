from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.config import settings


def _base_url(port: int) -> str:
    return f"http://localhost:{port}"


SUPERVISOR_CARD = AgentCard(
    name="Research Supervisor",
    description="Orchestrates research tasks by decomposing them and routing to specialized agents",
    version="1.0.0",
    url=_base_url(settings.supervisor_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "text/markdown"],
    skills=[
        AgentSkill(
            id="orchestrate_research",
            name="Orchestrate Research",
            description="Decompose a research task and coordinate search, code, and writing agents",
            tags=["orchestration", "research"],
            examples=["Research Python asyncio performance patterns"],
        )
    ],
)

SEARCH_AGENT_CARD = AgentCard(
    name="Search Agent",
    description="Performs web searches using Tavily API to find relevant information",
    version="1.0.0",
    url=_base_url(settings.search_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="web_search",
            name="Web Search",
            description="Search the web for information on a given topic",
            tags=["search", "web", "research"],
            examples=["Search for Python asyncio best practices"],
        )
    ],
)

CODE_AGENT_CARD = AgentCard(
    name="Code Agent",
    description="Generates and executes Python code in a sandboxed Docker environment",
    version="1.0.0",
    url=_base_url(settings.code_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="code_execution",
            name="Code Execution",
            description="Generate and execute Python code for computation and data analysis",
            tags=["code", "python", "execution", "analysis"],
            examples=["Analyze sorting algorithm performance", "Compute statistical metrics"],
        )
    ],
)

WRITER_AGENT_CARD = AgentCard(
    name="Writer Agent",
    description="Synthesizes research results into structured Markdown reports",
    version="1.0.0",
    url=_base_url(settings.writer_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/markdown"],
    skills=[
        AgentSkill(
            id="report_writing",
            name="Report Writing",
            description="Write a structured Markdown research report from provided data",
            tags=["writing", "report", "markdown"],
            examples=["Write a report on machine learning trends"],
        )
    ],
)

ALL_CARDS = {
    "supervisor": SUPERVISOR_CARD,
    "search": SEARCH_AGENT_CARD,
    "code": CODE_AGENT_CARD,
    "writer": WRITER_AGENT_CARD,
}
