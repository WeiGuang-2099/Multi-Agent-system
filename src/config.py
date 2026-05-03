import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # LLM
    default_llm_provider: str = "anthropic"
    default_llm_model: str = "claude-sonnet-4-20250514"
    search_agent_model: str | None = None
    code_agent_model: str | None = None
    writer_agent_model: str | None = None

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "research_assistant"
    postgres_user: str = "agent"
    postgres_password: str = ""

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "multi-agent-research-assistant"

    # A2A Ports
    supervisor_port: int = 8001
    search_agent_port: int = 8002
    code_agent_port: int = 8003
    writer_agent_port: int = 8004
    streamlit_port: int = 8501

    supervisor_url: str = "http://localhost:8001"

    # Sandbox
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: str = "1"
    sandbox_timeout: int = 30

    # Retry
    max_retries: int = 3

    @property
    def postgres_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def agent_urls(self) -> dict[str, str]:
        return {
            "search_agent": f"http://localhost:{self.search_agent_port}",
            "code_agent": f"http://localhost:{self.code_agent_port}",
            "writer_agent": f"http://localhost:{self.writer_agent_port}",
            "supervisor": f"http://localhost:{self.supervisor_port}",
        }

    def get_model(self, agent_name: str) -> str:
        override = getattr(self, f"{agent_name}_model", None)
        return override or self.default_llm_model

    def validate_required_keys(self) -> None:
        """Validate that required API keys are configured."""
        provider = self.default_llm_provider
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
        }
        api_key = key_map.get(provider, "")
        if not api_key:
            raise ValueError(
                f"API key for default LLM provider '{provider}' is not set. "
                f"Please set {provider.upper()}_API_KEY in your .env file."
            )

        if not self.tavily_api_key:
            logger.warning(
                "TAVILY_API_KEY not set — search functionality will not work"
            )

        if not self.postgres_password:
            logger.warning(
                "POSTGRES_PASSWORD not set — using empty password is insecure"
            )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
