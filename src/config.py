from pydantic_settings import BaseSettings


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
    postgres_password: str = "agent_password"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "multi-agent-research-assistant"

    # A2A Ports
    supervisor_port: int = 8001
    search_agent_port: int = 8002
    code_agent_port: int = 8003
    writer_agent_port: int = 8004

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

    def get_model(self, agent_name: str) -> str:
        override = getattr(self, f"{agent_name}_model", None)
        return override or self.default_llm_model

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
