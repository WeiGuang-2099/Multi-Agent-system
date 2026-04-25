from langchain_core.language_models import BaseChatModel

from src.config import settings


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    provider = provider or settings.default_llm_provider
    model = model or settings.default_llm_model

    common = {"temperature": temperature, **kwargs}

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            anthropic_api_key=settings.anthropic_api_key,
            **common,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            **common,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            **common,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_agent_llm(agent_name: str, **kwargs) -> BaseChatModel:
    model = settings.get_model(agent_name)
    return create_llm(model=model, **kwargs)
