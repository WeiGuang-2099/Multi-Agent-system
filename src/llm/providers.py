import logging

from langchain_core.language_models import BaseChatModel

from src.config import settings

logger = logging.getLogger(__name__)


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    provider = provider or settings.default_llm_provider
    model = model or settings.default_llm_model

    common = {"temperature": temperature, **kwargs}

    # Validate provider is supported before checking API key
    supported_providers = ["anthropic", "openai", "google"]
    if provider not in supported_providers:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # Validate API key
    key_map = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "google": settings.google_api_key,
    }
    api_key = key_map.get(provider, "")
    if not api_key:
        raise ValueError(
            f"API key for LLM provider '{provider}' is not configured. "
            f"Set {provider.upper()}_API_KEY in your .env file."
        )

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
        raise ValueError(f"Unsupported LLM provider: {provider}")  # pragma: no cover


def create_agent_llm(agent_name: str, **kwargs) -> BaseChatModel:
    model = settings.get_model(agent_name)
    return create_llm(model=model, **kwargs)


def create_llm_with_fallback(
    fallback_providers: list[str] | None = None,
    **kwargs,
) -> BaseChatModel:
    """Create LLM with automatic fallback to alternative providers."""
    primary = settings.default_llm_provider
    providers_to_try = [primary]

    if fallback_providers:
        providers_to_try.extend(p for p in fallback_providers if p != primary)
    else:
        all_providers = ["anthropic", "openai", "google"]
        providers_to_try.extend(p for p in all_providers if p != primary)

    last_error = None
    for provider in providers_to_try:
        key_map = {
            "anthropic": settings.anthropic_api_key,
            "openai": settings.openai_api_key,
            "google": settings.google_api_key,
        }
        if not key_map.get(provider, ""):
            continue  # Skip providers without API keys

        try:
            return create_llm(provider=provider, **kwargs)
        except Exception as e:
            logger.warning(f"LLM provider '{provider}' failed: {e}, trying next...")
            last_error = e

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_error}"
    )
