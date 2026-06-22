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

    # P0.4: attach the cost/usage callback so every LLM call is tracked.
    try:
        from src.observability.metrics import UsageEventHandler

        callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks.append(UsageEventHandler())
        common = {"temperature": temperature, "callbacks": callbacks, **kwargs}
    except Exception:  # noqa: BLE001
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


# Agents that benefit from a cheaper model (routine routing/tool-use).
_CHEAP_AGENT_HINTS = {"supervisor", "critic_agent", "retrieval_agent"}
# Agents that produce long-form output and benefit from a stronger model.
_STRONG_AGENT_HINTS = {"writer_agent"}


def create_agent_llm(agent_name: str, **kwargs) -> BaseChatModel:
    """Construct an LLM for a named agent.

    Per-agent model override (`*_AGENT_MODEL`) takes precedence. Otherwise we
    apply a cost-aware default (P1.10): routing/critic/retrieval use the
    `cheap_model`, writer uses `strong_model`, others fall back to the global
    default. This keeps token spend low on the high-frequency control path
    while preserving output quality where it matters.
    """
    # 1. Explicit per-agent override always wins.
    override = getattr(settings, f"{agent_name}_model", None)
    if override:
        return create_llm(model=override, **kwargs)

    # 2. Cost-aware default selection (only for OpenAI provider, where we have
    #    reliable cheap/strong split; other providers ignore this hint).
    if settings.default_llm_provider == "openai":
        if agent_name in _CHEAP_AGENT_HINTS and settings.cheap_model:
            return create_llm(model=settings.cheap_model, **kwargs)
        if agent_name in _STRONG_AGENT_HINTS and settings.strong_model:
            return create_llm(model=settings.strong_model, **kwargs)

    # 3. Global default.
    return create_llm(model=settings.default_llm_model, **kwargs)


def get_cost_snapshot() -> dict[str, dict[str, float]]:
    """Return the current per-model token/cost snapshot (P1.10 helper).

    Useful for surfacing a 'cost so far' badge in the UI or for budget checks
    before launching a new agent call.
    """
    try:
        from src.observability.metrics import get_global_tracker

        return get_global_tracker().snapshot()
    except Exception:  # noqa: BLE001
        return {}


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
