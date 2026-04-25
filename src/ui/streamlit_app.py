import asyncio

import streamlit as st

from src.config import settings

st.set_page_config(
    page_title="Research Assistant",
    page_icon="R",
    layout="wide",
)

st.title("Multi-Agent Research Assistant")


def get_llm_options():
    providers = {
        "Anthropic (Claude)": "anthropic",
        "OpenAI (GPT-4)": "openai",
        "Google (Gemini)": "google",
    }
    return providers


with st.sidebar:
    st.header("Configuration")

    providers = get_llm_options()
    selected_provider = st.selectbox(
        "LLM Provider",
        options=list(providers.keys()),
        index=0,
    )

    model_options = {
        "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "google": ["gemini-2.0-flash", "gemini-2.5-pro-preview-05-06"],
    }
    provider_key = providers[selected_provider]
    selected_model = st.selectbox(
        "Model",
        options=model_options.get(provider_key, []),
        index=0,
    )

    st.divider()
    st.caption("Architecture: Supervisor + Search/Code/Writer Agents")
    st.caption(f"Checkpointer: PostgreSQL")
    st.caption(f"Tracing: LangSmith ({'ON' if settings.langsmith_tracing else 'OFF'})")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your research task..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()
        progress_placeholder.info("Supervisor is analyzing your task...")

        async def run_research():
            from src.graph.workflow import build_workflow

            original_provider = settings.default_llm_provider
            original_model = settings.default_llm_model

            settings.default_llm_provider = provider_key
            settings.default_llm_model = selected_model

            try:
                graph = await build_workflow()
                config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}

                progress_placeholder.info("Search Agent: searching the web...")

                result = await graph.ainvoke(
                    {
                        "messages": [],
                        "task_description": prompt,
                        "search_results": [],
                        "code_results": [],
                        "final_report": "",
                        "retry_count": {},
                        "errors": [],
                    },
                    config,
                )

                progress_placeholder.empty()

                if result.get("final_report"):
                    st.markdown(result["final_report"])

                    st.download_button(
                        label="Download Report (Markdown)",
                        data=result["final_report"],
                        file_name="research_report.md",
                        mime="text/markdown",
                    )

                    return result["final_report"]
                else:
                    error_msg = "Failed to generate report."
                    if result.get("errors"):
                        error_msg += "\n\nErrors:\n" + "\n".join(f"- {e}" for e in result["errors"])
                    st.error(error_msg)
                    return error_msg
            finally:
                settings.default_llm_provider = original_provider
                settings.default_llm_model = original_model

        report = _run_async(run_research())

        st.session_state.messages.append({"role": "assistant", "content": report})


def _run_async(coro):
    """Run an async coroutine, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
