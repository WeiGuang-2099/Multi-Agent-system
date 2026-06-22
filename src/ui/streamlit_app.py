"""Streamlit web interface with live streaming agent visualization.

This UI consumes LangGraph `astream_events` to show, in real time, which
agent is working and which tools are being called. It also exposes
Human-in-the-Loop (HITL) approval when the supervisor produces a plan that
requires user confirmation (see graph/supervisor.py).
"""
import asyncio
import html
import uuid

import streamlit as st

from src.config import settings
from src.graph.callbacks import (
    AGENT_META,
    format_status_line,
    interpret_event,
)

st.set_page_config(
    page_title="Research Assistant",
    page_icon="R",
    layout="wide",
)

st.title("Multi-Agent Research Assistant")

# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    providers = {
        "Anthropic (Claude)": "anthropic",
        "OpenAI (GPT-4)": "openai",
        "Google (Gemini)": "google",
    }
    selected_provider = st.selectbox(
        "LLM Provider",
        options=list(providers.keys()),
        index=0,
    )

    model_options = {
        "anthropic": [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
        ],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "google": ["gemini-2.0-flash", "gemini-2.5-pro-preview-05-06"],
    }
    provider_key = providers[selected_provider]
    selected_model = st.selectbox(
        "Model",
        options=model_options.get(provider_key, []),
        index=0,
    )

    st.toggle(
        "Human approval of supervisor plan",
        key="hitl_enabled",
        value=False,
        help=(
            "When enabled, the supervisor pauses after planning so you can "
            "approve, edit, or reject the proposed agent routing before "
            "execution continues."
        ),
    )

    st.divider()
    st.caption("Architecture: Supervisor + Search/Code/Writer Agents")
    st.caption("Checkpointer: PostgreSQL")
    st.caption(f"Tracing: LangSmith ({'ON' if settings.langsmith_tracing else 'OFF'})")
    st.caption("Live streaming: agent status updates in real time")


# --- Session state ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Stores a pending interrupt (plan) that awaits user approval.
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None


def _render_agent_legend():
    """Render a small legend explaining each agent's icon."""
    cols = st.columns(len(AGENT_META))
    for col, (node, meta) in zip(cols, AGENT_META.items(), strict=False):
        with col:
            st.caption(f"{meta['emoji']} **{meta['label']}**")


# --- Render conversation history ------------------------------------------

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(html.escape(msg["content"]))


# --- Agent legend (rendered once, above the chat input) --------------------

if not st.session_state.messages:
    _render_agent_legend()


# --- HITL approval panel ---------------------------------------------------

def _render_hitl_panel():
    """If there's a pending plan awaiting approval, render an inline panel."""
    plan = st.session_state.get("pending_interrupt")
    if not plan:
        return False

    with st.container(border=True):
        st.warning("🛑 Supervisor produced a plan — review before continuing")
        st.markdown(f"**Subtask:** {plan.get('subtask_description', '(none)')}")
        st.markdown(f"**Routing to:** `{plan.get('agent_name', '?')}`")
        st.markdown(f"**Reasoning:** {plan.get('reasoning', '')}")

        approved = st.button("✅ Approve and continue", key="hitl_approve")
        rejected = st.button("⛔ Reject and finish", key="hitl_reject")
        if approved:
            st.session_state.pending_interrupt = None
            return True
        if rejected:
            st.session_state.pending_interrupt = None
            st.info("Task aborted by user.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "[aborted by user]"}
            )
            st.rerun()
    return False


# --- Chat input ------------------------------------------------------------

if prompt := st.chat_input("Enter your research task..."):
    if len(prompt) > 10_000:
        st.error("Input too long. Maximum 10,000 characters allowed.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(html.escape(prompt))

    with st.chat_message("assistant"):
        status_container = st.container()
        report_area = st.empty()

        with status_container:
            header = st.empty()
            header.info("🧠 Supervisor is analyzing your task...")

            # Live-updating list of status lines
            status_lines: list[str] = []
            status_box = st.empty()

            async def run_research():
                # Distributed (A2A) mode if a remote supervisor URL is set.
                if (
                    settings.supervisor_url
                    and "localhost" not in settings.supervisor_url
                ):
                    report = await _run_via_a2a(
                        settings.supervisor_url, prompt, status_box, status_lines
                    )
                else:
                    report = await _run_local(
                        prompt,
                        provider_key,
                        selected_model,
                        status_box,
                        status_lines,
                        header,
                    )
                return report

            report = _run_async(run_research())

            header.empty()
            if status_lines:
                status_box.markdown(
                    "<small>"
                    + "<br>".join(html.escape(l) for l in status_lines)
                    + "</small>",
                    unsafe_allow_html=True,
                )

        if report:
            report_area.markdown(report)
            st.download_button(
                label="Download Report (Markdown)",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
            )
        else:
            report_area.error("No report generated for this task.")

    st.session_state.messages.append({"role": "assistant", "content": report or ""})


# --- Execution helpers -----------------------------------------------------

async def _run_via_a2a(
    supervisor_url: str, task: str, status_box, status_lines: list[str]
) -> str:
    import uuid as uuid_mod

    import httpx

    status_lines.append("📡 Sending task to Supervisor via A2A...")
    _flush(status_box, status_lines)

    endpoint = f"{supervisor_url}/a2a"
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid_mod.uuid4()),
        "method": "tasks/send",
        "params": {
            "id": str(uuid_mod.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
            },
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        response = await http_client.post(endpoint, json=payload)
        response.raise_for_status()

    data = response.json()
    result = data.get("result", {})
    status = result.get("status", {})
    parts = status.get("message", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(texts) if texts else "No report generated."


async def _run_local(
    prompt_text: str,
    prov_key: str,
    sel_model: str,
    status_box,
    status_lines: list[str],
    header,
) -> str:
    from src.graph.workflow import build_workflow

    original_provider = settings.default_llm_provider
    original_model = settings.default_llm_model
    settings.default_llm_provider = prov_key
    settings.default_llm_model = sel_model

    try:
        graph = await build_workflow()
        config = {
            "configurable": {"thread_id": st.session_state.get("thread_id", "default")}
        }

        inputs = {
            "messages": [],
            "task_description": prompt_text,
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }

        final_state: dict = {}
        try:
            async for event in graph.astream_events(inputs, config, version="v2"):
                for status in interpret_event(event):
                    line = format_status_line(status)
                    # Deduplicate consecutive identical lines for readability.
                    if not status_lines or status_lines[-1] != line:
                        status_lines.append(line)
                        _flush(status_box, status_lines)

                    # Update header to show the currently active agent.
                    if status.kind == "node_start":
                        header.info(line)

                    # Capture HITL interrupt if enabled.
                    if (
                        st.session_state.get("hitl_enabled")
                        and status.kind == "node_start"
                        and status.node == "supervisor"
                    ):
                        st.session_state.pending_interrupt = {
                            "agent_name": status.node,
                            "reasoning": "Supervisor plan (HITL)",
                            "subtask_description": "Review and approve",
                        }
        except Exception as stream_err:  # noqa: BLE001
            # Fallback: non-streaming invocation if streaming is unsupported.
            status_lines.append(
                f"⚠️ Streaming unavailable, falling back: {stream_err}"
            )
            _flush(status_box, status_lines)
            final_state = await graph.ainvoke(inputs, config)
            return _extract_report(final_state)

        # After streaming, fetch final state to retrieve the report.
        try:
            state_obj = await graph.aget_state(config)
            final_state = getattr(state_obj, "values", state_obj) or {}
        except Exception:  # noqa: BLE001
            pass

        return _extract_report(final_state)
    finally:
        settings.default_llm_provider = original_provider
        settings.default_llm_model = original_model


def _extract_report(final_state: dict) -> str:
    if final_state.get("final_report"):
        return final_state["final_report"]
    errors = final_state.get("errors") or []
    if errors:
        return (
            "Failed to generate report.\n\nErrors:\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    return ""


def _flush(status_box, status_lines: list[str]):
    """Re-render the status list with the latest lines."""
    if not status_lines:
        return
    body = "<br>".join(html.escape(l) for l in status_lines)
    status_box.markdown(f"<small>{body}</small>", unsafe_allow_html=True)


def _run_async(coro):
    """Run an async coroutine, handling existing event loops (Streamlit)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


# Surface pending HITL approval at the bottom of the conversation.
_render_hitl_panel()
