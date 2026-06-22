"""Streaming event helpers for LangGraph visualization.

This module translates low-level LangGraph `astream_events` events into
human-readable status messages that the UI can render in real time, giving
users visibility into which agent is working and what tools are being called.

Usage in the UI layer:

    async for event in graph.astream_events(input, config, version="v2"):
        for status in interpret_event(event):
            yield status  # e.g. ("start", "search_agent", "🔍 Searching the web...")
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

# Agent display metadata: emoji + short label + descriptive verb phrase.
AGENT_META: dict[str, dict[str, str]] = {
    "supervisor": {
        "emoji": "🧠",
        "label": "Supervisor",
        "verb": "Analyzing task and planning routing",
    },
    "search_agent": {
        "emoji": "🔍",
        "label": "Search Agent",
        "verb": "Searching the web via Tavily",
    },
    "code_agent": {
        "emoji": "💻",
        "label": "Code Agent",
        "verb": "Generating and executing Python",
    },
    "writer_agent": {
        "emoji": "✍️",
        "label": "Writer Agent",
        "verb": "Synthesizing the research report",
    },
    "critic_agent": {
        "emoji": "🔎",
        "label": "Critic Agent",
        "verb": "Reviewing the report for quality",
    },
    "retrieval_agent": {
        "emoji": "📚",
        "label": "Retrieval Agent",
        "verb": "Searching the local knowledge base",
    },
}


@dataclass(frozen=True)
class StatusEvent:
    """A high-level status update derived from LangGraph events.

    kind is one of: "node_start", "node_end", "tool_start", "tool_end", "token".
    """

    kind: str
    node: str
    message: str
    detail: str = ""


def agent_status_icon(node: str) -> str:
    """Return a human-readable icon+label string for a node."""
    meta = AGENT_META.get(node, {"emoji": "⚙️", "label": node, "verb": "working"})
    return f"{meta['emoji']} {meta['label']}"


def _node_from_event(event: dict[str, Any]) -> str:
    """Extract the LangGraph node name from an astream_events event."""
    metadata = event.get("metadata", {}) or {}
    return str(metadata.get("langgraph_node", metadata.get("langgraph_step", "")))


def _tool_name(event: dict[str, Any]) -> str:
    name = event.get("name", "")
    runtime = event.get("run_id", "")
    # astream_events v2 exposes the tool under "name" for on_tool_* events
    return str(name or runtime)


def interpret_event(event: dict[str, Any]) -> list[StatusEvent]:
    """Translate a single astream_events (v2) dict into zero or more StatusEvents.

    This is the core mapping function. It is pure (no I/O) so it can be
    unit-tested without a running graph.
    """
    kind = event.get("event", "")
    node = _node_from_event(event)
    out: list[StatusEvent] = []

    if kind == "on_chain_start" and node:
        meta = AGENT_META.get(node)
        if meta:
            out.append(
                StatusEvent(
                    kind="node_start",
                    node=node,
                    message=f"{meta['emoji']} {meta['label']} — {meta['verb']}...",
                )
            )

    elif kind == "on_chain_end" and node:
        meta = AGENT_META.get(node)
        if meta:
            out.append(
                StatusEvent(
                    kind="node_end",
                    node=node,
                    message=f"✓ {meta['label']} finished",
                )
            )

    elif kind == "on_tool_start":
        tool = _tool_name(event)
        out.append(
            StatusEvent(
                kind="tool_start",
                node=node,
                message=f"  ↳ 🛠️ Calling tool: {tool}",
            )
        )

    elif kind == "on_tool_end":
        out.append(
            StatusEvent(
                kind="tool_end",
                node=node,
                message="  ↳ ✓ Tool completed",
            )
        )

    return out


async def stream_statuses(
    events: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[StatusEvent]:
    """Async generator wrapper: consume raw events, yield StatusEvents.

    Example:

        async for status in stream_statuses(graph.astream_events(..., version="v2")):
            ui_update(status)
    """
    async for event in events:
        for status in interpret_event(event):
            yield status


def format_status_line(status: StatusEvent) -> str:
    """Render a StatusEvent as a single log/UI line."""
    base = status.message
    if status.detail:
        return f"{base} ({status.detail})"
    return base


def summarize_statuses(statuses: Iterable[StatusEvent]) -> list[str]:
    """Deduplicate consecutive identical messages for compact display."""
    seen: list[str] = []
    for s in statuses:
        line = format_status_line(s)
        if not seen or seen[-1] != line:
            seen.append(line)
    return seen
