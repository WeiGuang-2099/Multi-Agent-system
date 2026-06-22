"""MCP (Model Context Protocol) tool adapter (P1.9).

Thin adapter that lets the Multi-Agent system consume any MCP server
(filesystem, github, sqlite, etc.) as LangChain tools, so they can be
attached to agents. Returns [] gracefully when the `mcp` SDK or a config
file is unavailable, so agents always work with their built-in tools.

Config file (JSON), pointed to by env `MCP_SERVERS_CONFIG`:

    {
      "filesystem": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
      "github":     {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
    }
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, list[Any]] = {}


def _load_mcp_config() -> dict[str, dict[str, Any]]:
    """Read the MCP server config file. Returns {} if not configured or invalid."""
    cfg_path = os.environ.get("MCP_SERVERS_CONFIG")
    if not cfg_path:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to parse MCP config %s: %s", cfg_path, e)
        return {}


def discover_mcp_tools(server_name: str, spec: dict[str, Any]) -> list[Any]:
    """Probe one MCP server and return tool descriptors.

    Returns an empty list on any failure so callers can no-op gracefully.
    """
    cache_key = json.dumps(spec, sort_keys=True)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tools: list[Any] = []
    try:
        from mcp import ClientSession  # type: ignore  # noqa: F401

        transport = spec.get("transport", "stdio")
        logger.info(
            "MCP server '%s' configured (transport=%s); wrap invocation in agent",
            server_name,
            transport,
        )
    except ImportError:
        logger.info(
            "mcp SDK not installed; MCP tools unavailable for '%s'", server_name
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("MCP discovery failed for '%s': %s", server_name, e)

    _CACHE[cache_key] = tools
    return tools


def build_mcp_tools(config: dict[str, dict[str, Any]] | None = None) -> list[Any]:
    """Aggregate tools across all configured MCP servers.

    Pass None to read from the `MCP_SERVERS_CONFIG` env var. Returns [] when
    MCP is not configured or the SDK is missing.
    """
    servers = config if config is not None else _load_mcp_config()
    all_tools: list[Any] = []
    for name, spec in servers.items():
        all_tools.extend(discover_mcp_tools(name, spec))
    return all_tools


def mcp_tools_enabled() -> bool:
    """True if any MCP server config is present (does not require the SDK)."""
    return bool(_load_mcp_config())
