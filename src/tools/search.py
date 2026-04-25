import json
from functools import lru_cache

from langchain_core.tools import tool

from src.config import settings


@lru_cache(maxsize=1)
def _get_client():
    from tavily import TavilyClient

    return TavilyClient(api_key=settings.tavily_api_key)


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API. Returns structured search results."""
    client = _get_client()
    results = client.search(query, max_results=max_results)

    formatted = []
    for r in results.get("results", []):
        formatted.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0),
        })

    return json.dumps(formatted, ensure_ascii=False, indent=2)
