import json
from unittest.mock import MagicMock, patch

from src.tools.search import tavily_search


@patch("src.tools.search._get_client")
def test_tavily_search_returns_json(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Python Asyncio",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "content": "Asyncio is a library for async programming",
                "score": 0.95,
            }
        ]
    }
    mock_get_client.return_value = mock_client

    result = tavily_search.invoke({"query": "Python asyncio"})
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Python Asyncio"
    assert parsed[0]["url"] == "https://docs.python.org/3/library/asyncio.html"
    assert parsed[0]["content"] == "Asyncio is a library for async programming"


@patch("src.tools.search._get_client")
def test_tavily_search_empty_results(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    result = tavily_search.invoke({"query": "nonexistent topic xyz123"})
    parsed = json.loads(result)

    assert len(parsed) == 0
