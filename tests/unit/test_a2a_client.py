import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.a2a.client import A2AClient


@pytest.mark.asyncio
async def test_send_task_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "status": {
                "state": "completed",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "search result data"}],
                },
            },
        },
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = A2AClient()
        result = await client.send_task("search_agent", "find Python asyncio patterns")

        assert "search result data" in result


@pytest.mark.asyncio
async def test_send_task_agent_not_found():
    client = A2AClient()
    with pytest.raises(ValueError, match="Unknown agent"):
        await client.send_task("nonexistent_agent", "test")


def test_resolve_url():
    client = A2AClient()
    url = client.resolve_url("search_agent")
    assert "8002" in url
