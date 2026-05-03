import logging
import uuid

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for sending tasks to remote A2A agents via HTTP JSON-RPC."""

    def resolve_url(self, agent_name: str) -> str:
        urls = settings.agent_urls
        if agent_name not in urls:
            raise ValueError(f"Unknown agent: {agent_name}")
        return urls[agent_name]

    async def send_task(self, agent_name: str, message: str) -> str:
        url = self.resolve_url(agent_name)
        endpoint = f"{url}/a2a"

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(f"A2A error from {agent_name}: {data['error']}")

        result = data.get("result", {})
        status = result.get("status", {})
        message_obj = status.get("message", {})
        parts = message_obj.get("parts", [])

        texts = [p.get("text", "") for p in parts if p.get("type") == "text" or "text" in p]
        return "\n".join(texts) if texts else str(result)
