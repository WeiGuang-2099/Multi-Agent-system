from unittest.mock import patch

from src.a2a.agent_cards import ALL_CARDS
from src.config import settings


def test_all_agent_cards_defined():
    assert "supervisor" in ALL_CARDS
    assert "search" in ALL_CARDS
    assert "code" in ALL_CARDS
    assert "writer" in ALL_CARDS


def test_agent_card_fields():
    card = ALL_CARDS["search"]
    assert card.name == "Search Agent"
    assert card.url == f"http://localhost:{settings.search_agent_port}"
    assert len(card.skills) == 1
    assert card.skills[0].id == "web_search"


def test_agent_card_capabilities():
    for name, card in ALL_CARDS.items():
        assert card.capabilities.streaming is True
        assert card.version == "1.0.0"
        assert len(card.skills) > 0


def test_create_a2a_app():
    with patch("src.a2a.server.InMemoryTaskStore"):
        from src.a2a.server import create_a2a_app

        for agent_name in ["search", "code", "writer"]:
            app = create_a2a_app(agent_name)
            assert app is not None
