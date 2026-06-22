import uvicorn
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from src.a2a.agent_cards import ALL_CARDS
from src.a2a.executor import EXECUTORS


def create_a2a_app(agent_name: str) -> A2AFastAPIApplication:
    card = ALL_CARDS[agent_name]
    executor_class = EXECUTORS[agent_name]

    task_store = InMemoryTaskStore()
    executor = executor_class()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    return A2AFastAPIApplication(
        agent_card=card,
        http_handler=handler,
    )


def run_a2a_server(agent_name: str, port: int) -> None:
    app_builder = create_a2a_app(agent_name)
    app = app_builder.build()
    uvicorn.run(app, host="0.0.0.0", port=port)
