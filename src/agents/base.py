from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.config import settings
from src.graph.state import AgentState


class BaseAgent(ABC):
    agent_name: str = "base"

    def execute(self, state: AgentState) -> Command:
        try:
            result = self._run(state)
            return Command(
                update={
                    "messages": [
                        AIMessage(content=f"[{self.agent_name}] {result}")
                    ]
                },
                goto="supervisor",
            )
        except Exception as e:
            retry_count = state.get("retry_count", {}).copy()
            current = retry_count.get(self.agent_name, 0)
            retry_count[self.agent_name] = current + 1

            errors = state.get("errors", []).copy()
            errors.append(f"{self.agent_name} failed: {str(e)}")

            if retry_count[self.agent_name] >= settings.max_retries:
                return Command(
                    update={
                        "retry_count": retry_count,
                        "errors": errors,
                    },
                    goto="supervisor",
                )

            return Command(
                update={
                    "retry_count": retry_count,
                    "errors": errors,
                },
                goto=self.agent_name,
            )

    @abstractmethod
    def _run(self, state: AgentState) -> str:
        ...
