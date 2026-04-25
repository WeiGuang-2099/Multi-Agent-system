from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class SubTask(BaseModel):
    name: str
    agent: Literal["search_agent", "code_agent", "writer_agent"]
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    task_description: str
    search_results: list[str]
    code_results: list[str]
    final_report: str
    retry_count: dict[str, int]
    errors: list[str]
    report_ready: bool


class RouteDecision(BaseModel):
    agent_name: Literal["search_agent", "code_agent", "writer_agent", "FINISH"]
    reasoning: str
    subtask_description: str = ""
