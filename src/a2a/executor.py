import asyncio
import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Message,
    Part,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from src.agents.search_agent import SearchAgent
from src.agents.code_agent import CodeAgent
from src.agents.writer_agent import WriterAgent
from src.graph.state import AgentState


def _build_state(user_input: str) -> AgentState:
    return {
        "messages": [],
        "task_description": user_input,
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
    }


def _make_completed_task(context: RequestContext, result: str) -> Task:
    return Task(
        id=context.task_id,
        context_id=context.context_id,
        status=TaskStatus(
            state=TaskState.completed,
            message=Message(
                role="agent",
                parts=[Part(root=TextPart(text=result))],
                message_id=f"msg-{context.task_id}",
            ),
        ),
    )


def _make_failed_task(context: RequestContext, error_msg: str) -> Task:
    return Task(
        id=context.task_id,
        context_id=context.context_id,
        status=TaskStatus(
            state=TaskState.failed,
            message=Message(
                role="agent",
                parts=[Part(root=TextPart(text=error_msg))],
                message_id=f"msg-{context.task_id}-error",
            ),
        ),
    )


def _make_canceled_task(context: RequestContext) -> Task:
    return Task(
        id=context.task_id,
        context_id=context.context_id,
        status=TaskStatus(state=TaskState.canceled),
    )


class SearchAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
        )
        await event_queue.enqueue_event(task)

        try:
            agent = SearchAgent()
            state = _build_state(user_input)
            result = await asyncio.to_thread(agent._run, state)

            task = _make_completed_task(context, result)
        except Exception as e:
            task = _make_failed_task(context, f"Search failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))


class CodeAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
        )
        await event_queue.enqueue_event(task)

        try:
            agent = CodeAgent()
            state = _build_state(user_input)
            result = await asyncio.to_thread(agent._run, state)

            task = _make_completed_task(context, result)
        except Exception as e:
            task = _make_failed_task(context, f"Code execution failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))


class WriterAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
        )
        await event_queue.enqueue_event(task)

        try:
            agent = WriterAgent()
            state = _build_state(user_input)
            result = await asyncio.to_thread(agent._run, state)

            task = _make_completed_task(context, result)
        except Exception as e:
            task = _make_failed_task(context, f"Report writing failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))


EXECUTORS = {
    "search": SearchAgentExecutor,
    "code": CodeAgentExecutor,
    "writer": WriterAgentExecutor,
}
