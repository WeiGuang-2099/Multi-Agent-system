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

from src.agents.code_agent import CodeAgent
from src.agents.search_agent import SearchAgent
from src.agents.writer_agent import WriterAgent
from src.graph.state import AgentState


def _build_state(user_input: str) -> AgentState:
    """Build agent state from user input. Supports JSON context for distributed mode."""
    try:
        context = json.loads(user_input)
        if isinstance(context, dict) and "task_description" in context:
            return {
                "messages": [],
                "task_description": context.get("task_description", ""),
                "search_results": context.get("search_results", []),
                "code_results": context.get("code_results", []),
                "final_report": context.get("final_report", ""),
                "retry_count": context.get("retry_count", {}),
                "errors": context.get("errors", []),
            }
    except (json.JSONDecodeError, TypeError):
        pass

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
            cmd = await agent.execute(state)

            messages = cmd.update.get("messages", [])
            content = ""
            if messages and isinstance(messages[0].content, str):
                content = messages[0].content

            task = _make_completed_task(context, content)
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
            cmd = await agent.execute(state)

            messages = cmd.update.get("messages", [])
            content = ""
            if messages and isinstance(messages[0].content, str):
                content = messages[0].content

            task = _make_completed_task(context, content)
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
            cmd = await agent.execute(state)

            messages = cmd.update.get("messages", [])
            content = ""
            if messages and isinstance(messages[0].content, str):
                content = messages[0].content

            task = _make_completed_task(context, content)
        except Exception as e:
            task = _make_failed_task(context, f"Report writing failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))


class SupervisorAgentExecutor(AgentExecutor):
    """Executor for the supervisor agent in distributed A2A mode."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()

        task = Task(
            id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.working),
        )
        await event_queue.enqueue_event(task)

        try:
            from src.graph.distributed_workflow import build_distributed_workflow

            graph = await build_distributed_workflow()
            config = {"configurable": {"thread_id": f"a2a-{context.task_id}"}}

            state = _build_state(user_input)
            result = await graph.ainvoke(state, config)

            report = result.get("final_report", "No report generated.")
            task = _make_completed_task(context, report)
        except Exception as e:
            task = _make_failed_task(context, f"Supervisor failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))


EXECUTORS = {
    "supervisor": SupervisorAgentExecutor,
    "search": SearchAgentExecutor,
    "code": CodeAgentExecutor,
    "writer": WriterAgentExecutor,
}
