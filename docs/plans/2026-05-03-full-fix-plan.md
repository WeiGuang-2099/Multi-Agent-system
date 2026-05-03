# Full Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all identified issues in the Multi-Agent Research Assistant — infrastructure, A2A distributed mode, interactive CLI, Streamlit UI, and test coverage.

**Architecture:** 4 modules executed in dependency order. Module 1 (infra) is the foundation. Module 2 (A2A) depends on Module 1. Modules 3 (CLI/UI) and 4 (tests) depend on Modules 1+2. All changes maintain backward compatibility with existing single-process mode.

**Tech Stack:** LangGraph, A2A SDK, httpx (async HTTP), Streamlit, pytest, Docker

---

### Task 1: Add missing config fields

**Files:**
- Modify: `src/config.py:1-87`
- Modify: `.env.example:1-42`

**Step 1: Update config.py**

Add `streamlit_port`, `supervisor_url`, and `agent_urls` property:

```python
# In class Settings, add after writer_agent_port (line ~38):

    # Streamlit
    streamlit_port: int = 8501

    # A2A Client
    supervisor_url: str = "http://localhost:8001"

    @property
    def agent_urls(self) -> dict[str, str]:
        return {
            "search_agent": f"http://localhost:{self.search_agent_port}",
            "code_agent": f"http://localhost:{self.code_agent_port}",
            "writer_agent": f"http://localhost:{self.writer_agent_port}",
            "supervisor": f"http://localhost:{self.supervisor_port}",
        }
```

**Step 2: Verify settings load correctly**

Run: `python -c "from src.config import settings; print(settings.streamlit_port, settings.supervisor_url, settings.agent_urls)"`
Expected: `8501 http://localhost:8001 {'search_agent': ...}`

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add streamlit_port, supervisor_url, agent_urls to config"
```

---

### Task 2: Fix build_workflow graceful PG degradation

**Files:**
- Modify: `src/graph/workflow.py:38-67`
- Test: `tests/unit/test_workflow_build.py` (create)

**Step 1: Write failing test**

Create `tests/unit/test_workflow_build.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_build_workflow_falls_back_to_memory():
    """When PostgreSQL is unavailable, workflow falls back to MemorySaver."""
    with patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
        side_effect=Exception("PG down"),
    ):
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None


@pytest.mark.asyncio
async def test_build_workflow_uses_pg_when_available():
    """When PostgreSQL is available, workflow uses AsyncPostgresSaver."""
    mock_checkpointer = MagicMock()
    mock_checkpointer.setup = AsyncMock()

    with patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
        return_value=mock_checkpointer,
    ):
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None


def test_build_workflow_sync_falls_back_to_memory():
    """Sync variant also falls back gracefully."""
    with patch(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string",
        side_effect=Exception("PG down"),
    ):
        from src.graph.workflow import build_workflow_sync

        graph = build_workflow_sync()
        assert graph is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_workflow_build.py -v`
Expected: FAIL — current code raises ConnectionError

**Step 3: Implement graceful degradation**

Replace `build_workflow()` and `build_workflow_sync()` in `src/graph/workflow.py`:

```python
async def build_workflow():
    """Build the async workflow with PostgreSQL checkpointing, falling back to MemorySaver."""
    builder = _build_graph_builder()

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpointer = AsyncPostgresSaver.from_conn_string(settings.postgres_uri)
        await checkpointer.setup()
        logger.info("Using PostgreSQL checkpointer")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}), falling back to MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def build_workflow_sync():
    """Build the sync workflow with PostgreSQL checkpointing, falling back to MemorySaver."""
    builder = _build_graph_builder()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver.from_conn_string(settings.postgres_uri)
        checkpointer.setup()
        logger.info("Using PostgreSQL checkpointer (sync)")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable ({e}), falling back to MemorySaver (sync)")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_workflow_build.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/graph/workflow.py tests/unit/test_workflow_build.py
git commit -m "feat: graceful PG-to-Memory checkpoint fallback"
```

---

### Task 3: Make BaseAgent.execute async

**Files:**
- Modify: `src/agents/base.py:1-56`
- Modify: `src/agents/search_agent.py:58-85`
- Modify: `src/agents/code_agent.py:65-85`
- Modify: `src/agents/writer_agent.py:45-64`
- Modify: `src/a2a/executor.py:88-110` (SearchAgentExecutor example)

**Step 1: Update BaseAgent**

Replace `src/agents/base.py` entirely:

```python
import asyncio
from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.config import settings
from src.graph.state import AgentState


class BaseAgent(ABC):
    agent_name: str = "base"

    async def execute(self, state: AgentState) -> Command:
        try:
            result = await asyncio.to_thread(self._run, state)
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
            errors.append(f"{self.agent_name} failed (attempt {current + 1}/{settings.max_retries}): {str(e)}")

            if retry_count[self.agent_name] >= settings.max_retries:
                return Command(
                    update={
                        "retry_count": retry_count,
                        "errors": errors,
                    },
                    goto="supervisor",
                )

            delay = min(2 ** current, 16)
            await asyncio.sleep(delay)

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
```

**Step 2: Update agent node wrappers to be async**

Update `src/agents/search_agent.py` — replace `search_agent_node`:

```python
async def search_agent_node(state: AgentState) -> Command:
    agent = SearchAgent()
    result = await agent.execute(state)

    if result.goto == "search_agent":
        return result

    search_results = state.get("search_results", [])

    return Command(
        update={
            "messages": result.update.get("messages", []),
            "search_results": search_results + [_extract_summary(result)],
            "retry_count": result.update.get("retry_count", state.get("retry_count", {})),
            "errors": result.update.get("errors", state.get("errors", [])),
        },
        goto=result.goto,
    )
```

Update `src/agents/code_agent.py` — replace `code_agent_node`:

```python
async def code_agent_node(state: AgentState) -> Command:
    agent = CodeAgent()
    result = await agent.execute(state)

    code_results = state.get("code_results", [])
    messages = result.update.get("messages", [])

    summary = ""
    if messages and hasattr(messages[0], "content"):
        summary = messages[0].content if isinstance(messages[0].content, str) else ""

    return Command(
        update={
            "messages": messages,
            "code_results": code_results + [summary] if summary else code_results,
            "retry_count": result.update.get("retry_count", state.get("retry_count", {})),
            "errors": result.update.get("errors", state.get("errors", [])),
        },
        goto=result.goto,
    )
```

Update `src/agents/writer_agent.py` — replace `writer_agent_node`:

```python
async def writer_agent_node(state: AgentState) -> Command:
    agent = WriterAgent()
    result = await agent.execute(state)

    if result.goto == "writer_agent":
        return result

    messages = result.update.get("messages", [])
    report = ""
    if messages and hasattr(messages[0], "content"):
        report = messages[0].content if isinstance(messages[0].content, str) else ""

    return Command(
        update={
            "messages": [AIMessage(content=report)],
            "final_report": report,
            "report_ready": True,
        },
        goto="__end__",
    )
```

**Step 3: Update A2A executors**

In `src/a2a/executor.py`, change all `asyncio.to_thread(agent._run, state)` to `await agent.execute(state)` and extract result from the returned Command. Simplify SearchAgentExecutor as example:

```python
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
            result = await agent.execute(state)

            messages = result.update.get("messages", [])
            content = messages[0].content if messages and isinstance(messages[0].content, str) else ""

            task = _make_completed_task(context, content)
        except Exception as e:
            task = _make_failed_task(context, f"Search failed: {str(e)}")

        await event_queue.enqueue_event(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(_make_canceled_task(context))
```

Apply same pattern to `CodeAgentExecutor` and `WriterAgentExecutor`.

**Step 4: Run existing tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agents/ src/a2a/executor.py
git commit -m "refactor: make BaseAgent.execute async with asyncio.sleep"
```

---

### Task 4: Delete empty markdown.py and fix main.py streamlit_port

**Files:**
- Delete: `src/tools/markdown.py`
- Modify: `src/main.py:47-56`

**Step 1: Delete markdown.py**

```bash
rm src/tools/markdown.py
```

**Step 2: Fix streamlit_port in main.py**

Replace `_run_streamlit()`:

```python
def _run_streamlit():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py",
         "--server.port", str(settings.streamlit_port)],
        cwd=".",
    )
```

**Step 3: Verify**

Run: `python -c "from src.main import main; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add -A
git commit -m "fix: delete empty markdown.py, remove streamlit_port hasattr hack"
```

---

### Task 5: Create A2A client

**Files:**
- Create: `src/a2a/client.py`
- Test: `tests/unit/test_a2a_client.py` (create)

**Step 1: Write failing test**

Create `tests/unit/test_a2a_client.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock, patch

from src.a2a.client import A2AClient


@pytest.mark.asyncio
async def test_send_task_success():
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "status": {"state": "completed", "message": {"role": "agent", "parts": [{"type": "text", "text": "search result data"}]}},
        },
    }
    mock_response.raise_for_status = lambda: None

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_a2a_client.py -v`
Expected: FAIL — module not found

**Step 3: Implement A2AClient**

Create `src/a2a/client.py`:

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_a2a_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/a2a/client.py tests/unit/test_a2a_client.py
git commit -m "feat: add A2A HTTP client for cross-agent communication"
```

---

### Task 6: Create distributed workflow

**Files:**
- Create: `src/graph/distributed_workflow.py`
- Test: `tests/unit/test_distributed_workflow.py` (create)

**Step 1: Write failing test**

Create `tests/unit/test_distributed_workflow.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_distributed_workflow_builds():
    with patch("src.graph.distributed_workflow.A2AClient") as mock_cls:
        mock_cls.return_value = AsyncMock()
        from src.graph.distributed_workflow import build_distributed_workflow

        graph = await build_distributed_workflow()
        assert graph is not None


@pytest.mark.asyncio
async def test_distributed_search_node_calls_client():
    mock_client = AsyncMock()
    mock_client.send_task = AsyncMock(return_value="search results here")

    with patch("src.graph.distributed_workflow.A2AClient", return_value=mock_client):
        from src.graph.distributed_workflow import _make_remote_agent_node

        node = _make_remote_agent_node("search_agent")
        state = {
            "messages": [],
            "task_description": "test task",
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = await node(state)
        assert mock_client.send_task.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_distributed_workflow.py -v`
Expected: FAIL — module not found

**Step 3: Implement distributed workflow**

Create `src/graph/distributed_workflow.py`:

```python
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph.state import AgentState
from src.graph.supervisor import route_after_worker, supervisor_node
from src.a2a.client import A2AClient

logger = logging.getLogger(__name__)

_client = A2AClient()


def _make_remote_agent_node(agent_name: str):
    async def remote_node(state: AgentState) -> Command:
        task_desc = state.get("task_description", "")
        last_msg = state["messages"][-1].content if state.get("messages") else ""

        context = json.dumps({
            "task_description": task_desc,
            "search_results": state.get("search_results", []),
            "code_results": state.get("code_results", []),
        }, ensure_ascii=False)

        message = f"{last_msg}\n\nContext: {context}" if last_msg else context

        try:
            result_text = await _client.send_task(agent_name, message)
        except Exception as e:
            logger.error(f"A2A call to {agent_name} failed: {e}")
            errors = state.get("errors", []).copy()
            errors.append(f"{agent_name} remote call failed: {str(e)}")
            return Command(
                update={"errors": errors},
                goto="supervisor",
            )

        update = {
            "messages": [AIMessage(content=f"[{agent_name}] {result_text}")],
        }

        if agent_name == "search_agent":
            search_results = state.get("search_results", [])
            update["search_results"] = search_results + [result_text]
        elif agent_name == "code_agent":
            code_results = state.get("code_results", [])
            update["code_results"] = code_results + [result_text]
        elif agent_name == "writer_agent":
            update["final_report"] = result_text
            update["report_ready"] = True
            return Command(update=update, goto="__end__")

        return Command(update=update, goto="supervisor")

    return remote_node


async def build_distributed_workflow():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("search_agent", _make_remote_agent_node("search_agent"))
    builder.add_node("code_agent", _make_remote_agent_node("code_agent"))
    builder.add_node("writer_agent", _make_remote_agent_node("writer_agent"))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor_dist,
        ["search_agent", "code_agent", "writer_agent", "__end__"],
    )
    builder.add_conditional_edges("search_agent", route_after_worker)
    builder.add_conditional_edges("code_agent", route_after_worker)
    builder.add_conditional_edges("writer_agent", route_after_worker)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def _route_from_supervisor_dist(state: AgentState) -> str:
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if isinstance(content, str):
            for agent in ["search_agent", "code_agent", "writer_agent"]:
                if f"Routing to {agent}" in content:
                    return agent
            if "All tasks completed" in content:
                return END
    return END
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_distributed_workflow.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/graph/distributed_workflow.py tests/unit/test_distributed_workflow.py
git commit -m "feat: distributed workflow with A2A HTTP agent nodes"
```

---

### Task 7: Wire SupervisorAgentExecutor to distributed workflow

**Files:**
- Modify: `src/a2a/executor.py:165-196`

**Step 1: Update SupervisorAgentExecutor**

Replace the `execute` method of `SupervisorAgentExecutor` in `src/a2a/executor.py`:

```python
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
```

**Step 2: Run existing tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add src/a2a/executor.py
git commit -m "feat: SupervisorExecutor uses distributed workflow with HTTP agent calls"
```

---

### Task 8: Make agent card URLs dynamic

**Files:**
- Modify: `src/a2a/agent_cards.py:1-84`

**Step 1: Update agent_cards.py**

Replace hardcoded URLs with settings-derived values:

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from src.config import settings


def _base_url(port: int) -> str:
    return f"http://localhost:{port}"


SUPERVISOR_CARD = AgentCard(
    name="Research Supervisor",
    description="Orchestrates research tasks by decomposing them and routing to specialized agents",
    version="1.0.0",
    url=_base_url(settings.supervisor_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "text/markdown"],
    skills=[
        AgentSkill(
            id="orchestrate_research",
            name="Orchestrate Research",
            description="Decompose a research task and coordinate search, code, and writing agents",
            tags=["orchestration", "research"],
            examples=["Research Python asyncio performance patterns"],
        )
    ],
)

SEARCH_AGENT_CARD = AgentCard(
    name="Search Agent",
    description="Performs web searches using Tavily API to find relevant information",
    version="1.0.0",
    url=_base_url(settings.search_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="web_search",
            name="Web Search",
            description="Search the web for information on a given topic",
            tags=["search", "web", "research"],
            examples=["Search for Python asyncio best practices"],
        )
    ],
)

CODE_AGENT_CARD = AgentCard(
    name="Code Agent",
    description="Generates and executes Python code in a sandboxed Docker environment",
    version="1.0.0",
    url=_base_url(settings.code_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/plain", "application/json"],
    skills=[
        AgentSkill(
            id="code_execution",
            name="Code Execution",
            description="Generate and execute Python code for computation and data analysis",
            tags=["code", "python", "execution", "analysis"],
            examples=["Analyze sorting algorithm performance", "Compute statistical metrics"],
        )
    ],
)

WRITER_AGENT_CARD = AgentCard(
    name="Writer Agent",
    description="Synthesizes research results into structured Markdown reports",
    version="1.0.0",
    url=_base_url(settings.writer_agent_port),
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain", "application/json"],
    default_output_modes=["text/markdown"],
    skills=[
        AgentSkill(
            id="report_writing",
            name="Report Writing",
            description="Write a structured Markdown research report from provided data",
            tags=["writing", "report", "markdown"],
            examples=["Write a report on machine learning trends"],
        )
    ],
)

ALL_CARDS = {
    "supervisor": SUPERVISOR_CARD,
    "search": SEARCH_AGENT_CARD,
    "code": CODE_AGENT_CARD,
    "writer": WRITER_AGENT_CARD,
}
```

**Step 2: Verify existing A2A tests still pass**

Run: `pytest tests/integration/test_a2a.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/a2a/agent_cards.py
git commit -m "refactor: agent card URLs from settings instead of hardcoded"
```

---

### Task 9: Implement interactive CLI (REPL)

**Files:**
- Modify: `src/main.py:81-85`
- Test: `tests/unit/test_interactive_cli.py` (create)

**Step 1: Write failing test**

Create `tests/unit/test_interactive_cli.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_interactive_mode_exists():
    """The _run_interactive function should exist."""
    from src.main import _run_interactive
    assert callable(_run_interactive)


@pytest.mark.asyncio
async def test_execute_task_returns_report():
    """_execute_task should invoke workflow and return report."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "# Test Report\nContent here",
        "errors": [],
    })

    with patch("src.main.build_workflow", AsyncMock(return_value=mock_graph)):
        from src.main import _execute_task

        result = await _execute_task("test task")
        assert result == "# Test Report\nContent here"


@pytest.mark.asyncio
async def test_execute_task_handles_no_report():
    """_execute_task should return None when no report generated."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "",
        "errors": ["something failed"],
    })

    with patch("src.main.build_workflow", AsyncMock(return_value=mock_graph)):
        from src.main import _execute_task

        result = await _execute_task("test task")
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_interactive_cli.py -v`
Expected: FAIL — `_run_interactive` not found

**Step 3: Implement interactive CLI**

Replace `_run_supervisor` in `src/main.py`:

```python
def _run_supervisor(task: str | None):
    if task:
        asyncio.run(_execute_and_print(task))
    else:
        _run_interactive()


def _run_interactive():
    print("Multi-Agent Research Assistant — Interactive Mode")
    print("Type your research task and press Enter. Type 'exit' or 'quit' to stop.")
    print()

    thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        asyncio.run(_execute_and_print(user_input, thread_id))
        thread_id = str(uuid.uuid4())  # New thread per task for clean state


async def _execute_and_print(task: str, thread_id: str | None = None):
    result = await _execute_task(task, thread_id)
    print()
    print("=" * 60)
    if result:
        print(result)
    else:
        print("No report generated for this task.")
    print("=" * 60)
    print()


async def _execute_task(task: str, thread_id: str | None = None):
    from src.graph.workflow import build_workflow

    graph = await build_workflow()
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    result = await graph.ainvoke(
        {
            "messages": [],
            "task_description": task,
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        },
        config,
    )

    if result.get("final_report"):
        return result["final_report"]

    if result.get("errors"):
        print("Errors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")

    return None
```

Remove old `_execute_task` and update import of `build_workflow` to be inside the function (lazy import).

**Step 4: Run tests**

Run: `pytest tests/unit/test_interactive_cli.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/main.py tests/unit/test_interactive_cli.py
git commit -m "feat: interactive REPL CLI mode with multi-turn support"
```

---

### Task 10: Streamlit UI distributed mode support

**Files:**
- Modify: `src/ui/streamlit_app.py:79-131`

**Step 1: Add A2A client path to Streamlit**

In `src/ui/streamlit_app.py`, replace the `run_research` function:

```python
async def run_research():
    supervisor_url = settings.supervisor_url

    if supervisor_url and "localhost" not in supervisor_url:
        report = await _run_via_a2a(supervisor_url, prompt)
    else:
        report = await _run_local(prompt, provider_key, selected_model)

    return report


async def _run_via_a2a(supervisor_url: str, task: str) -> str:
    from src.a2a.client import A2AClient

    client = A2AClient()
    # Override supervisor URL for docker network
    import httpx
    import uuid as uuid_mod

    endpoint = f"{supervisor_url}/a2a"
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid_mod.uuid4()),
        "method": "tasks/send",
        "params": {
            "id": str(uuid_mod.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
            },
        },
    }

    progress_placeholder.info("Sending task to Supervisor via A2A...")

    async with httpx.AsyncClient(timeout=300.0) as http_client:
        response = await http_client.post(endpoint, json=payload)
        response.raise_for_status()

    data = response.json()
    result = data.get("result", {})
    status = result.get("status", {})
    parts = status.get("message", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if "text" in p]

    progress_placeholder.empty()
    report = "\n".join(texts) if texts else "No report generated."
    st.markdown(report)
    return report


async def _run_local(prompt: str, provider_key: str, selected_model: str) -> str:
    from src.graph.workflow import build_workflow

    original_provider = settings.default_llm_provider
    original_model = settings.default_llm_model

    settings.default_llm_provider = provider_key
    settings.default_llm_model = selected_model

    try:
        graph = await build_workflow()
        config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}

        progress_placeholder.info("Processing research task...")

        result = await graph.ainvoke(
            {
                "messages": [],
                "task_description": prompt,
                "search_results": [],
                "code_results": [],
                "final_report": "",
                "retry_count": {},
                "errors": [],
            },
            config,
        )

        progress_placeholder.empty()

        if result.get("final_report"):
            st.markdown(result["final_report"])
            st.download_button(
                label="Download Report (Markdown)",
                data=result["final_report"],
                file_name="research_report.md",
                mime="text/markdown",
            )
            return result["final_report"]
        else:
            error_msg = "Failed to generate report."
            if result.get("errors"):
                error_msg += "\n\nErrors:\n" + "\n".join(f"- {e}" for e in result["errors"])
            st.error(error_msg)
            return error_msg
    finally:
        settings.default_llm_provider = original_provider
        settings.default_llm_model = original_model
```

**Step 2: Verify no syntax errors**

Run: `python -c "from src.ui.streamlit_app import *; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/ui/streamlit_app.py
git commit -m "feat: Streamlit UI supports A2A distributed mode via supervisor_url"
```

---

### Task 11: Add supervisor and writer agent unit tests

**Files:**
- Create: `tests/unit/test_supervisor.py`
- Create: `tests/unit/test_writer_agent.py`

**Step 1: Write supervisor tests**

Create `tests/unit/test_supervisor.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import AgentState


def test_supervisor_routes_to_search():
    from src.llm.providers import create_llm
    from src.graph.supervisor import supervisor_node

    mock_decision = MagicMock()
    mock_decision.agent_name = "search_agent"
    mock_decision.reasoning = "Need info"
    mock_decision.subtask_description = "Search for patterns"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Research Python asyncio",
            "search_results": [],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "search_agent"


def test_supervisor_routes_to_code():
    mock_decision = MagicMock()
    mock_decision.agent_name = "code_agent"
    mock_decision.reasoning = "Need computation"
    mock_decision.subtask_description = "Run analysis"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Analyze data",
            "search_results": ["some results"],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "code_agent"


def test_supervisor_finishes():
    mock_decision = MagicMock()
    mock_decision.agent_name = "FINISH"
    mock_decision.reasoning = "All done"

    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = mock_decision
    mock_llm.with_structured_output.return_value = mock_router

    with patch("src.graph.supervisor.create_llm", return_value=mock_llm):
        state: AgentState = {
            "messages": [],
            "task_description": "Done task",
            "search_results": ["results"],
            "code_results": ["output"],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = supervisor_node(state)
        assert result.goto == "__end__"


def test_route_after_worker_returns_supervisor():
    from src.graph.supervisor import route_after_worker

    state: AgentState = {
        "messages": [],
        "task_description": "test",
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
        "report_ready": False,
    }
    assert route_after_worker(state) == "supervisor"


def test_route_after_worker_ends_when_report_ready():
    from src.graph.supervisor import route_after_worker

    state: AgentState = {
        "messages": [],
        "task_description": "test",
        "search_results": [],
        "code_results": [],
        "final_report": "# Report",
        "retry_count": {},
        "errors": [],
        "report_ready": True,
    }
    assert route_after_worker(state) == "__end__"
```

**Step 2: Write writer agent tests**

Create `tests/unit/test_writer_agent.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import AgentState


@pytest.mark.asyncio
async def test_writer_agent_produces_report():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# Research Report\n\n## Summary\nFindings here."
    mock_llm.invoke.return_value = mock_response

    with patch("src.agents.writer_agent.create_agent_llm", return_value=mock_llm):
        from src.agents.writer_agent import WriterAgent

        agent = WriterAgent()
        state: AgentState = {
            "messages": [],
            "task_description": "Research Python",
            "search_results": ["Search result 1"],
            "code_results": ["Code output 1"],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = agent._run(state)
        assert "Research Report" in result


@pytest.mark.asyncio
async def test_writer_agent_node_routes_to_end():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# Final Report"
    mock_llm.invoke.return_value = mock_response

    with patch("src.agents.writer_agent.create_agent_llm", return_value=mock_llm):
        from src.agents.writer_agent import writer_agent_node

        state: AgentState = {
            "messages": [],
            "task_description": "Test task",
            "search_results": ["data"],
            "code_results": [],
            "final_report": "",
            "retry_count": {},
            "errors": [],
        }
        result = await writer_agent_node(state)
        assert result.goto == "__end__"
        assert result.update.get("report_ready") is True
        assert "# Final Report" in result.update.get("final_report", "")
```

**Step 3: Run all new tests**

Run: `pytest tests/unit/test_supervisor.py tests/unit/test_writer_agent.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/unit/test_supervisor.py tests/unit/test_writer_agent.py
git commit -m "test: add supervisor routing and writer agent unit tests"
```

---

### Task 12: Enhance existing tests for edge cases

**Files:**
- Modify: `tests/unit/test_code_executor.py:1-38`
- Modify: `tests/unit/test_llm_providers.py:1-29`
- Modify: `tests/unit/test_routing.py:1-57`

**Step 1: Add code executor edge case tests**

Append to `tests/unit/test_code_executor.py`:

```python
def test_code_size_limit():
    from src.tools.code_executor import execute_python_code, MAX_CODE_SIZE

    huge_code = "x = 1\n" * (MAX_CODE_SIZE + 1)
    with patch("src.tools.code_executor.subprocess.run"):
        result = execute_python_code.invoke({"code": huge_code})
    assert "Code too large" in result


@patch("src.tools.code_executor.subprocess.run")
def test_execute_code_timeout(mock_run):
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)

    from src.tools.code_executor import execute_python_code

    result = execute_python_code.invoke({"code": "while True: pass"})
    assert "timed out" in result
```

**Step 2: Add LLM provider fallback tests**

Append to `tests/unit/test_llm_providers.py`:

```python
def test_missing_api_key_raises():
    from src.config import Settings

    s = Settings(anthropic_api_key="", openai_api_key="", google_api_key="")
    with patch("src.llm.providers.settings", s):
        from src.llm.providers import create_llm
        with pytest.raises(ValueError, match="API key"):
            create_llm(provider="anthropic", model="test")


def test_fallback_creates_llm():
    from src.llm.providers import create_llm_with_fallback

    with (
        patch("src.llm.providers.create_llm", side_effect=lambda **kw: MagicMock()) as mock_create,
        patch("src.llm.providers.settings") as mock_settings,
    ):
        mock_settings.default_llm_provider = "anthropic"
        mock_settings.anthropic_api_key = "key"
        mock_settings.openai_api_key = "key"
        mock_settings.google_api_key = "key"

        result = create_llm_with_fallback()
        assert result is not None
```

**Step 3: Add routing edge case tests**

Append to `tests/unit/test_routing.py`:

```python
def test_route_empty_messages():
    state: AgentState = {"messages": []}
    from langgraph.graph import END
    result = _route_from_supervisor(state)
    assert result == END


def test_route_multiple_messages_picks_last():
    from langgraph.graph import END

    state: AgentState = {
        "messages": [
            MagicMock(content="Routing to search_agent: first search"),
            MagicMock(content="Routing to code_agent: then code"),
        ]
    }
    result = _route_from_supervisor(state)
    assert result == "code_agent"
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/unit/test_code_executor.py tests/unit/test_llm_providers.py tests/unit/test_routing.py
git commit -m "test: add edge case tests for executor, providers, and routing"
```

---

### Task 13: Final integration test and cleanup

**Files:**
- Modify: `tests/integration/test_workflow.py:1-43`
- Modify: `.env.example:1-42`

**Step 1: Update .env.example with new vars**

Append to `.env.example`:

```
# Streamlit
STREAMLIT_PORT=8501

# A2A Client (for distributed mode)
SUPERVISOR_URL=http://localhost:8001
```

**Step 2: Enhance integration test**

Update `tests/integration/test_workflow.py` to test both PG and Memory paths:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_workflow_builds_with_pg():
    pytest.importorskip("langgraph.checkpoint.postgres")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    mock_checkpointer = MagicMock()
    mock_checkpointer.setup = AsyncMock()

    with (
        patch.object(AsyncPostgresSaver, "from_conn_string", return_value=mock_checkpointer),
        patch("src.graph.workflow.search_agent_node", MagicMock()),
        patch("src.graph.workflow.code_agent_node", MagicMock()),
        patch("src.graph.workflow.writer_agent_node", MagicMock()),
        patch("src.graph.workflow.supervisor_node", MagicMock()),
    ):
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None


@pytest.mark.asyncio
async def test_workflow_builds_with_memory_fallback():
    with (
        patch("src.graph.workflow.search_agent_node", MagicMock()),
        patch("src.graph.workflow.code_agent_node", MagicMock()),
        patch("src.graph.workflow.writer_agent_node", MagicMock()),
        patch("src.graph.workflow.supervisor_node", MagicMock()),
        patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
            side_effect=Exception("PG down"),
        ),
    ):
        from src.graph.workflow import build_workflow

        graph = await build_workflow()
        assert graph is not None


@pytest.mark.asyncio
async def test_agent_state_structure():
    from src.graph.state import AgentState

    state: AgentState = {
        "messages": [],
        "task_description": "Research Python asyncio",
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
    }
    assert state["task_description"] == "Research Python asyncio"
```

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add .env.example tests/integration/test_workflow.py
git commit -m "test: enhance integration tests and update env example"
```
