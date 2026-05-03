# Full Fix Design — Multi-Agent Research Assistant

Date: 2026-05-03

## Scope

Fix all identified issues across 4 modules, ordered by dependency.

## Module 1: Configuration & Infrastructure

### config.py additions
- Add `streamlit_port: int = 8501`
- Add `supervisor_url: str = "http://localhost:8001"` (for Streamlit distributed mode)
- Add `agent_urls: dict` derived from port settings for A2A client resolution

### build_workflow() graceful degradation
- Try PostgreSQL `AsyncPostgresSaver` first
- On failure, fall back to `MemorySaver` from `langgraph.checkpoint.memory`
- Log warning when falling back
- Same pattern for sync variant `build_workflow_sync()`

### BaseAgent async fix
- Change `execute` to `async def execute`
- Replace `time.sleep(delay)` with `await asyncio.sleep(delay)`
- Change `_run` callers to `await agent.execute(state)` where needed
- Agent node wrappers (`search_agent_node`, etc.) remain sync Command factories but call `_run` via `asyncio.to_thread` only when needed by A2A executor; in LangGraph mode the graph handles async

### Cleanup
- Delete empty `src/tools/markdown.py`
- Remove `hasattr` hack in `main.py:53` for `streamlit_port`

## Module 2: A2A Distributed Mode

### New file: src/a2a/client.py
- `A2AClient` class using `httpx.AsyncClient`
- `async def send_task(agent_name: str, message: str) -> str` — sends A2A JSON-RPC `tasks/send` to target agent
- Resolves agent URL from `settings` (localhost:8002/8003/8004 by default)
- Handles timeout, retry, and error parsing

### New file: src/graph/distributed_workflow.py
- `build_distributed_workflow()` — same StateGraph structure but agent nodes call remote A2A agents via `A2AClient` instead of local Python functions
- Each distributed agent node: serialize state context → HTTP call → parse response → update state
- Supervisor node remains local (it's the orchestrator)

### Refactor SupervisorAgentExecutor
- In distributed mode, Supervisor runs the distributed workflow (HTTP calls to agents)
- In single-process mode, Supervisor runs the standard workflow (local calls)
- Mode selection via CLI flag or environment variable

### Agent Card URL from settings
- `agent_cards.py` reads port from `settings` to build URL dynamically

## Module 3: Interactive CLI & Streamlit UI

### Interactive CLI (main.py)
- When `--mode supervisor` without `--task`: enter REPL loop
- Read input with `input("You> ")`, support `exit`/`quit` to quit
- Maintain `thread_id` across turns for checkpoint continuity
- Each turn: invoke workflow with accumulated state
- Display report or errors after each turn

### Streamlit distributed mode
- Detect `SUPERVISOR_URL` env var
- If set: use `A2AClient` to call Supervisor instead of local `build_workflow()`
- If not set: keep current local workflow behavior

### Streamlit progress improvement
- Use `graph.astream_events()` to capture node transitions
- Display which agent is currently working based on event metadata
- Fallback: keep current static message if streaming unavailable

## Module 4: Test Coverage

### New test files
- `tests/unit/test_supervisor.py` — mock LLM structured output, verify routing decisions (FINISH, each agent)
- `tests/unit/test_writer_agent.py` — mock LLM, verify report output and `__end__` routing
- `tests/unit/test_distributed_workflow.py` — mock A2A client HTTP calls, verify state accumulation
- `tests/unit/test_interactive_cli.py` — mock workflow, test REPL input loop

### Enhanced existing tests
- `test_code_executor.py` — add test for MAX_CODE_SIZE rejection, timeout expired
- `test_llm_providers.py` — test `create_llm_with_fallback`, test missing API key error
- `test_routing.py` — test empty messages state, test `route_after_worker`

## Architecture After Fix

```
Single-process mode:
  CLI/UI → build_workflow() → [Supervisor → Agent nodes (local)] → report

Distributed mode:
  CLI/UI → build_distributed_workflow() → [Supervisor (local) → Agent nodes (HTTP/A2A)] → report
  Or:
  CLI/UI → A2AClient → Supervisor A2A Server → [HTTP → Agent A2A Servers] → report

Both modes share:
  Same AgentState, same agent implementations, same prompts
  Graceful PG → Memory checkpoint fallback
```
