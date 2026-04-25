# Multi-Agent Research Assistant

A production-ready Multi-Agent research assistant built with **LangGraph** and the **A2A (Agent-to-Agent) protocol**. It automatically decomposes research tasks, coordinates specialized agents for web search, code execution, and report writing, and produces structured Markdown reports.

## Architecture

```
                  Streamlit UI (port 8501)
                        |
                        v
    +--------- SUPERVISOR (port 8001) ---------+
    |    LangGraph StateGraph + PG checkpt     |
    |    Routes via structured LLM output      |
    +-----+------------+------------+-----------+
          |            |            |
          v            v            v
   SEARCH_AGENT   CODE_AGENT   WRITER_AGENT
   (port 8002)    (port 8003)   (port 8004)
       |              |
       v              v
   Tavily API    Docker Sandbox
                  (no network,
                   512m RAM,
                   1 CPU)
```

The system uses a **supervisor-based orchestration pattern**:

- **Supervisor** - Analyzes tasks, decomposes them into subtasks, and routes each subtask to the appropriate agent using structured LLM output
- **Search Agent** - Performs web searches via Tavily API and summarizes results
- **Code Agent** - Generates and executes Python code in an isolated Docker sandbox
- **Writer Agent** - Synthesizes all collected data into a structured Markdown report

## Features

- **Web Search** - Tavily API integration for authoritative web research
- **Sandboxed Code Execution** - Python code runs in an isolated Docker container with no network, resource limits, and timeout enforcement
- **Structured Report Generation** - Produces professional Markdown reports with citations
- **Multi-Provider LLM Support** - Anthropic (Claude), OpenAI (GPT-4), Google (Gemini)
- **Dual Deployment Modes** - Single-process (LangGraph) or distributed (A2A HTTP servers)
- **PostgreSQL Checkpointing** - Persistent state via LangGraph checkpointing for resumable workflows
- **Streamlit Web UI** - Interactive chat interface with provider/model selection
- **Retry & Error Handling** - Automatic retries per agent with configurable limits

## Project Structure

```
.
├── docker-compose.yml           # Full stack orchestration
├── Dockerfile                   # Main app container
├── pyproject.toml               # Python package & dependencies
├── .env.example                 # Environment variable template
├── sandbox/
│   └── Dockerfile.sandbox       # Isolated code execution sandbox
├── src/
│   ├── main.py                  # CLI entry point (4 run modes)
│   ├── config.py                # Pydantic Settings (env vars, API keys)
│   ├── agents/
│   │   ├── base.py              # BaseAgent ABC with retry logic
│   │   ├── search_agent.py      # Web search via Tavily
│   │   ├── code_agent.py        # Code gen + sandbox execution
│   │   └── writer_agent.py      # Report synthesis
│   ├── graph/
│   │   ├── state.py             # AgentState, RouteDecision models
│   │   ├── supervisor.py        # Supervisor node + routing logic
│   │   └── workflow.py          # LangGraph StateGraph assembly
│   ├── llm/
│   │   ├── providers.py         # LLM factory (Anthropic/OpenAI/Google)
│   │   └── prompts.py           # System prompts for all agents
│   ├── tools/
│   │   ├── search.py            # Tavily web search tool
│   │   └── code_executor.py     # Docker sandbox code execution
│   ├── a2a/
│   │   ├── agent_cards.py       # A2A AgentCard definitions
│   │   ├── executor.py          # A2A AgentExecutor wrappers
│   │   └── server.py            # A2A FastAPI server
│   └── ui/
│       └── streamlit_app.py     # Streamlit web interface
└── tests/
    ├── unit/                    # Unit tests (routing, search, LLM, executor)
    └── integration/             # Integration tests (workflow, A2A)
```

## Quick Start

### Prerequisites

- Python >= 3.11
- Docker (for code sandbox execution)
- API keys for LLM provider and Tavily search

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys in `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-xxx       # or OPENAI_API_KEY / GOOGLE_API_KEY
TAVILY_API_KEY=tvly-xxx
DEFAULT_LLM_PROVIDER=anthropic     # anthropic | openai | google
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Build the Sandbox Image

```bash
cd sandbox && docker build -t research-assistant-sandbox:latest . && cd ..
```

### 4. Run

**CLI mode** (single-process, requires PostgreSQL):
```bash
# Start PostgreSQL
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_DB=research_assistant \
  -e POSTGRES_USER=agent \
  -e POSTGRES_PASSWORD=agent_password \
  postgres:16-alpine

# Run a research task
python -m src.main --mode supervisor --task "Research Python asyncio performance patterns"
```

**Streamlit UI mode**:
```bash
python -m src.main --mode ui
```

**Distributed mode** (A2A, via docker-compose):
```bash
docker-compose up --build
```

## Run Modes

| Mode | Command | Description |
|------|---------|-------------|
| `supervisor` | `--mode supervisor --task "..."` | Single-process LangGraph workflow |
| `distributed` | `--mode distributed --agent <name>` | A2A server for one agent |
| `agent` | `--mode agent --agent <name>` | Standalone A2A agent server |
| `ui` | `--mode ui` | Streamlit web interface |

## Data Flow

```
User Task --> Supervisor analyzes & routes
  |
  +--> Search Agent: web search via Tavily, results accumulate in state
  |
  +--> Code Agent: generates Python, executes in Docker sandbox
  |
  +--> (loop back to supervisor for next subtask)
  |
  +--> Writer Agent: synthesizes all results into Markdown report
  |
  v
Final Report returned to user
```

The supervisor can loop through agents multiple times (search -> code -> more search -> writer) until it determines all tasks are complete.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v
```

## Configuration

All configuration is managed via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_PROVIDER` | `anthropic` | LLM provider (anthropic/openai/google) |
| `DEFAULT_LLM_MODEL` | `claude-sonnet-4-20250514` | Default model name |
| `SEARCH_AGENT_MODEL` | - | Per-agent model override |
| `CODE_AGENT_MODEL` | - | Per-agent model override |
| `WRITER_AGENT_MODEL` | - | Per-agent model override |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Docker sandbox memory limit |
| `SANDBOX_CPU_LIMIT` | `1` | Docker sandbox CPU limit |
| `SANDBOX_TIMEOUT` | `30` | Code execution timeout (seconds) |
| `MAX_RETRIES` | `3` | Max retry attempts per agent |

## License

MIT
