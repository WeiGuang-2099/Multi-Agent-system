import argparse
import asyncio
import uuid

from src.config import settings

AGENT_PORTS = {
    "supervisor": settings.supervisor_port,
    "search": settings.search_agent_port,
    "code": settings.code_agent_port,
    "writer": settings.writer_agent_port,
}


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument(
        "--mode",
        choices=["supervisor", "distributed", "ui", "agent"],
        default="supervisor",
        help="Run mode: supervisor (single process), distributed (A2A servers), ui (Streamlit), agent (single A2A agent)",
    )
    parser.add_argument(
        "--agent",
        choices=["supervisor", "search", "code", "writer"],
        default="supervisor",
        help="Which agent to run in agent mode",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Research task to execute (supervisor mode only)",
    )
    args = parser.parse_args()

    if args.mode == "ui":
        _run_streamlit()
    elif args.mode == "distributed":
        _run_distributed(args.agent)
    elif args.mode == "agent":
        _run_single_agent(args.agent)
    elif args.mode == "supervisor":
        _run_supervisor(args.task)


def _run_streamlit():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py",
         "--server.port", str(settings.streamlit_port)],
        cwd=".",
    )


def _run_distributed(agent_name: str):
    from src.a2a.server import run_a2a_server

    if agent_name == "supervisor":
        asyncio.run(_run_supervisor_distributed())
    else:
        run_a2a_server(agent_name, AGENT_PORTS[agent_name])


async def _run_supervisor_distributed():
    from src.a2a.server import run_a2a_server

    port = AGENT_PORTS["supervisor"]
    run_a2a_server("supervisor", port)


def _run_single_agent(agent_name: str):
    from src.a2a.server import run_a2a_server

    port = AGENT_PORTS[agent_name]
    run_a2a_server(agent_name, port)


def _run_supervisor(task: str | None):
    if task:
        thread_id = str(uuid.uuid4())
        asyncio.run(_execute_and_print(task, thread_id))
    else:
        _run_interactive()


def _run_interactive():
    print("Multi-Agent Research Assistant — Interactive Mode")
    print("Type your research task and press Enter. Type 'exit' or 'quit' to stop.")
    print()

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

        thread_id = str(uuid.uuid4())
        asyncio.run(_execute_and_print(user_input, thread_id))


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


if __name__ == "__main__":
    main()
