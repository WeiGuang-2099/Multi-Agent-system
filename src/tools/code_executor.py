import logging
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)

MAX_CODE_SIZE = 50_000


@tool
def execute_python_code(code: str) -> str:
    """Execute Python code in a Docker sandbox. Returns stdout, stderr, and exit code."""
    if len(code) > MAX_CODE_SIZE:
        return _format_result(
            stdout="",
            stderr=f"Code too large ({len(code)} bytes). Maximum allowed: {MAX_CODE_SIZE} bytes.",
            exit_code=-1,
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="sandbox_", delete=False
    ) as f:
        f.write(code)
        code_path = f.name

    try:
        os.chmod(code_path, stat.S_IRUSR | stat.S_IRGRP)
    except OSError as e:
        logger.warning(f"Failed to set file permissions on {code_path}: {e}")

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--memory", settings.sandbox_memory_limit,
                "--cpus", settings.sandbox_cpu_limit,
                "--network", "none",
                "--cap-drop=ALL",
                "--pids-limit", "50",
                "--memory-swap", settings.sandbox_memory_limit,
                "--security-opt", "no-new-privileges",
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{code_path}:/sandbox/code.py:ro",
                "research-assistant-sandbox:latest",
            ],
            capture_output=True,
            text=True,
            timeout=settings.sandbox_timeout,
        )

        return _format_result(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            stdout="",
            stderr=f"Execution timed out after {settings.sandbox_timeout}s",
            exit_code=-1,
        )
    except FileNotFoundError:
        return _format_result(
            stdout="",
            stderr="Docker not available. Please ensure Docker is installed and running.",
            exit_code=-1,
        )
    finally:
        try:
            Path(code_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {code_path}: {e}")


def _format_result(stdout: str, stderr: str, exit_code: int) -> str:
    return (
        f"Exit Code: {exit_code}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}"
    )
