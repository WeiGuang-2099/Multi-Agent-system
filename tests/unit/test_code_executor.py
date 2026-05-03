from unittest.mock import MagicMock, patch

from src.tools.code_executor import _format_result


def test_format_result_success():
    result = _format_result(stdout="Hello World", stderr="", exit_code=0)
    assert "Exit Code: 0" in result
    assert "Hello World" in result
    assert "--- stdout ---" in result
    assert "--- stderr ---" in result


def test_format_result_error():
    result = _format_result(
        stdout="", stderr="NameError: name 'x' is not defined", exit_code=1
    )
    assert "Exit Code: 1" in result
    assert "NameError" in result


def test_format_result_timeout():
    result = _format_result(
        stdout="", stderr="Execution timed out after 30s", exit_code=-1
    )
    assert "Exit Code: -1" in result
    assert "timed out" in result


@patch("src.tools.code_executor.subprocess.run")
def test_execute_python_code_docker_not_found(mock_run):
    mock_run.side_effect = FileNotFoundError("Docker not found")

    from src.tools.code_executor import execute_python_code

    result = execute_python_code.invoke({"code": "print('hello')"})
    assert "Docker not available" in result


def test_code_size_limit():
    from src.tools.code_executor import MAX_CODE_SIZE

    huge_code = "x = 1\n" * (MAX_CODE_SIZE + 1)
    with patch("src.tools.code_executor.subprocess.run"):
        from src.tools.code_executor import execute_python_code
        result = execute_python_code.invoke({"code": huge_code})
    assert "Code too large" in result


@patch("src.tools.code_executor.subprocess.run")
def test_execute_code_timeout(mock_run):
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)

    from src.tools.code_executor import execute_python_code
    result = execute_python_code.invoke({"code": "while True: pass"})
    assert "timed out" in result
