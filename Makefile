# Multi-Agent Research Assistant — common developer commands.
# Usage: `make <target>`. Windows users: run under WSL/Git Bash or invoke the
# underlying commands directly.

PYTHON ?= python
PORT ?= 8501

.PHONY: help install dev lint format test test-unit test-integration eval \
        run-ui run-cli docker-build docker-up docker-down clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

dev: install  ## Install everything including RAG extras
	$(PYTHON) -m pip install -e ".[dev,rag]"

lint:  ## Lint with ruff
	ruff check src tests eval

format:  ## Auto-format with ruff
	ruff format src tests eval
	ruff check --fix src tests eval

test: test-unit test-integration  ## Run all tests

test-unit:  ## Run unit tests only
	$(PYTHON) -m pytest tests/unit -v

test-integration:  ## Run integration tests only
	$(PYTHON) -m pytest tests/integration -v

eval:  ## Run the evaluation suite (requires API keys)
	$(PYTHON) -m eval.run_eval

run-ui:  ## Launch the Streamlit UI
	$(PYTHON) -m src.main --mode ui

run-cli:  ## Run a single research task via CLI (override TASK=...)
	$(PYTHON) -m src.main --mode supervisor --task "$(TASK)"

docker-build:  ## Build the Docker images
	docker build -t multi-agent-research-assistant:latest .
	cd sandbox && docker build -t research-assistant-sandbox:latest .

docker-up:  ## Start the full distributed stack
	docker compose up --build

docker-down:  ## Stop the distributed stack
	docker compose down

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +