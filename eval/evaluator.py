"""Evaluation framework for the Multi-Agent Research Assistant (P0.3).

This module evaluates two dimensions of system quality:

1. **Routing accuracy**: Did the supervisor route to the expected agents
   (in any order, deduplicated)? Computed deterministically by comparing the
   observed `node_start` events against the dataset's `expected_agents`.

2. **Report quality**: Using LLM-as-judge, score the final report on:
   - completeness (did it address the task?)
   - keyword coverage (did it mention the expected terms?)
   - citation/structure quality
   Returns a 1-5 score with rationale.

Usage:

    from eval.evaluator import run_evaluation
    results = await run_evaluation(dataset_path="eval/datasets/routing_eval.csv")
    print(results.summary())

The evaluator is hermetic (no network beyond LLM/Tavily API calls the graph
already makes) and writes a JSON report to `eval/results/`.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.graph.callbacks import interpret_event

EVAL_PROJECT = "multi-agent-eval"


@dataclass
class SampleResult:
    task: str
    expected_agents: list[str]
    actual_agents: list[str]
    expected_keywords: list[str]
    report: str
    routing_correct: bool
    keyword_hits: list[str]
    keyword_misses: list[str]
    judge_score: float
    judge_rationale: str
    error: str | None = None


@dataclass
class EvalSummary:
    total: int
    routing_accuracy: float
    avg_keyword_coverage: float
    avg_judge_score: float
    failures: int


@dataclass
class EvalReport:
    started_at: str
    finished_at: str
    model: str
    summary: EvalSummary
    samples: list[SampleResult] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_csv_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse a CSV row, turning the stringified JSON lists into Python lists."""
    import ast

    return {
        "task": row["task"],
        "expected_agents": ast.literal_eval(row["expected_agents"]),
        "expected_report_keywords": ast.literal_eval(row["expected_report_keywords"]),
    }


def _load_dataset(path: str | Path) -> list[dict[str, Any]]:
    import csv

    with Path(path).open(encoding="utf-8") as f:
        return [_parse_csv_row(r) for r in csv.DictReader(f)]


def _extract_agents(events: list[dict[str, Any]]) -> list[str]:
    """Return the deduplicated, order-preserved list of agent nodes invoked."""
    seen: list[str] = []
    for event in events:
        for status in interpret_event(event):
            if status.kind == "node_start" and status.node not in seen:
                seen.append(status.node)
    return seen


def _check_keywords(report: str, expected: list[str]) -> tuple[list[str], list[str]]:
    """Return (hits, misses) based on case-insensitive substring matching."""
    low = report.lower()
    hits = [k for k in expected if k.lower() in low]
    misses = [k for k in expected if k.lower() not in low]
    return hits, misses


async def _judge_report(task: str, report: str, llm) -> tuple[float, str]:
    """Use an LLM as judge to score the report 1-5 with rationale.

    Returns (score, rationale). Falls back to (0.0, "judge failed") on error.
    """
    if not report.strip():
        return 0.0, "Empty report"

    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"""You are a strict evaluator for a research assistant. Score the
following report on a 1-5 scale based on: completeness, accuracy of addressing
the task, structure, and citation quality.

Task: {task}

Report:
---
{report[:6000]}
---

Respond with ONLY valid JSON:
{{"score": <1-5 float>, "rationale": "<one sentence>"}}
"""
    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are an expert evaluator."),
                HumanMessage(content=prompt),
            ]
        )
        # Robust JSON extraction (LLMs sometimes wrap in ```json).
        text = response.content if hasattr(response, "content") else str(response)
        if isinstance(text, list):
            text = " ".join(str(x) for x in text)
        text = str(text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return 0.0, f"Judge returned non-JSON: {text[:100]}"
        payload = json.loads(text[start : end + 1])
        return float(payload.get("score", 0.0)), str(
            payload.get("rationale", "")
        )
    except Exception as e:  # noqa: BLE001
        return 0.0, f"judge failed: {e}"


async def _run_one(sample: dict[str, Any], graph, judge_llm) -> SampleResult:
    task = sample["task"]
    expected_agents = sample["expected_agents"]
    expected_keywords = sample["expected_report_keywords"]

    config = {
        "configurable": {"thread_id": f"eval-{uuid.uuid4()}"},
        "tags": ["eval"],
    }
    inputs = {
        "messages": [],
        "task_description": task,
        "search_results": [],
        "code_results": [],
        "final_report": "",
        "retry_count": {},
        "errors": [],
    }

    events: list[dict[str, Any]] = []
    final_state: dict = {}
    error: str | None = None

    try:
        async for event in graph.astream_events(inputs, config, version="v2"):
            events.append(event)
    except Exception as e:  # noqa: BLE001
        error = f"streaming error: {e}"

    try:
        state_obj = await graph.aget_state(config)
        final_state = getattr(state_obj, "values", state_obj) or {}
    except Exception as e:  # noqa: BLE001
        if error is None:
            error = f"state fetch error: {e}"

    actual_agents = _extract_agents(events)
    report = final_state.get("final_report", "") or ""

    # Routing correctness: expected agents must all appear (order-agnostic).
    actual_set = set(actual_agents)
    expected_set = set(expected_agents)
    routing_correct = expected_set.issubset(actual_set)

    hits, misses = _check_keywords(report, expected_keywords)
    score, rationale = await _judge_report(task, report, judge_llm)

    return SampleResult(
        task=task,
        expected_agents=expected_agents,
        actual_agents=actual_agents,
        expected_keywords=expected_keywords,
        report=report,
        routing_correct=routing_correct,
        keyword_hits=hits,
        keyword_misses=misses,
        judge_score=score,
        judge_rationale=rationale,
        error=error,
    )


async def run_evaluation(
    dataset_path: str | Path = "eval/datasets/routing_eval.csv",
    output_path: str | Path | None = None,
) -> EvalReport:
    """Run the full evaluation suite and return an EvalReport.

    Requires a running graph (with API keys configured). Safe to call from
    CLI via `python -m eval.run_eval`.
    """
    from src.config import settings
    from src.graph.workflow import build_workflow
    from src.llm.providers import create_llm

    dataset = _load_dataset(dataset_path)
    graph = await build_workflow()
    judge_llm = create_llm(temperature=0.0)

    started = datetime.now(UTC).isoformat()
    samples: list[SampleResult] = []
    for sample in dataset:
        # Run samples sequentially to avoid cross-thread state collisions.
        result = await _run_one(sample, graph, judge_llm)
        samples.append(result)
        print(
            f"  [{'PASS' if result.routing_correct else 'FAIL'}] "
            f"routing={result.actual_agents} | score={result.judge_score:.1f} | "
            f"{result.task[:60]}"
        )
    finished = datetime.now(UTC).isoformat()

    total = len(samples)
    routing_accuracy = (
        sum(1 for s in samples if s.routing_correct) / total if total else 0.0
    )
    avg_keyword_coverage = (
        sum(
            len(s.keyword_hits) / max(1, len(s.expected_keywords))
            for s in samples
        )
        / total
        if total
        else 0.0
    )
    avg_judge_score = sum(s.judge_score for s in samples) / total if total else 0.0
    failures = sum(1 for s in samples if s.error or not s.routing_correct)

    report = EvalReport(
        started_at=started,
        finished_at=finished,
        model=settings.default_llm_model,
        summary=EvalSummary(
            total=total,
            routing_accuracy=round(routing_accuracy, 4),
            avg_keyword_coverage=round(avg_keyword_coverage, 4),
            avg_judge_score=round(avg_judge_score, 2),
            failures=failures,
        ),
        samples=samples,
    )

    if output_path:
        report.save(output_path)
    return report


def print_summary(report: EvalReport) -> None:
    s = report.summary
    print("\n" + "=" * 60)
    print(f"Evaluation Summary (n={s.total}, model={report.model})")
    print("=" * 60)
    print(f"  Routing accuracy:      {s.routing_accuracy:.1%}")
    print(f"  Avg keyword coverage:  {s.avg_keyword_coverage:.1%}")
    print(f"  Avg judge score:       {s.avg_judge_score:.2f} / 5.0")
    print(f"  Failures:              {s.failures}")
    print("=" * 60 + "\n")
