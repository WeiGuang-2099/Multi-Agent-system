# Evaluation Framework

This directory contains the evaluation suite for the Multi-Agent Research
Assistant. It quantifies system quality across two dimensions so the project
can demonstrate **engineering rigor** (not just "it works on a demo") — a
skill that most candidate portfolios lack.

## What it measures

| Metric | Method | Range |
|--------|--------|-------|
| **Routing accuracy** | Deterministic — compares observed `node_start` events against the dataset's `expected_agents` (order-agnostic subset match) | 0–100% |
| **Keyword coverage** | Deterministic — case-insensitive substring match of `expected_report_keywords` in the final report | 0–100% |
| **Report quality (judge)** | LLM-as-judge — a temperature-0 LLM scores the report 1–5 on completeness, accuracy, structure, citations | 1.0–5.0 |

## Why this matters

- **Routing accuracy** proves the supervisor's structured-output routing is
  reliable — the core claim of any multi-agent system.
- **LLM-as-judge** shows familiarity with modern evaluation methodology
  (RAGAS-style holistic scoring), which interviewers probe.
- The JSON report artifacts in `eval/results/` are concrete evidence you can
  cite in interviews ("routing accuracy: 90%, judge score: 4.3/5").

## Usage

```bash
# Run the full suite (writes a timestamped JSON report to eval/results/)
make eval
# or directly:
python -m eval.run_eval

# Use a custom dataset / output path
python -m eval.run_eval \
  --dataset eval/datasets/routing_eval.csv \
  --output eval/results/latest.json

# Print summary only, don't save
python -m eval.run_eval --no-save
```

## Dataset format

`eval/datasets/routing_eval.csv` — CSV with three columns:

| Column | Type | Example |
|--------|------|---------|
| `task` | string | `"Research Python asyncio patterns"` |
| `expected_agents` | JSON array | `["search_agent","writer_agent"]` |
| `expected_report_keywords` | JSON array | `["asyncio","performance"]` |

To extend coverage, add rows here. The arrays are parsed with `ast.literal_eval`
so they must be valid Python literals.

## How routing accuracy is computed

```
observed_agents = {nodes that emitted a node_start event during the run}
routing_correct = set(expected_agents).issubset(observed_agents)
```

Order-agnostic: we care that the *right* agents ran, not the sequence. The
supervisor may legitimately loop (search → code → search → writer), and
repeated visits are deduplicated before comparison.

## How the LLM judge works

`_judge_report()` sends the task + report to a temperature-0 LLM with a strict
JSON-only output contract:

```json
{"score": 4.5, "rationale": "Comprehensive but missing one citation."}
```

The parser is defensive: it extracts the outermost `{...}` block and degrades
gracefully to `(0.0, "judge failed: ...")` on any error, so a malformed judge
response never crashes the suite.

## Outputs

Each run writes `eval/results/eval-<UTC timestamp>.json`:

```json
{
  "started_at": "...",
  "finished_at": "...",
  "model": "gpt-4o",
  "summary": {
    "total": 10,
    "routing_accuracy": 0.9,
    "avg_keyword_coverage": 0.85,
    "avg_judge_score": 4.2,
    "failures": 1
  },
  "samples": [ { "task": "...", "actual_agents": [...], "judge_score": 4.5, ... } ]
}
```

Diff these between commits to prove improvements (or catch regressions).

## CI integration

A GitHub Actions workflow (`.github/workflows/eval.yml`) can run this suite on
every PR and comment the summary back. Token cost is the main constraint, so
the dataset is kept small (10 samples) and the workflow is gated behind a
label or manual trigger by default.