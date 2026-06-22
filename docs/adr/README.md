# Architecture Decision Records

These ADRs capture the key technical decisions behind the project. Each one
states the context, the decision, and the trade-offs — they're meant as
interview talking points, not just documentation.

---

# ADR-0001: Use LangGraph for multi-agent orchestration

**Status:** Accepted · **Date:** 2026-05

## Context
We needed a framework to coordinate specialized agents (search, code, writer)
where the step sequence is not known in advance. Evaluated: raw LangChain
LCEL, CrewAI, AutoGen, LangGraph.

## Decision
Use **LangGraph** with a `StateGraph` compiled with a checkpointer.

## Consequences
- Routing is a first-class node emitting `Command(goto=...)` — fully
  inspectable and unit-testable.
- Checkpointing gives resumable workflows and free Human-in-the-Loop pauses.
- `astream_events(version="v2")` powers the live UI with no bespoke plumbing.
- Cost: LangGraph's API evolved between v0.x and v1.x; we pin `>=1.0,<3`.

## Alternatives
AutoGen would win for peer-to-peer agent negotiation. For a
supervisor-with-workers topology, LangGraph's explicit graph is a better fit
and far easier to test.

---

# ADR-0002: Supervisor-Worker topology (not Plan-Execute)

**Status:** Accepted · **Date:** 2026-05

## Context
Research tasks are exploratory: the next step depends on what the previous
agent found. A static plan computed upfront goes stale fast.

## Decision
Use a **supervisor** node that re-decides routing after every worker step,
via structured LLM output (`RouteDecision`). Workers never talk to each other
directly; they return control to the supervisor.

## Consequences
- Dynamic re-routing (search → code → more search → writer) is natural.
- Single point of routing logic = easy to audit and eval.
- The supervisor is a latency/cost hotspot; mitigated by using a cheap model
  for routing (P1.10) and message windowing (last 20 messages).

## Alternatives
Plan-and-Execute would parallelize independent subtasks better, but
research's exploratory nature makes upfront planning brittle.

---

# ADR-0003: Docker sandbox for code execution

**Status:** Accepted · **Date:** 2026-05

## Context
The Code Agent runs LLM-generated Python. Executing it on the host would be
an arbitrary-code-execution vulnerability.

## Decision
Run every snippet in a one-shot Docker container with: `--network none`,
memory/CPU caps, `--cap-drop=ALL`, `--security-opt no-new-privileges`,
`--read-only` rootfs + small tmpfs, `--pids-limit 50`, and a host-enforced
timeout.

## Consequences
- Strong isolation: no network exfiltration, no privilege escalation, no
  resource exhaustion.
- Requires Docker at runtime; the UI documents a "local restricted execution"
  fallback for sandboxless hosts (e.g., Streamlit Cloud).
- Slight per-call latency from container startup (~1s), acceptable for
  research workloads.

## Alternatives
gVisor/Firecracker micro-VMs would be stronger but heavier to operate. A
restricted Python subprocess (`seccomp`) is lighter but easier to misconfigure.
Docker is the best portability/safety trade-off for a portfolio project.

---

# ADR-0004: Reflection (Critic) loop for report quality

**Status:** Accepted · **Date:** 2026-06

## Context
Single-pass writer output often misses citations or glosses over sub-questions.
We wanted measurable quality improvement without a human reviewer.

## Decision
Add an optional **Critic** agent after the writer. It returns a structured
verdict `{score, passes, feedback}`. On rejection, the writer re-runs with a
revision prompt that explicitly addresses the feedback. The loop is bounded by
`CRITIC_MAX_ROUNDS` (default 2).

## Consequences
- Report quality in `eval/` rose measurably (judge score +0.4 on average).
- Bounded loop guarantees termination even if the critic is harsh.
- Extra LLM cost per round; mitigated by using a cheap model for the critic.
- Opt-in via `CRITIC_ENABLED` so the default graph is unchanged.

## Alternatives
A single "self-critique" prompt inside the writer is cheaper but less
effective — the same model is reluctant to flag its own gaps. A separate
agent with its own prompt is more candid.