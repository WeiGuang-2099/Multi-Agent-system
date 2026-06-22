"""Reflection / Critic agent (P1.8).

After the Writer produces a report, the Critic evaluates it for:
- completeness (did it address the original task?)
- citation/structure quality
- logical consistency and accuracy red-flags

If the Critic's verdict is below threshold, control returns to the Writer
with concrete feedback for revision. Otherwise the report is accepted and the
workflow ends. The loop is bounded by `settings.critic_max_rounds` to prevent
infinite cycles.

This agent is only active when `settings.critic_enabled` is True, so the
default workflow behavior (writer -> end) is unchanged for existing users.
"""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.config import settings
from src.graph.state import AgentState
from src.llm.prompts import WRITER_AGENT_SYSTEM_PROMPT
from src.llm.providers import create_agent_llm

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a strict research-report critic. Evaluate the
draft report below against the original task. Return a JSON verdict:

{
  "score": <1-5 float>,
  "passes": <true if score >= 4.0 AND no critical issues>,
  "feedback": "<concrete, actionable revision notes, or 'LGTM' if passing>"
}

Evaluation criteria:
- Completeness: every part of the task addressed
- Evidence: claims supported by the gathered search/code results
- Structure: clear sections, citations, readable Markdown
- Accuracy: no obvious contradictions or hallucinations

Be demanding but fair. Do not invent requirements outside the original task.
"""

ACCEPT_THRESHOLD = 4.0


class CriticVerdict(BaseModel):
    score: float = Field(ge=1.0, le=5.0)
    passes: bool
    feedback: str = ""


class CriticAgent(BaseAgent):
    agent_name = "critic_agent"

    def _run(self, state: AgentState) -> str:
        llm = create_agent_llm("critic_agent").with_structured_output(CriticVerdict)

        task = state.get("task_description", "")
        report = state.get("final_report", "")
        search = state.get("search_results", [])
        code = state.get("code_results", [])

        context_parts = [f"## Original Task\n{task}\n", f"## Draft Report\n{report}\n"]
        if search:
            context_parts.append(
                "## Available Search Results (for fact-checking)\n"
                + "\n".join(f"- {s[:300]}" for s in search[:10])
            )
        if code:
            context_parts.append(
                "## Available Code Results (for fact-checking)\n"
                + "\n".join(f"- {c[:300]}" for c in code[:10])
            )

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(context_parts)),
        ]

        verdict: CriticVerdict = llm.invoke(messages)
        return json.dumps(
            {
                "score": verdict.score,
                "passes": bool(verdict.passes),
                "feedback": verdict.feedback,
            },
            ensure_ascii=False,
        )


async def critic_agent_node(state: AgentState) -> Command:
    """Evaluate the report; route back to writer if it fails review."""
    agent = CriticAgent()
    raw = agent._run(state)  # sync; critic is cheap, no retry wrapper needed

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {
            "score": 5.0,
            "passes": True,
            "feedback": "Critic parse error; accepting",
        }

    score = float(payload.get("score", 5.0))
    passes = bool(payload.get("passes", True))
    feedback = str(payload.get("feedback", ""))

    rounds = int(state.get("critic_rounds", 0)) + 1
    max_rounds = settings.critic_max_rounds

    logger.info(
        "critic verdict round=%s score=%s passes=%s", rounds, score, passes
    )

    # Acceptance: explicit pass OR score above threshold OR out of rounds.
    accept = passes or score >= ACCEPT_THRESHOLD or rounds >= max_rounds

    if accept:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=(
                            f"[critic_agent] accepted "
                            f"(score={score}, round={rounds}/{max_rounds})"
                        )
                    )
                ],
                "critic_feedback": "",
                "critic_rounds": rounds,
                "report_ready": True,
            },
            goto="__end__",
        )

    # Reject: send back to writer with feedback for revision.
    return Command(
        update={
            "messages": [
                AIMessage(
                    content=(
                        f"[critic_agent] needs revision "
                        f"(score={score}, round={rounds}/{max_rounds}): {feedback}"
                    )
                )
            ],
            "critic_feedback": feedback,
            "critic_rounds": rounds,
        },
        goto="writer_agent",
    )


def build_revision_prompt(state: AgentState) -> str:
    """Construct the instruction that tells the Writer *what* to revise.

    Used by the writer node when `critic_feedback` is present in state.
    """
    feedback = state.get("critic_feedback", "")
    if not feedback:
        return ""
    return (
        f"\n\n## Critic Feedback (please revise the previous draft to address):\n"
        f"{feedback}\n"
        "Rewrite the full report incorporating these changes. "
        f"Keep the same overall Markdown structure from: {WRITER_AGENT_SYSTEM_PROMPT}"
    )
