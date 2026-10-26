r"""DAY 2 — YOU WRITE THIS. Two-role agent: an INVESTIGATOR that uses tools, then a WRITER
that turns the evidence into a short report for an engineer.

Tests: tests/test_agent.py

Why two roles: the investigator's context fills
with raw tool output; the writer gets only the question + the evidence, so its prompt is short
(cheaper, faster) and it can't call tools or wander off.

Loop rules (you wrote these in CS329z P7, now do them with real tool calling):
- Stop when the model replies with no tool calls (it's done) -> stop_reason "done"
- Stop after max_turns model calls                           -> stop_reason "max_turns"
- Stop if total tokens would exceed token_budget (check BEFORE each call) -> "token_budget"
- Tool errors go back to the model as tool messages; never raise.
- Every tool result message needs {"role": "tool", "tool_call_id": <id>, "content": <observation>}

Memory: before investigating, load_cases() and put the most relevant past findings (same lot
ids or sensors mentioned in the question; max 3) into the investigator's system prompt.
After the writer finishes, save_case(): finding = first line of the report, lots = lot ids in the
question/evidence (regex r"lot\s*(\d+)"), sensors = ids like "s059" in the evidence/report.

The writer ALWAYS runs, even after max_turns or token_budget (it reports whatever evidence exists).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fabpqagent import memory, tools  # call tools.execute / memory.load_cases via the module: tests patch them
from fabpqagent.llm import LLM
from fabpqagent.tools import ToolContext

MAX_TURNS = 8
TOKEN_BUDGET = 40_000

INVESTIGATOR_PROMPT = """TODO: role, what the data is, which tool to use when, how to finish
(when you have enough evidence, reply WITHOUT tool calls and list the evidence as bullet points)."""

WRITER_PROMPT = """TODO: write a <=120-word report for a process engineer: conclusion first,
then evidence with numbers, then a recommended next check. Only use the evidence given."""


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: str
    observation: str


@dataclass
class Investigation:
    question: str
    report: str = ""
    evidence: str = ""               # investigator's final message
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""            # done | max_turns | token_budget
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    models: list[str] = field(default_factory=list)   # model that served each call


def investigate(question: str, llm: LLM, ctx: ToolContext, max_turns: int = MAX_TURNS,
                token_budget: int = TOKEN_BUDGET, use_memory: bool = True) -> Investigation:
    """Run investigator loop, then one writer call. Sum tokens/latency over ALL calls."""
    raise NotImplementedError
