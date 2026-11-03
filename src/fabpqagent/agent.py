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

import re
from dataclasses import dataclass, field
from typing import Any

from fabpqagent import memory, tools  # call tools.execute / memory.load_cases via the module: tests patch them
from fabpqagent.llm import LLM, LLMReply
from fabpqagent.tools import ToolContext

MAX_TURNS = 8
TOKEN_BUDGET = 40_000
MAX_RECALLED_CASES = 3
LOT_PATTERN = re.compile(r"lot\s*(\d+)", re.IGNORECASE)
SENSOR_PATTERN = re.compile(r"\bs\d{3}\b", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")  # whitespace that follows . ! or ?

INVESTIGATOR_PROMPT = """You are a product-quality investigator in a semiconductor fab. You answer a
process engineer's question by gathering evidence with tools.

Data: the UCI SECOM dataset, 1,567 production lots (lot_id 0-1566), each with a timestamp, a pass/fail
label and 590 anonymised sensor readings (s000-s589).

Which tool to use:
- query_lots: counts, fail rates, anything over time, or finding lots (SQL on table lots).
- predict_fail: the fail classifier's probability for one lot.
- top_features: which of the model's important sensors look most unusual on one lot.
- spc_check: whether one sensor is in statistical control (limits from a baseline period).
- search_notes: troubleshooting advice. Pass keywords such as "s510 high", not a full sentence.

Rules:
- Every number you report must come from a tool result. Never guess values.
- If a tool returns "Error: ...", read the hint, fix the arguments and try again.
- Call several tools in one turn when they do not depend on each other.
- A sensor is unusual on a lot only if |z| >= 2. A sensor "in common" across lots must be unusual
  on most of them; a sensor that appears on every list with a small z is normal, not common.
- For "recent" or trend questions, group by week or month (not by day), also compute the overall
  fail rate with SELECT COUNT(*) AS lots, SUM(failed) AS fails, AVG(failed) AS fail_rate
  FROM lots, and state both numbers in the evidence, e.g.
  "recent X% vs overall Y%".
- An out-of-control sensor is not a cause by itself. Call a sensor a likely cause only if it is
  also unusual (top_features) on the failing lots of that period; otherwise say the cause is not
  established.
- Stop as soon as you can answer: reply WITHOUT tool calls and list the evidence as short bullet
  points, each with its numbers and the tool it came from."""

WRITER_PROMPT = """You write short reports for a process engineer in a semiconductor fab.
Using ONLY the evidence you are given, write at most 120 words of plain text, no title:
1. First line: one sentence that answers the question (the conclusion).
2. Then the evidence, with its numbers.
3. Last: the recommended next check, plus any corrective action the evidence names (e.g. recalibrate
   or replace a part). If the evidence names none, suggest one next check that follows from it;
   never write that none was given.
If the evidence is incomplete or does not answer the question, say so plainly instead of guessing.
Do not call anything a cause unless the evidence links it to the failing lots themselves."""


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
    inv = Investigation(question)
    past = _recall(question) if use_memory else []
    _run_investigator(inv, llm, ctx, _investigator_prompt(past), max_turns, token_budget)
    inv.report = _write_report(inv, llm)
    if use_memory:
        _remember(inv)
    return inv


def _run_investigator(inv: Investigation, llm: LLM, ctx: ToolContext, system_prompt: str,
                      max_turns: int, token_budget: int) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": inv.question},
    ]
    inv.stop_reason = "max_turns"  # overwritten if the loop ends any other way
    for _ in range(max_turns):
        if inv.prompt_tokens + inv.completion_tokens >= token_budget:
            inv.stop_reason = "token_budget"
            break
        reply = llm.chat(messages, tools=tools.TOOL_SCHEMAS)
        _add_usage(inv, reply)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            inv.stop_reason = "done"
            inv.evidence = reply.content
            break
        for call in reply.tool_calls:
            observation = tools.execute(call.name, call.arguments, ctx)
            inv.steps.append(Step(call.name, call.arguments, observation))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
    if not inv.evidence:
        # stopped early (or the model ended with no text): hand the writer the raw tool results
        inv.evidence = "\n".join(f"- {s.tool}({s.arguments}): {s.observation}" for s in inv.steps)


def _write_report(inv: Investigation, llm: LLM) -> str:
    note = "" if inv.stop_reason == "done" else (
        f"Note: the investigation stopped early ({inv.stop_reason}), so this evidence may be incomplete.\n")
    messages = [
        {"role": "system", "content": WRITER_PROMPT},
        {"role": "user", "content": f"Question: {inv.question}\n\n{note}Evidence:\n{inv.evidence or '(none)'}"},
    ]
    reply = llm.chat(messages)  # no tools: the writer can only write
    _add_usage(inv, reply)
    return reply.content.strip()


def _add_usage(inv: Investigation, reply: LLMReply) -> None:
    inv.prompt_tokens += reply.prompt_tokens
    inv.completion_tokens += reply.completion_tokens
    inv.latency_s += reply.latency_s
    inv.models.append(reply.model)


def _recall(question: str) -> list[memory.Case]:
    """Most recent past cases that share a lot id or a sensor with the question."""
    lots, sensors = set(_lot_ids(question)), set(_sensor_ids(question))
    relevant = [c for c in memory.load_cases() if lots & set(c.lots) or sensors & set(c.sensors)]
    return relevant[-MAX_RECALLED_CASES:]


def _investigator_prompt(past: list[memory.Case]) -> str:
    if not past:
        return INVESTIGATOR_PROMPT
    lines = "\n".join(f"- Q: {c.question} -> {c.finding}" for c in past)
    return f"{INVESTIGATOR_PROMPT}\n\nPast findings (leads only; verify with tools before relying on them):\n{lines}"


def _remember(inv: Investigation) -> None:
    if inv.stop_reason != "done" or not inv.report:
        return  # an unfinished investigation's guess must not become a "past finding"
    memory.save_case(
        question=inv.question,
        finding=SENTENCE_END.split(inv.report.splitlines()[0], maxsplit=1)[0],
        lots=_lot_ids(f"{inv.question} {inv.evidence}"),
        sensors=_sensor_ids(f"{inv.evidence} {inv.report}"),
    )


def _lot_ids(text: str) -> list[int]:
    return sorted({int(n) for n in LOT_PATTERN.findall(text)})


def _sensor_ids(text: str) -> list[str]:
    return sorted({s.lower() for s in SENSOR_PATTERN.findall(text)})
