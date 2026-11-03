"""Evaluate the agent on eval/golden.jsonl.

Tests: tests/test_evaluate.py      Run: uv run python -m fabpqagent.evaluate

Each golden line: {"id": "q01", "question": "...", "must_mention": ["s059", "fail"], "expected_tool": "top_features"}
must_mention holds facts checked by hand with SQL; "a|b" accepts either of two equivalent wordings.

Score per question (deterministic and cheap, unlike an LLM judge; see the README for what it
does not measure):
  - mention_recall = fraction of must_mention strings found in the report (case-insensitive)
  - used_expected_tool = expected_tool appears in the investigation's steps
  - tokens, latency_s, stop_reason
Write all rows to eval/results.json and print a summary: mean mention_recall, tool-use rate,
median latency, total tokens.

Running the suite twice with nothing changed shows the run-to-run noise floor.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from fabpqagent.agent import Investigation
from fabpqagent.data import ROOT

GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.json"


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    must_mention: tuple[str, ...]
    expected_tool: str


@dataclass(frozen=True)
class Score:
    id: str
    mention_recall: float
    used_expected_tool: bool
    tokens: int
    latency_s: float
    stop_reason: str


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [_golden_item(json.loads(line)) for line in lines]


def _golden_item(raw: dict) -> GoldenItem:
    return GoldenItem(raw["id"], raw["question"], tuple(raw["must_mention"]), raw["expected_tool"])


def score(item: GoldenItem, inv: Investigation) -> Score:
    report = inv.report.lower()
    found = sum(_mentioned(phrase, report) for phrase in item.must_mention)
    return Score(
        id=item.id,
        mention_recall=found / len(item.must_mention) if item.must_mention else 1.0,
        used_expected_tool=any(step.tool == item.expected_tool for step in inv.steps),
        tokens=inv.prompt_tokens + inv.completion_tokens,
        latency_s=inv.latency_s,
        stop_reason=inv.stop_reason,
    )


def _mentioned(phrase: str, report: str) -> bool:
    """'a|b' accepts either wording; only list wordings that mean the same thing."""
    return any(wording.strip().lower() in report for wording in phrase.split("|"))


def summarize(scores: list[Score]) -> dict[str, float]:
    """Keys: mean_mention_recall, tool_use_rate, median_latency_s, total_tokens, n."""
    if not scores:
        return {"mean_mention_recall": 0.0, "tool_use_rate": 0.0, "median_latency_s": 0.0, "total_tokens": 0, "n": 0}
    return {
        "mean_mention_recall": statistics.mean(s.mention_recall for s in scores),
        "tool_use_rate": statistics.mean(s.used_expected_tool for s in scores),
        "median_latency_s": statistics.median(s.latency_s for s in scores),
        "total_tokens": sum(s.tokens for s in scores),
        "n": len(scores),
    }


def run_suite(items: list[GoldenItem]) -> tuple[list[Score], list[dict]]:
    """Run the live agent on every item. One failing question is recorded, not fatal."""
    from fabpqagent import db, model
    from fabpqagent.agent import investigate
    from fabpqagent.llm import LLMError, OpenRouterLLM
    from fabpqagent.retrieval import NoteIndex, load_notes
    from fabpqagent.tools import ToolContext

    ctx = ToolContext(lots=db.load_lots(), model=model.load(model.MODEL_PATH), notes=NoteIndex(load_notes()))
    llm = OpenRouterLLM()
    scores, rows = [], []
    for item in items:
        try:
            inv = investigate(item.question, llm, ctx, use_memory=False)  # no leaks between items
        except LLMError as err:
            inv = Investigation(item.question, report=f"(LLM error: {err})", stop_reason="error")
        result = score(item, inv)
        scores.append(result)
        rows.append({**asdict(result), "question": item.question, "report": inv.report,
                     "tools": [step.tool for step in inv.steps], "models": inv.models})
        print(f"{item.id}: recall {result.mention_recall:.2f}, tool {result.used_expected_tool}, "
              f"{result.tokens} tokens, {result.latency_s:.1f}s, {result.stop_reason}", flush=True)
    return scores, rows


if __name__ == "__main__":
    scores, rows = run_suite(load_golden())
    RESULTS_PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps(summarize(scores), indent=1))
