"""DAY 3 — YOU WRITE THIS. Evaluate the agent on eval/golden.jsonl.

Tests: tests/test_evaluate.py      Run: uv run python -m fabpqagent.evaluate

Each golden line: {"id": "q01", "question": "...", "must_mention": ["s059", "fail"], "expected_tool": "top_features"}
Fill in must_mention for each question AFTER Day 1, from facts you checked yourself with SQL.

Score per question (keep it deterministic and cheap; this is the same lesson as LLMJudge:
know what your metric measures):
  - mention_recall = fraction of must_mention strings found in the report (case-insensitive)
  - used_expected_tool = expected_tool appears in the investigation's steps
  - tokens, latency_s, stop_reason
Write all rows to eval/results.json and print a summary: mean mention_recall, tool-use rate,
median latency, total tokens.

Stretch (only if time): run the suite twice and report how much mention_recall moves between
runs with nothing changed (the LLMJudge noise-floor idea, applied to an agent).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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
    raise NotImplementedError


def score(item: GoldenItem, inv: Investigation) -> Score:
    raise NotImplementedError


def summarize(scores: list[Score]) -> dict[str, float]:
    """Keys: mean_mention_recall, tool_use_rate, median_latency_s, total_tokens, n."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: build ToolContext (load lots, load model, NoteIndex), OpenRouterLLM(),
    # run investigate() for every golden item (use_memory=False so items don't leak into each other),
    # score, save RESULTS_PATH, print summary.
    raise NotImplementedError
