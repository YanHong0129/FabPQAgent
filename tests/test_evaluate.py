"""Day 3: evaluate.py"""
import json

import pytest

from fabpqagent.agent import Investigation, Step
from fabpqagent.evaluate import GOLDEN_PATH, GoldenItem, Score, load_golden, score, summarize


def test_load_golden(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"id": "q1", "question": "Q?", "must_mention": ["s059"], "expected_tool": "spc_check"}) + "\n\n")
    assert load_golden(path) == [GoldenItem("q1", "Q?", ("s059",), "spc_check")]


def test_shipped_golden_set_is_valid():
    items = load_golden(GOLDEN_PATH)
    assert len(items) >= 10
    assert len({i.id for i in items}) == len(items)


def test_score_counts_mentions_case_insensitive_and_tool_use():
    item = GoldenItem("q1", "Q?", ("S059", "fail", "pump"), "spc_check")
    inv = Investigation("Q?", report="s059 drift makes lots FAIL", steps=[Step("spc_check", "{}", "ok")],
                        stop_reason="done", prompt_tokens=100, completion_tokens=20, latency_s=2.0)
    s = score(item, inv)
    assert s.mention_recall == pytest.approx(2 / 3)
    assert s.used_expected_tool and s.tokens == 120 and s.stop_reason == "done"


def test_summarize():
    scores = [Score("a", 1.0, True, 100, 1.0, "done"), Score("b", 0.5, False, 300, 3.0, "done"),
              Score("c", 0.0, True, 200, 5.0, "max_turns")]
    s = summarize(scores)
    assert s["mean_mention_recall"] == pytest.approx(0.5)
    assert s["tool_use_rate"] == pytest.approx(2 / 3)
    assert s["median_latency_s"] == pytest.approx(3.0)
    assert s["total_tokens"] == 600 and s["n"] == 3
