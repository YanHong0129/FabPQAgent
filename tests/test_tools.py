"""Tests for tools.py (needs the trained model and the note index)."""
import json
import re

import pytest

from fabpqagent.retrieval import NoteIndex, load_notes
from fabpqagent.tools import MAX_OBSERVATION_CHARS, TOOL_SCHEMAS, ToolContext, execute

EXPECTED_TOOLS = {"query_lots", "predict_fail", "top_features", "spc_check", "search_notes"}


@pytest.fixture(scope="module")
def ctx(lots, trained):
    return ToolContext(lots=lots, model=trained, notes=NoteIndex(load_notes()))


def test_five_valid_schemas():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == EXPECTED_TOOLS
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert s["function"]["description"]
        assert s["function"]["parameters"]["type"] == "object"
        json.dumps(s)  # must be JSON-serialisable


@pytest.mark.parametrize("name,args", [
    ("query_lots", "not json"),
    ("no_such_tool", "{}"),
    ("predict_fail", "{}"),
    ("predict_fail", '{"lot_id": "two"}'),
    ("predict_fail", '{"lot_id": 99999}'),
    ("spc_check", '{"sensor": "s999"}'),
    ("query_lots", '{"sql": "DELETE FROM lots"}'),
])
def test_bad_calls_become_error_observations(ctx, name, args):
    out = execute(name, args, ctx)
    assert isinstance(out, str) and out.startswith("Error")


def test_query_lots_returns_csv(ctx):
    out = execute("query_lots", '{"sql": "SELECT failed, COUNT(*) AS n FROM lots GROUP BY failed"}', ctx)
    assert "failed" in out and "104" in out


def test_predict_fail_mentions_probability_and_actual(ctx):
    out = execute("predict_fail", '{"lot_id": 2}', ctx).lower()
    assert "prob" in out and ("actual" in out or "label" in out)


def test_top_features_lists_sensors(ctx):
    out = execute("top_features", '{"lot_id": 2, "k": 3}', ctx)
    assert len(set(re.findall(r"\bs\d{3}\b", out))) == 3


def test_spc_check_reports_limits(ctx):
    out = execute("spc_check", '{"sensor": "s059"}', ctx).lower()
    assert "ucl" in out and "lcl" in out


def test_search_notes(ctx):
    out = execute("search_notes", '{"query": "s059 drift"}', ctx)
    assert "s059" in out


def test_observations_are_capped(ctx):
    out = execute("query_lots", '{"sql": "SELECT * FROM lots"}', ctx)
    assert len(out) <= MAX_OBSERVATION_CHARS + 200
    assert "truncat" in out.lower()
