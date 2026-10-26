"""Tests for the PROVIDED modules (data, db, llm, memory). These should already pass."""
import pytest

from fabpqagent import memory
from fabpqagent.db import QueryRejected, run_query
from fabpqagent.llm import ScriptedLLM, text_reply, tool_reply


def test_lots_table_shape(lots):
    assert len(lots) == 1567
    assert int(lots["failed"].sum()) == 104
    assert lots["ts"].is_monotonic_increasing


@pytest.mark.parametrize("sql", ["DELETE FROM lots", "DROP TABLE lots", "SELECT 1; DROP TABLE lots", "PRAGMA table_info(lots)"])
def test_run_query_rejects_non_select(lots, sql):
    with pytest.raises(QueryRejected):
        run_query(sql)


def test_run_query_select_and_cap(lots):
    df = run_query("SELECT lot_id, failed FROM lots", max_rows=10)
    assert len(df) == 10 and df.attrs.get("truncated")
    assert run_query("SELECT COUNT(*) AS n FROM lots WHERE failed = 1")["n"][0] == 104


def test_run_query_bad_sql_is_rejected_not_crash(lots):
    with pytest.raises(QueryRejected, match="SQL error"):
        run_query("SELECT nope FROM lots")


def test_scripted_llm_replays_and_records():
    llm = ScriptedLLM([tool_reply("query_lots", '{"sql": "SELECT 1"}'), text_reply("done")])
    first = llm.chat([{"role": "user", "content": "hi"}])
    assert first.tool_calls[0].name == "query_lots"
    assert first.as_message()["tool_calls"][0]["function"]["arguments"] == '{"sql": "SELECT 1"}'
    assert llm.chat([]).content == "done"
    assert llm.requests[0] == [{"role": "user", "content": "hi"}]


def test_memory_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "m.json")
    assert memory.load_cases() == []
    memory.save_case("why lot 2?", "s059 high", [2], ["s059"])
    cases = memory.load_cases()
    assert len(cases) == 1 and cases[0].lots == (2,) and cases[0].sensors == ("s059",)
