"""Day 2: agent.py. No network and no model needed: tools.execute is patched, the LLM is scripted."""
import pytest

from fabpqagent import agent, memory, tools
from fabpqagent.llm import LLMReply, ScriptedLLM, ToolCall, text_reply, tool_reply


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch, tmp_path):
    calls = []

    def fake_execute(name, arguments, ctx):
        calls.append((name, arguments))
        return "Error: lot 99999 does not exist" if "99999" in arguments else f"result of {name}"

    monkeypatch.setattr(tools, "execute", fake_execute)
    monkeypatch.setattr(memory, "MEMORY_PATH", tmp_path / "memory.json")
    return calls


def with_usage(reply: LLMReply, p: int, c: int) -> LLMReply:
    return LLMReply(reply.content, reply.tool_calls, reply.model, p, c, 0.5)


def test_tool_then_done_then_writer(fake_tools):
    llm = ScriptedLLM([
        with_usage(tool_reply("predict_fail", '{"lot_id": 2}', "c1"), 100, 10),
        with_usage(text_reply("- lot 2 prob 0.8"), 200, 20),
        with_usage(text_reply("Lot 2 is likely to fail."), 50, 30),
    ])
    inv = agent.investigate("Will lot 2 fail?", llm, ctx=None)
    assert fake_tools == [("predict_fail", '{"lot_id": 2}')]
    assert inv.stop_reason == "done"
    assert inv.evidence == "- lot 2 prob 0.8"
    assert inv.report == "Lot 2 is likely to fail."
    assert [s.tool for s in inv.steps] == ["predict_fail"]
    assert inv.steps[0].observation == "result of predict_fail"
    assert (inv.prompt_tokens, inv.completion_tokens) == (350, 60)
    assert inv.latency_s == pytest.approx(1.5)


def test_tool_result_message_links_call_id():
    llm = ScriptedLLM([tool_reply("spc_check", '{"sensor": "s059"}', "abc"), text_reply("ok"), text_reply("r")])
    agent.investigate("check s059", llm, ctx=None)
    second_request = llm.requests[1]
    tool_msgs = [m for m in second_request if m.get("role") == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "abc", "content": "result of spc_check"}]
    assert any(m.get("tool_calls") for m in second_request if m["role"] == "assistant")


def test_multiple_tool_calls_in_one_reply_all_run(fake_tools):
    both = LLMReply("", (ToolCall("a", "predict_fail", '{"lot_id": 1}'), ToolCall("b", "predict_fail", '{"lot_id": 2}')), "x")
    llm = ScriptedLLM([both, text_reply("ok"), text_reply("r")])
    inv = agent.investigate("lots 1 and 2?", llm, ctx=None)
    assert len(fake_tools) == 2 and len(inv.steps) == 2


def test_tool_error_is_fed_back_not_raised():
    llm = ScriptedLLM([tool_reply("predict_fail", '{"lot_id": 99999}'), text_reply("no such lot"), text_reply("r")])
    inv = agent.investigate("lot 99999?", llm, ctx=None)
    assert inv.steps[0].observation.startswith("Error")
    assert inv.stop_reason == "done"


def test_max_turns_stops_loop_then_still_writes_report():
    replies = [tool_reply("query_lots", '{"sql": "SELECT 1"}', f"c{i}") for i in range(3)] + [text_reply("partial report")]
    llm = ScriptedLLM(replies)
    inv = agent.investigate("loop forever", llm, ctx=None, max_turns=3)
    assert inv.stop_reason == "max_turns"
    assert inv.report == "partial report"
    assert len(llm.requests) == 4  # 3 investigator calls + 1 writer


def test_token_budget_checked_before_each_call():
    llm = ScriptedLLM([
        with_usage(tool_reply("query_lots", '{"sql": "SELECT 1"}'), 900, 200),
        text_reply("budget report"),  # 1100 tokens used >= 1000 budget: skip 2nd investigator call, go to writer
    ])
    inv = agent.investigate("q", llm, ctx=None, token_budget=1000)
    assert inv.stop_reason == "token_budget"
    assert inv.report == "budget report"


def test_writer_sees_evidence_but_no_tool_traffic():
    llm = ScriptedLLM([tool_reply("predict_fail", '{"lot_id": 2}'), text_reply("EVIDENCE-XYZ"), text_reply("r")])
    agent.investigate("Will lot 2 fail?", llm, ctx=None)
    writer_request = llm.requests[-1]
    assert not any(m["role"] == "tool" for m in writer_request)
    joined = " ".join(str(m.get("content", "")) for m in writer_request)
    assert "EVIDENCE-XYZ" in joined and "Will lot 2 fail?" in joined


def test_memory_saved_and_recalled():
    first = ScriptedLLM([text_reply("- s059 high on lot 2"), text_reply("Lot 2 failed due to s059.")])
    agent.investigate("Why did lot 2 fail?", first, ctx=None)
    assert len(memory.load_cases()) == 1

    second = ScriptedLLM([text_reply("ok"), text_reply("r")])
    agent.investigate("Tell me more about lot 2", second, ctx=None)
    system_prompt = second.requests[0][0]["content"]
    assert second.requests[0][0]["role"] == "system"
    assert "s059" in system_prompt  # past finding about lot 2 was injected


def test_use_memory_false_neither_reads_nor_writes():
    agent.investigate("Why did lot 2 fail?", ScriptedLLM([text_reply("e"), text_reply("r")]), ctx=None, use_memory=False)
    assert memory.load_cases() == []


def test_memory_skips_unfinished_investigations():
    replies = [tool_reply("query_lots", '{"sql": "SELECT 1"}', f"c{i}") for i in range(2)] + [text_reply("Maybe s000.")]
    agent.investigate("Why is lot 7 bad?", ScriptedLLM(replies), ctx=None, max_turns=2)
    assert memory.load_cases() == []  # a max_turns guess must not become a "past finding"


def test_memory_finding_is_first_sentence():
    llm = ScriptedLLM([text_reply("- s059 high"), text_reply("Lot 2 failed due to s059. It read 23.8 vs 2.6.")])
    agent.investigate("Why did lot 2 fail?", llm, ctx=None)
    assert memory.load_cases()[0].finding == "Lot 2 failed due to s059."
