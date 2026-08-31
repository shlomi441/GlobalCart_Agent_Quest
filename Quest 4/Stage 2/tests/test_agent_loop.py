"""Phase 2: the loop with a scripted model — output validation, retries, hygiene, honesty, both output modes."""

import json

import crew  # noqa: F401
import multi_agent_tools as mat

from crew.agent_loop import AgentSpec, run_agent
from crew.dispatch import CommsDispatcher, ResearcherDispatcher
from crew.schemas import CommsOutput, ResearcherOutput
from conftest import make_decision, make_report
from fake_client import FakeClient, final

RESEARCH = [("get_order_details", {"order_id": "ORD-1001"}), ("get_user_profile", {"user_id": "USR-101"}),
            ("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-101"})]
REPORT = {"ticket_facts": {"order_id": "ORD-1001", "demanded_amount": None, "return_reason": "damaged_on_arrival",
                           "reason_source": "order_record", "language": "en", "sentiment": "calm"},
          "identity_check": "match", "findings": ["ORD-1001 delivered 2026-07-25, earbuds damaged_on_arrival, risk 0/100 low"],
          "tools_called": ["get_order_details", "get_user_profile", "audit_fraud_risk"]}


def researcher_spec(**kw):
    return AgentSpec(name="researcher", system_prompt="test", tools=mat.RESEARCHER_TOOLS, output_model=ResearcherOutput, **kw)


def comms_spec(**kw):
    return AgentSpec(name="comms", system_prompt="test", tools=mat.COMMS_TOOLS, output_model=CommsOutput,
                     hygiene_field="customer_reply", **kw)


def test_happy_path_validates_and_records_honesty():
    client = FakeClient([RESEARCH, final(REPORT)])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.output.identity_check == "match"
    assert res.run.honest and res.run.executed_tools == ["get_order_details", "get_user_profile", "audit_fraud_risk"]
    assert res.run.steps == 2 and res.run.format_retries == 0
    assert [t["name"] for t in client.requests[0]["tools"]] == ["get_order_details", "get_user_profile", "audit_fraud_risk"]
    assert json.dumps(res.transcript)   # serialisable, so reports and grounding checks can read it


def test_dishonest_tool_claim_is_measured_not_trusted():
    client = FakeClient([RESEARCH, final({**REPORT, "tools_called": ["get_order_details"]})])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.honest is False


def test_invalid_output_gets_one_retry_then_fails_loudly():
    client = FakeClient([RESEARCH, "I think it is fine.", final(REPORT)])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.format_retries == 1
    client = FakeClient([RESEARCH, "no", final({**REPORT, "identity_check": "definitely"})])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.output is None and res.error == "INVALID_OUTPUT" and "identity_check" in res.detail


def test_max_steps_returns_data_not_an_exception():
    client = FakeClient([[("get_user_profile", {"user_id": "USR-101"})]] * 3)
    res = run_agent(researcher_spec(max_steps=3), "ticket", ResearcherDispatcher(), client)
    assert res.error == "MAX_STEPS_EXCEEDED" and res.run.steps == 3


def test_hygiene_gate_rewrites_once_then_fails(isolated_outbox):
    report = make_report("ORD-1005", demanded=480.0)
    dispatcher = CommsDispatcher(report, make_decision(report))
    leaky = {"customer_reply": "Your account was flagged for fraud, so no refund.", "reply_language": "en", "tools_called": []}
    clean = {"customer_reply": "Your request is under review by our team; we will follow up.", "reply_language": "en", "tools_called": []}
    res = run_agent(comms_spec(), "brief", dispatcher, FakeClient([final(leaky), final(clean)]))
    assert res.error is None and res.run.format_retries == 1
    res = run_agent(comms_spec(), "brief", dispatcher, FakeClient([final(leaky), final(leaky)]))
    assert res.error == "REPLY_HYGIENE_VIOLATION" and "fraud" in res.detail


def test_tool_output_mode_accepts_the_finish_call_and_rejects_text():
    spec = researcher_spec(output_mode="tool")
    client = FakeClient([RESEARCH, [("finish_researcher", REPORT)]])
    res = run_agent(spec, "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.honest
    offered = [t["name"] for t in client.requests[0]["tools"]]
    assert offered[-1] == "finish_researcher" and "$ref" not in json.dumps(client.requests[0]["tools"][-1])
    client = FakeClient([RESEARCH, "here is my answer", [("finish_researcher", REPORT)]])
    res = run_agent(spec, "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.format_retries == 1
    client = FakeClient([RESEARCH, [("finish_researcher", {"bad": 1})], [("finish_researcher", REPORT)]])
    res = run_agent(spec, "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.format_retries == 1
    roles = [m["role"] for m in res.transcript]
    assert all(a != b for a, b in zip(roles, roles[1:]))   # strict alternation survived the retry


def test_empty_tool_use_turn_never_produces_an_empty_message():
    """Incident 11 (Haiku 4.5 live): stop_reason 'tool_use' with no tool_use block crashed the API call."""
    from fake_client import EMPTY_TOOL_TURN, OddTurn
    client = FakeClient([RESEARCH, EMPTY_TOOL_TURN, final(REPORT)])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.format_retries == 0 and res.run.anomalies and "stop_reason=tool_use" in res.run.anomalies[0]
    for msg in res.transcript:
        assert msg["content"], "an empty message reached the transcript"
    roles = [m["role"] for m in res.transcript]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    # text arriving under a tool_use stop reason is still a valid final answer
    client = FakeClient([RESEARCH, OddTurn("tool_use", [final(REPORT)])])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error is None and res.run.anomalies
    # two empty turns in a row spend the retry budget, then fail as data
    client = FakeClient([RESEARCH, EMPTY_TOOL_TURN, EMPTY_TOOL_TURN, EMPTY_TOOL_TURN])
    res = run_agent(researcher_spec(), "ticket", ResearcherDispatcher(), client)
    assert res.error == "INVALID_OUTPUT" and res.output is None
