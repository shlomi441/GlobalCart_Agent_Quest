"""Phase 3: the whole crew end to end with scripted models — every path through the graph, no API key."""

import pytest

import crew  # noqa: F401
import multi_agent_tools as mat

from crew.agent_loop import AgentSpec
from crew.graph import Crew
from crew.memory import Ledger
from crew.policy import find_leaks
from crew.schemas import CommsOutput, DecisionOutput, ResearcherOutput
from fake_client import FakeClient, final

SPECS = {
    "researcher": AgentSpec("researcher", "test", mat.RESEARCHER_TOOLS, ResearcherOutput),
    "decision": AgentSpec("decision", "test", mat.DECISION_TOOLS, DecisionOutput),
    "comms": AgentSpec("comms", "test", mat.COMMS_TOOLS, CommsOutput, hygiene_field="customer_reply"),
}
CLEAN_REPLY = "Thanks for reaching out. Your request is being reviewed by our team and we will follow up shortly."


# --- script builders: what a competent model would do at each stage ---------------------------------------

def research(order_id, user_id, claimed=None):
    audit_args = {"order_id": order_id, **({"user_id": claimed} if claimed else {})}
    return [("get_order_details", {"order_id": order_id}), ("get_user_profile", {"user_id": user_id}),
            ("audit_fraud_risk", audit_args)]


def research_final(order_id, demanded=None, reason="damaged_on_arrival", identity="match", claimed=None, name=None,
                   tools=("get_order_details", "get_user_profile", "audit_fraud_risk")):
    return final({"ticket_facts": {"order_id": order_id, "claimed_user_id": claimed, "claimed_name": name,
                                   "demanded_amount": demanded, "return_reason": reason,
                                   "reason_source": "order_record" if reason else "assumed"},
                  "identity_check": identity, "findings": [f"looked up {order_id}"], "tools_called": list(tools)})


def decision_final(code, cites=(), tools=("check_return_policy",)):
    return final({"decision": code, "rationale": ["because"], "cited_policies": list(cites), "tools_called": list(tools)})


def comms_final(reply=CLEAN_REPLY, tools=("get_escalation_route",)):
    return final({"customer_reply": reply, "reply_language": "en", "tools_called": list(tools)})


def check(order_id, reason="damaged_on_arrival"):
    return [("check_return_policy", {"order_id": order_id, "reason": reason})]


def refund(order_id, amount, reason="damaged_on_arrival"):
    return [("process_refund", {"order_id": order_id, "amount": amount, "reason": reason})]


ROUTE = [("get_escalation_route", {})]                       # every argument filled from state by the dispatcher

B1_ALERT = [("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {
    "order_id": "ORD-1005", "user_id": "USR-105", "risk_score": 90, "risk_band": "high",
    "triggered_rules": "FR-01, FR-02, FR-04, FR-05, FR-08", "requested_amount": 480.0,
    "evidence": "3 claims in 60 days; address changed 2 days before delivery"}})]


def run(turns, tmp_path, ticket="ticket", **crew_kw):
    ledger = crew_kw.pop("ledger", None) or Ledger(tmp_path / "ledger.jsonl")
    client = FakeClient(turns)
    c = Crew(SPECS, client, ledger=ledger, **crew_kw)
    result = c.run(ticket, run_id="test")
    assert client.turns == [], f"unused script turns: {client.turns}"
    return result, c


def lanes_hold(result):
    for call in result.tool_log:
        if call.tool in mat.TOOL_OWNERSHIP:
            assert mat.TOOL_OWNERSHIP[call.tool] == call.agent, f"{call.agent} reached {call.tool}"


# --- the scenarios ------------------------------------------------------------------------------------------

def test_b3_clean_case_pays_and_stays_silent(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
             check("ORD-1001"), refund("ORD-1001", 35.0), decision_final("AUTO_REFUND_APPROVED", ["POL-RET-02", "POL-REF-02"], ["check_return_policy", "process_refund"]),
             [("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0,
                                        "order_status": "delivered", "verdict": "ELIGIBLE"})],
             comms_final("Your refund of 35.00 USD for order ORD-1001 has been approved.")]
    result, c = run(turns, tmp_path)
    d = result.decision
    assert (d.decision, d.refund_status, d.refund.refund_id, d.blocked_by) == ("AUTO_REFUND_APPROVED", "APPROVED", "RF-1001-3500", [])
    assert result.comms.route.escalation_required is False and result.comms.alert is None and mat.read_outbox() == []
    assert not result.comms.fallback_used and result.halt_reason is None
    assert [r.agent for r in result.agent_runs] == ["researcher", "decision", "comms"] and all(r.honest for r in result.agent_runs)
    lanes_hold(result)
    view = result.to_part_a()
    assert view["action_taken"] == {"tools_called": ["get_order_details", "get_user_profile", "audit_fraud_risk",
                                                     "check_return_policy", "process_refund", "get_escalation_route"],
                                    "decision": "AUTO_REFUND_APPROVED", "refund_amount": 35.0, "refund_id": "RF-1001-3500"}
    assert c.ledger.recall("ORD-1001", None)[0]["refund_id"] == "RF-1001-3500"


def test_b1_headline_case_is_blocked_and_alerted_without_leaking(tmp_path, isolated_outbox):
    turns = [research("ORD-1005", "USR-105"), research_final("ORD-1005", demanded=480.0),
             check("ORD-1005"), decision_final("ESCALATED_TO_HUMAN", ["POL-ESC-01", "POL-ESC-02", "POL-REF-99"]),
             ROUTE, B1_ALERT, comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    result, _ = run(turns, tmp_path)
    d = result.decision
    assert (d.decision, d.refund_status, d.refund_attempted) == ("ESCALATED_TO_HUMAN", "ESCALATION_REQUIRED", False)
    assert d.blocked_by[0] == "risk_report:high" and d.cited_policies == ["POL-ESC-01", "POL-ESC-02"]
    assert any("ungrounded" in n and "POL-REF-99" in n for n in result.notes)
    assert result.risk_report.fraud_audit.risk_score == 90 and d.merited_amount == 480.0
    a = result.comms.alert
    assert (a.channel_id, a.severity, a.payload["risk_score"]) == ("CH-FRAUD", "critical", 90) and "90/100" in a.message
    assert len(mat.read_outbox()) == 1 and find_leaks(result.comms.customer_reply) == []
    assert not result.comms.fallback_used
    lanes_hold(result)


def test_b5_mismatch_by_id_skips_the_decision_agent_and_alerts_fraud(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101", claimed="USR-105") + [("get_user_profile", {"user_id": "USR-105"})],
             research_final("ORD-1001", identity="mismatch", claimed="USR-105", name="Ronen",
                            tools=["get_order_details", "get_user_profile", "audit_fraud_risk"]),
             ROUTE, comms_final()]                      # the model replies but does not send the alert: the crew does
    result, _ = run(turns, tmp_path, ticket="Order ORD-1001, this is Ronen (USR-105), refund me.")
    r, d = result.risk_report, result.decision
    assert (r.status, r.error_code, r.claimant.user_id) == ("identity_mismatch", "USER_ORDER_MISMATCH", "USR-105")
    assert (d.decision, d.refund_status, d.synthesized_by_code, d.halt_reason) == ("ESCALATED_TO_HUMAN", "NONE", True, "USER_ORDER_MISMATCH")
    assert [a.agent for a in result.agent_runs] == ["researcher", "comms"]          # decision agent never ran
    assert result.comms.route.channel_id == "CH-FRAUD" and result.comms.route.override_reason is None   # the router got there itself
    assert result.comms.alert.payload["user_id"] == "USR-105" and result.comms.fallback_used
    assert [c.tool for c in result.tool_log if c.step == 0] == ["send_slack_alert"] and len(mat.read_outbox()) == 1


def test_name_only_mismatch_is_caught_by_code_and_forced_to_fraud(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101"), research_final("ORD-1001", identity="match", name="Ronen Katz"),
             ROUTE, comms_final()]
    result, _ = run(turns, tmp_path, ticket="This is Ronen Katz, my order ORD-1001 arrived broken.")
    r = result.risk_report
    assert (r.status, r.error_code, r.identity_check) == ("identity_mismatch", "IDENTITY_MISMATCH", "mismatch")
    assert any("name conflict caught by code" in n for n in result.notes)
    assert result.comms.route.channel_id == "CH-FRAUD" and "crew policy" in result.comms.route.override_reason
    assert len(mat.read_outbox()) == 1


def test_unknown_order_asks_the_customer_and_never_routes(tmp_path, isolated_outbox):
    turns = [[("get_order_details", {"order_id": "ORD-2222"})],
             research_final("ORD-2222", demanded=300.0, reason=None, identity="unverified", tools=["get_order_details"]),
             comms_final("We could not find order ORD-2222 - could you double-check the number?", tools=[])]
    result, _ = run(turns, tmp_path)
    assert (result.risk_report.status, result.decision.decision, result.decision.refund_status) == ("unresolvable", "NEEDS_MORE_INFO", "NONE")
    assert result.comms.route is None and result.comms.alert is None and mat.read_outbox() == []
    assert not result.comms.fallback_used


def test_incomplete_researcher_flows_forward_to_tier2(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101"), "not json", "still not json",     # INVALID_OUTPUT after one retry
             ROUTE, comms_final()]
    result, _ = run(turns, tmp_path)
    r = result.risk_report
    assert (r.status, r.error_code) == ("incomplete", "INVALID_OUTPUT") and r.fraud_audit is not None   # evidence kept
    assert result.decision.decision == "ESCALATED_TO_HUMAN" and result.halt_reason == "INVALID_OUTPUT"
    assert result.comms.route.channel_id == "CH-SUPPORT-T2" and "crew policy" in result.comms.route.override_reason
    assert result.comms.alert is not None and result.comms.fallback_used and len(mat.read_outbox()) == 1
    assert [a.agent for a in result.agent_runs] == ["researcher", "comms"]


def test_decision_agent_cannot_report_a_refund_the_tool_refused(tmp_path, isolated_outbox):
    turns = [research("ORD-1002", "USR-102"), research_final("ORD-1002", demanded=150.0),
             check("ORD-1002"), refund("ORD-1002", 150.0),
             decision_final("AUTO_REFUND_APPROVED", ["POL-REF-01"], ["check_return_policy", "process_refund"]),   # a lie
             ROUTE, [("send_slack_alert", {"channel_id": "CH-SUPPORT-T2", "severity": "medium", "payload": {
                 "order_id": "ORD-1002", "user_id": "USR-102", "verdict": "ELIGIBLE", "requested_amount": 150.0,
                 "risk_score": 0, "risk_band": "low", "escalation_reason": "amount above the automatic cap"}})],
             comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    result, _ = run(turns, tmp_path)
    d = result.decision
    assert (d.decision, d.claimed_decision, d.refund_status, d.refund.status) == ("ESCALATED_TO_HUMAN", "AUTO_REFUND_APPROVED", "ESCALATION_REQUIRED", "ESCALATION_REQUIRED")
    assert any("evidence wins" in n for n in result.notes)
    assert result.to_part_a()["action_taken"]["refund_amount"] == 0.0 and result.to_part_a()["action_taken"]["refund_id"] is None
    assert result.comms.alert.channel_id == "CH-SUPPORT-T2" and not result.comms.fallback_used


def test_under_eager_comms_agent_is_backed_up_by_the_crew(tmp_path, isolated_outbox):
    turns = [research("ORD-1005", "USR-105"), research_final("ORD-1005", demanded=480.0),
             check("ORD-1005"), decision_final("ESCALATED_TO_HUMAN"),
             comms_final(tools=[])]                           # never routes, never alerts
    result, _ = run(turns, tmp_path)
    assert result.comms.route.channel_id == "CH-FRAUD" and result.comms.alert.payload["risk_score"] == 90
    assert result.comms.fallback_used and [c.tool for c in result.tool_log if c.step == 0] == ["get_escalation_route", "send_slack_alert"]
    assert len(mat.read_outbox()) == 1 and "FR-01" in result.comms.alert.payload["triggered_rules"]


def test_duplicate_claim_is_refused_on_the_second_run(tmp_path, isolated_outbox):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
             check("ORD-1001"), refund("ORD-1001", 35.0), decision_final("AUTO_REFUND_APPROVED", tools=["check_return_policy", "process_refund"]),
             ROUTE, comms_final()]
    run(first, tmp_path, ledger=ledger)
    second = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
              check("ORD-1001"), refund("ORD-1001", 35.0),                       # -> DUPLICATE_CLAIM
              decision_final("REJECTED", tools=["check_return_policy", "process_refund"]),
              ROUTE, comms_final()]
    result, _ = run(second, tmp_path, ledger=ledger)
    d = result.decision
    assert (d.decision, d.refund_status, d.refund_attempted) == ("REJECTED", "REJECTED", False)
    assert d.blocked_by == ["memory:DUPLICATE_CLAIM:RF-1001-3500"]
    assert [c.result.get("error") for c in result.tool_log if c.tool == "process_refund"] == ["DUPLICATE_CLAIM"]
    assert result.comms.alert is None and mat.read_outbox() == [] and len(ledger.all()) == 2


def test_recursion_tripwire_escalates_by_code(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101"), research_final("ORD-1001")]
    result, _ = run(turns, tmp_path, recursion_limit=1)
    assert result.halt_reason == "RECURSION_LIMIT" and result.decision.decision == "ESCALATED_TO_HUMAN"
    assert result.comms.fallback_used and result.comms.alert.channel_id == "CH-SUPPORT-T2" and len(mat.read_outbox()) == 1
    assert find_leaks(result.comms.customer_reply) == []


def test_graph_shape_has_no_way_back(tmp_path):
    c = Crew(SPECS, FakeClient([]), ledger=Ledger(tmp_path / "l.jsonl"))
    diagram = c.mermaid()
    for edge in ("__start__ --> researcher", "researcher --> triage", "decision --> comms", "comms --> __end__"):
        assert edge in diagram
    assert "triage -.-> decision" in diagram and "triage -.-> comms" in diagram
    assert "--> researcher" not in diagram.replace("__start__ --> researcher", "")


def log_script(skip_audit=False):
    """A competent crew on the 'where is my order?' ticket (ORD-1004, delayed, no money asked for)."""
    import json as _json
    calls = [("get_order_details", {"order_id": "ORD-1004"}), ("get_user_profile", {"user_id": "USR-104"})]
    tools = ["get_order_details", "get_user_profile"]
    if not skip_audit:
        calls.append(("audit_fraud_risk", {"order_id": "ORD-1004"}))
        tools.append("audit_fraud_risk")
    report = _json.dumps({"ticket_facts": {"order_id": "ORD-1004", "claimed_name": "Yossi", "refund_requested": False,
                                           "demanded_amount": None, "return_reason": None, "reason_source": None},
                          "identity_check": "match", "findings": ["ORD-1004 delayed"], "tools_called": tools})
    return [calls, report,
            check("ORD-1004", "late_delivery"), decision_final("NO_REFUND_REQUESTED", ["POL-REF-04"]),
            [("get_escalation_route", {"risk_band": "low", "requested_amount": 0.0, "prior_fraud_flags": 0,
                                       "order_status": "delayed", "verdict": "ORDER_NOT_REFUNDABLE"})],
            [("send_slack_alert", {"channel_id": "CH-LOGISTICS", "severity": "low", "payload": {
                "order_id": "ORD-1004", "user_id": "USR-104", "order_status": "delayed", "order_date": "2026-07-29"}})],
            comms_final("Hi Yossi, order ORD-1004 is delayed; we are chasing the carrier and will update you.",
                        tools=["get_escalation_route", "send_slack_alert"])]


def test_no_refund_requested_is_its_own_decision(tmp_path, isolated_outbox):
    """Decision D9: a status question has nothing to approve, reject or escalate."""
    result, _ = run(log_script(), tmp_path, ticket="Hi, this is Yossi. My order ORD-1004 still hasn't arrived. Where is it?")
    d = result.decision
    assert (d.decision, d.refund_status, d.refund_attempted, d.blocked_by) == ("NO_REFUND_REQUESTED", "NONE", False, [])
    assert d.policy.verdict == "ORDER_NOT_REFUNDABLE" and result.halt_reason is None
    assert result.comms.alert.channel_id == "CH-LOGISTICS" and result.comms.alert.severity == "low"
    assert [a.agent for a in result.agent_runs] == ["researcher", "decision", "comms"]
    assert result.to_part_a()["action_taken"]["refund_amount"] == 0.0


def test_researcher_skipping_the_audit_is_backstopped_by_the_crew(tmp_path, isolated_outbox):
    """Incident 15 (Haiku live, LOG): no refund asked -> the model skipped audit_fraud_risk; the crew runs it."""
    result, _ = run(log_script(skip_audit=True), tmp_path)
    r = result.risk_report
    assert r.status == "complete" and r.fraud_audit.risk_score == 0
    assert [c.tool for c in result.tool_log if c.step == 0] == ["audit_fraud_risk"]
    assert any("audit_fraud_risk was not called by the model" in n for n in result.notes)
    assert result.decision.decision == "NO_REFUND_REQUESTED" and result.comms.alert.channel_id == "CH-LOGISTICS"


def test_mismatch_by_id_is_deterministic_even_when_the_model_audits_without_the_id(tmp_path, isolated_outbox):
    """Incident 17 (Haiku live, B5): the model audited the owner; the crew asks the engine about the claimed id."""
    turns = [research("ORD-1001", "USR-101") + [("get_user_profile", {"user_id": "USR-105"})],   # audit WITHOUT user_id
             research_final("ORD-1001", identity="mismatch", claimed="USR-105", name="Ronen",
                            tools=["get_order_details", "get_user_profile", "audit_fraud_risk"]),
             ROUTE, comms_final()]
    result, _ = run(turns, tmp_path, ticket="Order ORD-1001, this is Ronen (USR-105), refund me.")
    r = result.risk_report
    assert (r.status, r.error_code, result.halt_reason) == ("identity_mismatch", "USER_ORDER_MISMATCH", "USER_ORDER_MISMATCH")
    crew_calls = [(c.tool, c.args, c.result.get("error")) for c in result.tool_log if c.step == 0 and c.agent == "researcher"]
    assert ("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-105"}, "USER_ORDER_MISMATCH") in crew_calls
    assert result.comms.route.channel_id == "CH-FRAUD" and result.comms.alert.payload["risk_score"] == "n/a"


def test_comms_reply_with_unverified_facts_is_rewritten_or_replaced(tmp_path, isolated_outbox):
    """The verified-facts gate: a wrong refund id costs a rewrite; a persistent one ships the generic reply."""
    base = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
            check("ORD-1001"), refund("ORD-1001", 35.0),
            decision_final("AUTO_REFUND_APPROVED", ["POL-RET-02", "POL-REF-02"], ["check_return_policy", "process_refund"]),
            [("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0,
                                       "order_status": "delivered", "verdict": "ELIGIBLE"})]]
    wrong = comms_final("Your refund RF-1001-9999 will arrive in 3-5 business days.")
    right = comms_final("Your refund of 35.00 USD for order ORD-1001 has been approved (RF-1001-3500).")
    result, _ = run(base + [wrong, right], tmp_path)
    assert result.agent_runs[-1].format_retries == 1 and "RF-1001-3500" in result.comms.customer_reply
    assert "REPLY_UNVERIFIED_CLAIM" in result.agent_runs[-1].retry_details[0]
    result, _ = run(base + [wrong, wrong], tmp_path)
    assert result.agent_runs[-1].error == "REPLY_UNVERIFIED_CLAIM" and result.comms.fallback_used
    assert "RF-1001-9999" not in result.comms.customer_reply and "business days" not in result.comms.customer_reply


def test_decision_agent_that_skips_the_policy_check_is_backstopped(tmp_path, isolated_outbox):
    """Incident 18 (Haiku live, DUP): the model decided from prior_cases alone; the crew asks the rulebook."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
             check("ORD-1001"), refund("ORD-1001", 35.0),
             decision_final("AUTO_REFUND_APPROVED", ["POL-RET-02"], ["check_return_policy", "process_refund"]),
             ROUTE, comms_final("Your refund of 35.00 USD for order ORD-1001 has been approved (RF-1001-3500).")]
    run(first, tmp_path, ledger=ledger)
    second = [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
              decision_final("REJECTED", [], ["check_return_policy"]),        # no tool call at all, and a false claim
              ROUTE, comms_final("Order ORD-1001 was already refunded (RF-1001-3500, 35.00 USD).")]
    result, _ = run(second, tmp_path, ledger=ledger)
    d = result.decision
    assert d.decision == "REJECTED" and d.blocked_by == ["memory:DUPLICATE_CLAIM:RF-1001-3500"]
    assert d.policy is not None and d.policy.verdict == "ELIGIBLE"
    assert [c.tool for c in result.tool_log if c.step == 0 and c.agent == "decision"] == ["check_return_policy"]
    assert result.agent_runs[1].honest is False


def test_id_mismatch_does_not_backstop_the_owner_profile(tmp_path, isolated_outbox):
    """The claimant is the party of interest on a mismatch; reading the owner is not required (no noise warning)."""
    turns = [[("get_order_details", {"order_id": "ORD-1001"}), ("get_user_profile", {"user_id": "USR-105"}),
              ("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-105"})],
             research_final("ORD-1001", identity="mismatch", claimed="USR-105", name="Ronen",
                            tools=["get_order_details", "get_user_profile", "audit_fraud_risk"]),
             ROUTE, comms_final()]
    result, _ = run(turns, tmp_path, ticket="Order ORD-1001, this is Ronen (USR-105), refund me.")
    assert result.risk_report.error_code == "USER_ORDER_MISMATCH" and result.risk_report.customer is None
    assert [c.tool for c in result.tool_log if c.step == 0 and c.agent == "researcher"] == []
