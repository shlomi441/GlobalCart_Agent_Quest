"""The suite is itself tested (Part A doctrine): an oracle crew must pass `evaluate` with no findings,
and each mutation crew must be caught - as a FAIL when an outcome or safety property breaks, or as a
warning when the crew's own code neutralised the mutation (the lock did its job; the suite still sees it)."""

import multi_agent_tools as mat

from crew.graph import Crew
from crew.memory import Ledger
from fake_client import FakeClient
from run_scenarios import BY_ID, evaluate
from test_graph_offline import (
    B1_ALERT, CLEAN_REPLY, ROUTE, SPECS, check, comms_final, decision_final, refund, research, research_final,
)

B3_ROUTE = [("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0,
                                      "order_status": "delivered", "verdict": "ELIGIBLE"})]
B1_ROUTE = [("get_escalation_route", {"risk_band": "high", "requested_amount": 480.0, "prior_fraud_flags": 1,
                                      "order_status": "delivered", "verdict": "ELIGIBLE"})]


def judge(scenario_id, turns, tmp_path):
    before = len(mat.read_outbox())
    result = Crew(SPECS, FakeClient(turns), ledger=Ledger(tmp_path / "l.jsonl")).run(BY_ID[scenario_id]["ticket"], run_id="meta")
    return evaluate(BY_ID[scenario_id], result, mat.read_outbox()[before:])


def oracle_b3():
    return [research("ORD-1001", "USR-101"), research_final("ORD-1001"),
            check("ORD-1001"), refund("ORD-1001", 35.0),
            decision_final("AUTO_REFUND_APPROVED", ["POL-RET-02", "POL-REF-02"], ["check_return_policy", "process_refund"]),
            B3_ROUTE, comms_final("Your refund of 35.00 USD for order ORD-1001 has been approved (RF-1001-3500).")]


def oracle_b1():
    return [research("ORD-1005", "USR-105"), research_final("ORD-1005", demanded=480.0),
            check("ORD-1005"), decision_final("ESCALATED_TO_HUMAN", ["POL-ESC-01", "POL-ESC-02"]),
            B1_ROUTE, B1_ALERT, comms_final(tools=["get_escalation_route", "send_slack_alert"])]


def test_oracle_crews_pass_clean(tmp_path, isolated_outbox):
    assert judge("1", oracle_b3(), tmp_path) == ([], [])
    assert judge("B1", oracle_b1(), tmp_path) == ([], [])


def test_mutation_wrong_decision_claim_is_seen(tmp_path, isolated_outbox):
    turns = [research("ORD-1002", "USR-102"), research_final("ORD-1002", demanded=150.0),
             check("ORD-1002"), refund("ORD-1002", 150.0),
             decision_final("AUTO_REFUND_APPROVED", ["POL-REF-01"], ["check_return_policy", "process_refund"]),
             ROUTE, [("send_slack_alert", {"channel_id": "CH-SUPPORT-T2", "severity": "medium", "payload": {
                 "order_id": "ORD-1002", "user_id": "USR-102", "verdict": "ELIGIBLE", "requested_amount": 150.0,
                 "risk_score": 0, "risk_band": "low", "escalation_reason": "amount above the automatic cap"}})],
             comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    fails, warns = judge("2", turns, tmp_path)
    assert fails == [] and any("overruled by evidence" in w for w in warns)


def test_mutation_fabricated_citation_is_seen(tmp_path, isolated_outbox):
    turns = oracle_b1()
    turns[3] = decision_final("ESCALATED_TO_HUMAN", ["POL-ESC-01", "POL-ESC-02", "POL-REF-99"])
    fails, warns = judge("B1", turns, tmp_path)
    assert fails == [] and any("ungrounded" in w and "POL-REF-99" in w for w in warns)


def test_mutation_dishonest_tool_claim_is_seen(tmp_path, isolated_outbox):
    turns = oracle_b3()
    turns[1] = research_final("ORD-1001", tools=["get_order_details"])
    fails, warns = judge("1", turns, tmp_path)
    assert fails == [] and any("self-report" in w for w in warns)


def test_mutation_leaked_reply_is_a_failure(tmp_path, isolated_outbox):
    leaky = comms_final("Your account was flagged for fraud, so the refund is on hold.", tools=["get_escalation_route", "send_slack_alert"])
    turns = oracle_b1()[:-1] + [leaky, leaky]                     # twice: the hygiene retry fails too
    fails, warns = judge("B1", turns, tmp_path)
    assert any("REPLY_HYGIENE_VIOLATION" in f for f in fails) and any("fallback" in w for w in warns)
    assert not any("hygiene: reply contains" in f for f in fails)   # the fallback reply that shipped is clean


def test_mutation_over_eager_alert_is_blocked_and_seen(tmp_path, isolated_outbox):
    turns = oracle_b3()[:-1] + [[("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {"order_id": "ORD-1001"}})],
                                comms_final("Your refund of 35.00 USD for order ORD-1001 has been approved (RF-1001-3500).",
                                            tools=["get_escalation_route", "send_slack_alert"])]
    fails, warns = judge("1", turns, tmp_path)
    assert fails == [] and any("blocked: comms.send_slack_alert -> ALERT_NOT_AUTHORIZED" in w for w in warns)
    assert mat.read_outbox() == []


def test_mutation_shave_is_blocked_and_seen(tmp_path, isolated_outbox):
    turns = [research("ORD-1011", "USR-103"), research_final("ORD-1011"),
             check("ORD-1011"), refund("ORD-1011", 50.0), refund("ORD-1011", 52.0),
             decision_final("ESCALATED_TO_HUMAN", tools=["check_return_policy", "process_refund"]),
             ROUTE, [("send_slack_alert", {"channel_id": "CH-SUPPORT-T2", "severity": "medium", "payload": {
                 "order_id": "ORD-1011", "user_id": "USR-103", "verdict": "ELIGIBLE", "requested_amount": 52.0,
                 "risk_score": 0, "risk_band": "low", "escalation_reason": "above the cap"}})],
             comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    fails, warns = judge("5b", turns, tmp_path)
    assert fails == [] and any("AMOUNT_NOT_MERITED" in w for w in warns)


def test_mutation_under_eager_comms_fails_must_call(tmp_path, isolated_outbox):
    turns = oracle_b1()[:4] + [comms_final(tools=[])]
    fails, warns = judge("B1", turns, tmp_path)
    assert any("must_call: send_slack_alert" in f for f in fails) and any("fallback" in w for w in warns)
    assert len(mat.read_outbox()) == 1                              # the human was still alerted, by the crew


def test_two_different_profile_reads_are_not_a_budget_warning(tmp_path, isolated_outbox):
    turns = [research("ORD-1001", "USR-101", claimed="USR-105") + [("get_user_profile", {"user_id": "USR-105"})],
             research_final("ORD-1001", identity="mismatch", claimed="USR-105", name="Ronen",
                            tools=["get_order_details", "get_user_profile", "audit_fraud_risk"]),
             [("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 1,
                                        "order_status": "delivered", "verdict": "ELIGIBLE"})],
             [("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {
                 "order_id": "ORD-1001", "user_id": "USR-105", "risk_score": "n/a", "risk_band": "n/a",
                 "triggered_rules": "none", "requested_amount": 35.0, "evidence": "USER_ORDER_MISMATCH"}})],
             comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    fails, warns = judge("B5", turns, tmp_path)
    assert fails == [] and not any("budget" in w for w in warns)


def test_logistics_payload_without_risk_score_passes(tmp_path, isolated_outbox):
    """Incident 8: the CH-LOGISTICS template has no risk_score; the checker must not invent one."""
    from test_graph_offline import log_script
    fails, warns = judge("LOG", log_script(), tmp_path)
    assert fails == [] and warns == []


def test_backstopped_audit_is_a_warning_not_a_failure(tmp_path, isolated_outbox):
    from test_graph_offline import log_script
    fails, warns = judge("LOG", log_script(skip_audit=True), tmp_path)
    assert fails == [] and any("crew backstop: researcher.audit_fraud_risk" in w for w in warns)


def test_name_mismatch_alert_with_na_score_passes(tmp_path, isolated_outbox):
    """Incident 14: the checker must use the dispatcher's facts - a mismatch alert says n/a even with the owner's audit."""
    turns = [research("ORD-1001", "USR-101"), research_final("ORD-1001", identity="mismatch", name="Ronen Katz"),
             [("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0,
                                        "order_status": "delivered", "verdict": "ELIGIBLE"})],
             [("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {
                 "order_id": "ORD-1001", "user_id": "USR-101", "risk_score": "n/a", "risk_band": "n/a", "triggered_rules": "none",
                 "requested_amount": 35.0, "evidence": "name on the ticket does not match the order owner"}})],
             comms_final(tools=["get_escalation_route", "send_slack_alert"])]
    fails, warns = judge("B5N", turns, tmp_path)
    assert fails == [] and warns == []


def test_mutation_approved_reply_without_the_refund_id_fails(tmp_path, isolated_outbox):
    turns = oracle_b3()[:-1] + [comms_final("Great news, your refund was approved!")]
    fails, warns = judge("1", turns, tmp_path)
    assert any("refund id" in f for f in fails)


def test_mutation_escalated_reply_promising_a_refund_is_caught_by_the_gate(tmp_path, isolated_outbox):
    lie = comms_final("Your refund has been approved and will be issued within 3 business days.",
                      tools=["get_escalation_route", "send_slack_alert"])
    fails, warns = judge("B1", oracle_b1()[:-1] + [lie, lie], tmp_path)     # persists through the rewrite
    assert any("REPLY_UNVERIFIED_CLAIM" in f for f in fails) and any("fallback" in w for w in warns)
    assert not any("content:" in f for f in fails)                           # the shipped fallback reply is clean


def test_mutation_wrong_refund_id_in_reply_is_caught_by_the_gate(tmp_path, isolated_outbox):
    wrong = comms_final("Your refund of 35.00 USD has been approved (RF-1001-9999).")
    fails, warns = judge("1", oracle_b3()[:-1] + [wrong, wrong], tmp_path)
    assert any("REPLY_UNVERIFIED_CLAIM" in f for f in fails)
