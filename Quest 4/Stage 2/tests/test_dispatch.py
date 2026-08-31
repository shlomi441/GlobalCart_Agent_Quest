"""Phase 2: every lock trips, every lane holds — no model, no API key."""

import pytest

import crew  # noqa: F401
import multi_agent_tools as mat

from crew.dispatch import CommsDispatcher, DecisionDispatcher, ResearcherDispatcher
from crew.policy import halt_plan, merited_amount
from conftest import make_decision, make_report

B1_PAYLOAD = {"order_id": "ORD-1005", "user_id": "USR-105", "risk_score": 90, "risk_band": "high",
              "triggered_rules": ["FR-01", "FR-02", "FR-04", "FR-05", "FR-08"], "requested_amount": 480.0,
              "evidence": "3 claims in 60 days; address re-routed 2 days before delivery"}


def errors(dispatcher):
    return [(c.tool, c.result.get("error"), c.synthetic) for c in dispatcher.log]


# --- lane and guards (any dispatcher) ------------------------------------------------------------

def test_lane_out_of_lane_and_unknown_tools_never_execute():
    d = CommsDispatcher(make_report("ORD-1001"), make_decision(make_report("ORD-1001")))
    assert d.call("process_refund", {"order_id": "ORD-1001", "amount": 35.0}, 1)["error"] == "OUT_OF_LANE"
    assert "decision agent" in d.log[-1].result["message"]
    assert d.call("none", {}, 1)["error"] == "UNKNOWN_TOOL"
    assert all(c.synthetic for c in d.log) and d.executed_tools() == []
    assert d.registry.keys() == {"get_escalation_route", "send_slack_alert"}


def test_repeat_guard_third_identical_call_is_refused():
    d = ResearcherDispatcher()
    for _ in range(2):
        assert "error" not in d.call("get_order_details", {"order_id": "ORD-1001"}, 1)
    assert d.call("get_order_details", {"order_id": "ORD-1001"}, 2)["error"] == "REPEATED_CALL"
    assert d.executed_tools() == ["get_order_details"]


def test_execution_errors_become_data():
    d = ResearcherDispatcher()
    assert d.call("get_order_details", {"order_id": 1001}, 1)["error"] == "TOOL_EXECUTION_ERROR"   # TypeError in the kit


# --- researcher --------------------------------------------------------------------------------

def test_researcher_no_id_probing_after_a_mismatch():
    d = ResearcherDispatcher()
    assert d.call("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-105"}, 1)["error"] == "USER_ORDER_MISMATCH"
    assert d.call("audit_fraud_risk", {"order_id": "ORD-1001", "user_id": "USR-101"}, 2)["error"] == "ID_PROBING_BLOCKED"
    assert d.call("audit_fraud_risk", {"order_id": "ORD-1001"}, 2)["error"] == "ID_PROBING_BLOCKED"
    assert d.call("get_order_details", {"order_id": "ORD-1005"}, 2)["error"] == "ID_PROBING_BLOCKED"
    assert d.call("get_user_profile", {"user_id": "USR-105"}, 3)["prior_fraud_flags"] == 1   # reading the claimant is allowed


def test_researcher_no_guessing_after_order_not_found():
    d = ResearcherDispatcher()
    assert d.call("get_order_details", {"order_id": "ORD-2222"}, 1)["error"] == "ORDER_NOT_FOUND"
    assert d.call("get_order_details", {"order_id": "ORD-1001"}, 2)["error"] == "ID_PROBING_BLOCKED"
    assert d.call("audit_fraud_risk", {"order_id": "ORD-2222"}, 2)["error"] == "ID_PROBING_BLOCKED"


# --- decision ----------------------------------------------------------------------------------

def _decision_dispatcher(order_id, demanded=None, prior=()):
    report = make_report(order_id, demanded=demanded)
    return DecisionDispatcher(report, merited_amount(report.order.total_amount, demanded), list(prior))


def test_headline_case_is_blocked_by_the_risk_report_before_anything_else():
    d = _decision_dispatcher("ORD-1005", demanded=480.0)
    assert d.call("process_refund", {"order_id": "ORD-1005", "amount": 480.0}, 1)["error"] == "BLOCKED_BY_RISK_REPORT"
    assert d.call("check_return_policy", {"order_id": "ORD-1005", "reason": "damaged_on_arrival"}, 1)["verdict"] == "ELIGIBLE"
    assert d.call("process_refund", {"order_id": "ORD-1005", "amount": 480.0}, 2)["error"] == "BLOCKED_BY_RISK_REPORT"
    assert d.real_results("process_refund") == []          # the kit never saw a payout attempt


def test_sequencing_verdict_and_amount_locks_on_a_clean_order():
    d = _decision_dispatcher("ORD-1001", demanded=999.0)     # the greed test
    assert d.call("process_refund", {"order_id": "ORD-1001", "amount": 35.0}, 1)["error"] == "SEQUENCING_VIOLATION"
    d.call("check_return_policy", {"order_id": "ORD-1001", "reason": "damaged_on_arrival"}, 2)
    out = d.call("process_refund", {"order_id": "ORD-1001", "amount": 999.0}, 3)
    assert out["error"] == "AMOUNT_NOT_MERITED" and out["merited_amount"] == 35.0
    assert d.call("process_refund", {"order_id": "ORD-1001", "amount": 35.0}, 4)["status"] == "APPROVED"
    assert d.call("process_refund", {"order_id": "ORD-1006", "amount": 1.0}, 5)["error"] == "WRONG_ORDER"


def test_the_shave_is_refused_and_the_kit_arbitrates_the_cap():
    d = _decision_dispatcher("ORD-1011")                     # $52, cap $50 — incident 8 from Part A
    d.call("check_return_policy", {"order_id": "ORD-1011", "reason": "damaged_on_arrival"}, 1)
    assert d.call("process_refund", {"order_id": "ORD-1011", "amount": 50.0}, 2)["error"] == "AMOUNT_NOT_MERITED"
    assert d.call("process_refund", {"order_id": "ORD-1011", "amount": 52.0}, 3)["status"] == "ESCALATION_REQUIRED"


def test_ineligible_and_escalation_verdicts_block_the_attempt():
    d = _decision_dispatcher("ORD-1003")
    d.call("check_return_policy", {"order_id": "ORD-1003", "reason": "changed_mind"}, 1)
    assert d.call("process_refund", {"order_id": "ORD-1003", "amount": 42.5}, 2)["error"] == "BLOCKED_BY_POLICY_VERDICT"


def test_duplicate_claim_from_memory_is_refused():
    prior = [{"order_id": "ORD-1001", "refund_status": "APPROVED", "refund_id": "RF-1001-3500", "timestamp": "2026-08-20"}]
    d = _decision_dispatcher("ORD-1001", prior=prior)
    d.call("check_return_policy", {"order_id": "ORD-1001", "reason": "damaged_on_arrival"}, 1)
    assert d.call("process_refund", {"order_id": "ORD-1001", "amount": 35.0}, 2)["error"] == "DUPLICATE_CLAIM"


# --- comms -------------------------------------------------------------------------------------

def _comms(order_id, demanded=None, claimed=None):
    report = make_report(order_id, demanded=demanded, claimed_user_id=claimed)
    decision = make_decision(report)
    plan = None if report.status == "complete" else halt_plan(report.status, report.error_code)
    return CommsDispatcher(report, decision, plan)


def test_headline_case_routes_alerts_once_and_only_with_true_facts(isolated_outbox):
    d = _comms("ORD-1005", demanded=480.0)
    assert d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": B1_PAYLOAD}, 1)["error"] == "ALERT_NOT_AUTHORIZED"
    wrong = d.call("get_escalation_route", {"risk_band": "low", "requested_amount": 480.0}, 1)
    assert wrong["error"] == "ARGUMENT_MISMATCH" and "risk_band" in wrong["message"]
    route = d.call("get_escalation_route", {"risk_band": "high"}, 2)   # omitted args are filled from state
    assert route["channel_id"] == "CH-FRAUD" and any("filled prior_fraud_flags=1" in n for n in d.notes)
    assert d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "low", "payload": B1_PAYLOAD}, 3)["error"] == "ROUTE_MISMATCH"
    bad = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical",
                                      "payload": {**B1_PAYLOAD, "risk_score": 70, "evidence": None}}, 3)
    assert bad["error"] == "PAYLOAD_MISMATCH" and "risk_score" in bad["message"]
    missing = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical",
                                          "payload": {k: v for k, v in B1_PAYLOAD.items() if k != "evidence"}}, 3)
    assert "evidence" in missing["message"]
    sent = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": B1_PAYLOAD}, 4)
    assert sent["delivered"] and "90/100" in sent["message"] and d.alert is sent
    assert d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": B1_PAYLOAD}, 5)["error"] == "DUPLICATE_ALERT"
    assert len(mat.read_outbox()) == 1


def test_clean_case_never_alerts(isolated_outbox):
    d = _comms("ORD-1001")
    route = d.call("get_escalation_route", {"risk_band": "low", "requested_amount": 35.0}, 1)
    assert route["escalation_required"] is False
    out = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": {"order_id": "ORD-1001"}}, 2)
    assert out["error"] == "ALERT_NOT_AUTHORIZED" and mat.read_outbox() == []


def test_mismatch_routes_on_the_claimants_flags_or_by_crew_override():
    flagged = _comms("ORD-1001", claimed="USR-105")               # the kit's own B5 ticket
    route = flagged.call("get_escalation_route", {}, 1)
    assert route["channel_id"] == "CH-FRAUD" and route.get("override_reason") is None    # the router got there itself
    clean = _comms("ORD-1001", claimed="USR-102")                 # a clean claimant: the router says nothing
    route = clean.call("get_escalation_route", {}, 1)
    assert route["channel_id"] == "CH-FRAUD" and "crew policy" in route["override_reason"]


def test_unestablished_case_is_never_routed():
    d = _comms("ORD-2222", demanded=300.0)
    assert d.call("get_escalation_route", {"risk_band": "low", "requested_amount": 300.0}, 1)["error"] == "ROUTING_NOT_APPLICABLE"
    assert d.call("send_slack_alert", {"channel_id": "CH-FINANCE", "severity": "high", "payload": {}}, 2)["error"] == "ALERT_NOT_AUTHORIZED"


def test_ungrounded_policy_ids_in_an_alert_are_refused(isolated_outbox):
    d = _comms("ORD-1002", demanded=150.0)                       # cap breach -> Tier 2
    route = d.call("get_escalation_route", {}, 1)
    assert route["channel_id"] == "CH-SUPPORT-T2"
    payload = {"order_id": "ORD-1002", "user_id": "USR-102", "verdict": "ELIGIBLE", "requested_amount": 150.0,
               "risk_score": 0, "risk_band": "low", "escalation_reason": "above the automatic cap",
               "applicable_policies": ["POL-RET-01", "POL-ESC-02"]}
    out = d.call("send_slack_alert", {"channel_id": "CH-SUPPORT-T2", "severity": "medium", "payload": payload}, 2)
    assert out["error"] == "PAYLOAD_MISMATCH" and "POL-ESC-02" in out["message"]
    payload["applicable_policies"] = ["POL-RET-01", "POL-REF-01"]
    assert d.call("send_slack_alert", {"channel_id": "CH-SUPPORT-T2", "severity": "medium", "payload": payload}, 3)["delivered"]


def test_no_audit_means_no_score_in_the_alert(isolated_outbox):
    d = _comms("ORD-1001", claimed="USR-105")                    # incident 5: the mismatch alert said "61/100 (low)"
    d.call("get_escalation_route", {}, 1)
    fabricated = {"order_id": "ORD-1001", "user_id": "USR-105", "risk_score": 61, "risk_band": "low",
                  "triggered_rules": "USER_ORDER_MISMATCH", "requested_amount": 35.0, "evidence": "mismatch"}
    out = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": fabricated}, 2)
    assert out["error"] == "PAYLOAD_MISMATCH" and "risk_score" in out["message"] and "risk_band" in out["message"]
    honest = {**fabricated, "risk_score": "n/a", "risk_band": "n/a"}
    sent = d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": honest}, 3)
    assert sent["delivered"] and "n/a/100 (n/a)" in sent["message"]


def test_mismatch_alert_never_carries_the_owners_audit(isolated_outbox):
    """Incident 12 (Haiku live, B5N): the researcher audited the owner's order before the name conflict was caught."""
    report = make_report("ORD-1001")                            # complete audit for Maya's order (0/100 low)...
    report = report.model_copy(update={"status": "identity_mismatch", "error_code": "IDENTITY_MISMATCH",
                                       "identity_check": "mismatch",
                                       "ticket_facts": report.ticket_facts.model_copy(update={"claimed_name": "Ronen Katz"})})
    d = CommsDispatcher(report, make_decision(report), halt_plan("identity_mismatch", "IDENTITY_MISMATCH"))
    route = d.call("get_escalation_route", {}, 1)
    assert route["channel_id"] == "CH-FRAUD" and "crew policy" in route["override_reason"]
    owners = {"order_id": "ORD-1001", "user_id": "USR-101", "risk_score": 0, "risk_band": "low", "triggered_rules": "none",
              "requested_amount": 35.0, "evidence": "name conflict"}
    assert d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": owners}, 2)["error"] == "PAYLOAD_MISMATCH"
    honest = {**owners, "risk_score": "n/a", "risk_band": "n/a"}
    assert d.call("send_slack_alert", {"channel_id": "CH-FRAUD", "severity": "critical", "payload": honest}, 3)["delivered"]


def test_no_refund_requested_ticket_cannot_be_paid():
    report = make_report("ORD-1004", refund_requested=False)
    d = DecisionDispatcher(report, merited_amount(report.order.total_amount, None), [])
    d.call("check_return_policy", {"order_id": "ORD-1004", "reason": "late_delivery"}, 1)
    assert d.call("process_refund", {"order_id": "ORD-1004", "amount": 78.9}, 2)["error"] == "NO_REFUND_REQUESTED"
