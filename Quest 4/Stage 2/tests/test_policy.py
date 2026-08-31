"""Phase 1: every crew rule pinned, with no model and no API key."""

import pytest

import crew  # noqa: F401
import mock_services as gc
import multi_agent_tools as mat

from crew.policy import (
    compute_blocks, derive_outcome, fallback_reply, find_leaks, halt_plan, merited_amount,
    reason_from_condition, routing_amount, template_fields,
)
from crew.schemas import FraudAudit, OrderSummary, PolicyCheck, RefundOutcome, RiskReport, TicketFacts, UserSummary


def _report(order_id):
    order = gc.get_order_details(order_id)
    return RiskReport(
        status="complete", ticket_facts=TicketFacts(order_id=order_id), identity_check="match",
        order=OrderSummary.from_record(order), customer=UserSummary.from_record(gc.get_user_profile(order["user_id"])),
        fraud_audit=FraudAudit.model_validate(mat.audit_fraud_risk(order_id)), findings=["x"],
    )


# --- amounts -------------------------------------------------------------------------------------

@pytest.mark.parametrize("total,demanded,expected", [
    (35.0, None, 35.0),      # no number in the ticket: the whole order
    (35.0, 999.0, 35.0),     # greed test: capped at what they paid
    (52.0, 52.0, 52.0),      # exactly what they paid, one dollar over the cap: not shaved
    (119.0, 59.5, 59.5),     # asked for less than the total (one of two items): honoured
    (35.0, 0.0, 35.0),       # "0" is no number
])
def test_merited_amount(total, demanded, expected):
    assert merited_amount(total, demanded) == expected


def test_routing_amount_is_zero_when_no_money_is_in_play():
    assert routing_amount(True, 78.9) == 78.9
    assert routing_amount(False, 78.9) == 0.0
    assert mat.get_escalation_route(risk_band="low", requested_amount=routing_amount(False, 78.9),
                                    order_status="delayed")["channel_id"] == "CH-LOGISTICS"


def test_reason_from_condition():
    assert reason_from_condition("missing") == "item_missing"
    assert reason_from_condition("damaged_on_arrival") == "damaged_on_arrival"
    assert reason_from_condition("new") is None


# --- blocks and outcomes ---------------------------------------------------------------------------

def test_blocks_on_the_headline_case_name_risk_first_then_policy():
    report = _report("ORD-1005")
    policy = PolicyCheck.model_validate(gc.check_return_policy("ORD-1005", "damaged_on_arrival"))
    blocks = compute_blocks(report, policy, prior_cases=[])
    assert blocks[0] == "risk_report:high"
    assert blocks[1] == "policy:escalation:POL-ESC-01,POL-ESC-02"
    assert derive_outcome(policy, None, blocks) == ("ESCALATION_REQUIRED", "ESCALATED_TO_HUMAN")


def test_ineligible_claim_is_rejected_even_when_risky():
    report = _report("ORD-1013")   # USR-105's older order: high risk AND outside the window
    policy = PolicyCheck.model_validate(gc.check_return_policy("ORD-1013", "damaged_on_arrival"))
    blocks = compute_blocks(report, policy, prior_cases=[])
    assert blocks[0] == "policy:verdict:OUTSIDE_RETURN_WINDOW" and "risk_report:high" in blocks
    assert derive_outcome(policy, None, blocks) == ("REJECTED", "REJECTED")


def test_duplicate_claim_from_long_term_memory_is_rejected():
    report = _report("ORD-1001")
    policy = PolicyCheck.model_validate(gc.check_return_policy("ORD-1001", "damaged_on_arrival"))
    prior = [{"order_id": "ORD-1001", "refund_status": "APPROVED", "refund_id": "RF-1001-3500"}]
    assert compute_blocks(report, policy, prior) == ["memory:DUPLICATE_CLAIM:RF-1001-3500"]
    assert compute_blocks(report, policy, [{"order_id": "ORD-1006", "refund_status": "APPROVED"}]) == []


def test_clean_case_has_no_blocks_and_the_tool_result_decides():
    report = _report("ORD-1001")
    policy = PolicyCheck.model_validate(gc.check_return_policy("ORD-1001", "damaged_on_arrival"))
    assert compute_blocks(report, policy, []) == []
    approved = RefundOutcome.model_validate(gc.process_refund("ORD-1001", 35.0))
    assert derive_outcome(policy, approved, []) == ("APPROVED", "AUTO_REFUND_APPROVED")
    escalated = RefundOutcome.model_validate(gc.process_refund("ORD-1002", 150.0))
    assert derive_outcome(policy, escalated, []) == ("ESCALATION_REQUIRED", "ESCALATED_TO_HUMAN")
    assert derive_outcome(policy, None, []) == ("NONE", "ESCALATED_TO_HUMAN")   # eligible, unblocked, never attempted
    assert derive_outcome(None, None, []) == ("NONE", "ESCALATED_TO_HUMAN")     # policy never established


# --- halts -------------------------------------------------------------------------------------

def test_halt_plans():
    nf = halt_plan("unresolvable", "ORDER_NOT_FOUND")
    assert nf.decision == "NEEDS_MORE_INFO" and not nf.consult_router and nf.forced_channel is None
    mm = halt_plan("identity_mismatch", "USER_ORDER_MISMATCH")
    assert mm.decision == "ESCALATED_TO_HUMAN" and mm.consult_router and mm.forced_channel == "CH-FRAUD"
    inc = halt_plan("incomplete", "MAX_STEPS_EXCEEDED")
    assert inc.forced_channel == "CH-SUPPORT-T2" and "MAX_STEPS_EXCEEDED" in inc.reply_intent


# --- hygiene -----------------------------------------------------------------------------------

@pytest.mark.parametrize("leak", [
    "Per POL-ESC-01 your case is escalated.", "Rules FR-01 and FR-04 fired.", "Routed to CH-FRAUD.",
    "We posted in #fraud-security.", "process_refund returned ESCALATION_REQUIRED.", "Your account was flagged.",
    "This looks like fraud.", "Your risk score is 90.", "This is a high-risk order.", "The activity is suspicious.",
    "הפנייה נחסמה עקב חשד להונאה.",
])
def test_hygiene_catches_internal_terms(leak):
    assert find_leaks(leak)


@pytest.mark.parametrize("clean", [
    "Your refund of 35.00 USD for order ORD-1001 has been approved.",
    "Your request is being reviewed by our team and we will follow up.",
    "We could not find order ORD-2222; could you double-check the number?",
    "הבקשה שלך בבדיקה אצל הצוות שלנו ונחזור אליך בהקדם.",
])
def test_hygiene_passes_customer_language(clean):
    assert find_leaks(clean) == []


def test_fallback_replies_are_clean():
    assert find_leaks(fallback_reply("ORD-1005")) == [] and find_leaks(fallback_reply("ORD-1005", "he")) == []


# --- alert helpers ---------------------------------------------------------------------------------

def test_template_fields_are_read_from_the_kit():
    channels = {c["channel_id"]: c["template"] for c in mat.get_escalation_channels()["channels"]}
    assert template_fields(channels["CH-FRAUD"]) == {
        "order_id", "user_id", "risk_score", "risk_band", "triggered_rules", "requested_amount", "evidence"}
    assert "auto_refund_cap" in template_fields(channels["CH-FINANCE"])
    assert "escalation_reason" in template_fields(channels["CH-SUPPORT-T2"])


# --- identity ------------------------------------------------------------------------------------

def test_names_conflict_is_conservative():
    from crew.policy import names_conflict
    assert names_conflict("Ronen", "Maya Levi")
    assert names_conflict("Ronen Katz", "Maya Levi")
    assert not names_conflict("Maya", "Maya Levi")
    assert not names_conflict("Dani", "Daniel Peretz")
    assert not names_conflict("מאיה", "Maya Levi")       # script mismatch is not evidence
    assert not names_conflict(None, "Maya Levi")


def test_no_refund_requested_outcome():
    from crew.policy import derive_outcome
    policy = PolicyCheck.model_validate(gc.check_return_policy("ORD-1004", "late_delivery"))
    assert derive_outcome(policy, None, ["policy:verdict:ORDER_NOT_REFUNDABLE"], refund_requested=False) == ("NONE", "NO_REFUND_REQUESTED")
    assert derive_outcome(None, None, [], refund_requested=False) == ("NONE", "NO_REFUND_REQUESTED")
    approved = RefundOutcome.model_validate(gc.process_refund("ORD-1001", 35.0))
    assert derive_outcome(policy, approved, [], refund_requested=False) == ("APPROVED", "AUTO_REFUND_APPROVED")   # evidence still wins


# --- verified facts -------------------------------------------------------------------------------

def _known(order_id="ORD-1001", demanded=None, prior=(), approved=True):
    from crew.policy import reply_facts
    from conftest import make_decision, make_report
    report = make_report(order_id, demanded=demanded)
    decision = make_decision(report, prior_cases=prior)
    return reply_facts(report, decision, list(prior))


def test_reply_gate_lets_grounded_facts_through():
    from crew.policy import find_unverified_claims
    known = _known()
    assert find_unverified_claims("Your refund of 35.00 USD for order ORD-1001 has been approved (RF-1001-3500).", **known) == []
    assert find_unverified_claims("We refunded $35 for ORD-1001; we will follow up with you.", **known) == []


def test_reply_gate_catches_unverified_facts():
    from crew.policy import find_unverified_claims
    known = _known()
    assert any("RF-1001-9999" in p for p in find_unverified_claims("Refund RF-1001-9999 is on its way.", **known))
    assert any("ORD-1002" in p for p in find_unverified_claims("Regarding order ORD-1002...", **known))
    assert any("50" in p for p in find_unverified_claims("We refunded $50.", **known))
    assert any("timeline" in p for p in find_unverified_claims("Please allow a few business days.", **known))
    assert any("timeline" in p for p in find_unverified_claims("It will arrive within 3-5 days.", **known))
    assert any("timeline" in p for p in find_unverified_claims("הכסף יגיע תוך 5 ימי עסקים.", **known))


def test_reply_gate_promise_rule():
    from crew.policy import find_unverified_claims
    escalated = _known("ORD-1002", demanded=150.0)
    assert not escalated["refund_approved"]
    assert any("no refund was approved" in p for p in find_unverified_claims("Your refund has been approved.", **escalated))
    assert find_unverified_claims("Your request for 150.00 USD is being reviewed by our team.", **escalated) == []
    prior = [{"order_id": "ORD-1001", "refund_status": "APPROVED", "refund_id": "RF-1001-3500", "approved_amount": 35.0}]
    dup = _known(prior=prior)
    assert not dup["refund_approved"]
    assert find_unverified_claims("Order ORD-1001 was already refunded (RF-1001-3500, 35.00 USD).", **dup) == []
    assert any("no refund was approved" in p for p in find_unverified_claims("Your refund has been processed.", **dup))
