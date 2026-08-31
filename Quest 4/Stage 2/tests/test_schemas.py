"""Phase 1: the handoff contracts accept exactly what the kit produces, and reject inconsistent stories."""

import pytest
from pydantic import ValidationError

import crew  # noqa: F401  (sys.path bootstrap for the kit)
import mock_services as gc
import multi_agent_tools as mat

from crew.schemas import (
    AlertReceipt, Decision, FraudAudit, OrderSummary, PolicyCheck, RefundOutcome, RiskReport,
    RouteResult, TicketFacts, UserSummary,
)


# --- verbatim tool results validate --------------------------------------------------------

@pytest.mark.parametrize("order_id", ["ORD-1005", "ORD-1012", "ORD-1001"])
def test_fraud_audit_accepts_the_engine_verbatim(order_id):
    audit = FraudAudit.model_validate(mat.audit_fraud_risk(order_id))
    assert audit.model_dump() == mat.audit_fraud_risk(order_id)   # nothing added, nothing lost


def test_fraud_audit_rejects_a_tampered_band():
    raw = mat.audit_fraud_risk("ORD-1005")
    raw["risk_band"] = "low"                  # a model "overruling" the engine would look like this
    with pytest.raises(ValidationError):
        FraudAudit.model_validate(raw)


def test_fraud_audit_rejects_a_tampered_score():
    raw = mat.audit_fraud_risk("ORD-1005")
    raw["risk_score"] = 70
    with pytest.raises(ValidationError):
        FraudAudit.model_validate(raw)


@pytest.mark.parametrize("order_id,reason", [("ORD-1001", "damaged_on_arrival"), ("ORD-1003", "changed_mind"),
                                             ("ORD-1008", "damaged_on_arrival"), ("ORD-1007", "changed_mind")])
def test_policy_check_accepts_every_verdict_shape(order_id, reason):
    raw = gc.check_return_policy(order_id, reason)
    check = PolicyCheck.model_validate(raw)
    assert check.verdict == raw["verdict"]
    assert check.model_dump() == raw           # extra keys are kept, not dropped


@pytest.mark.parametrize("order_id,amount", [("ORD-1001", 35.0), ("ORD-1002", 150.0), ("ORD-1001", 999.0), ("ORD-1005", 480.0)])
def test_refund_outcome_accepts_every_status(order_id, amount):
    raw = gc.process_refund(order_id, amount)
    out = RefundOutcome.model_validate(raw)
    assert (out.status == "APPROVED") == (out.refund_id is not None)


def test_route_result_accepts_match_and_no_match():
    hit = RouteResult.model_validate(mat.get_escalation_route(risk_band="high"))
    miss = RouteResult.model_validate(mat.get_escalation_route(risk_band="low", requested_amount=35.0))
    assert hit.channel_id == "CH-FRAUD" and miss.channel_id is None and not miss.escalation_required


def test_route_result_rejects_a_channel_without_escalation():
    raw = mat.get_escalation_route(risk_band="low", requested_amount=35.0)
    raw["channel_id"] = "CH-FRAUD"
    with pytest.raises(ValidationError):
        RouteResult.model_validate(raw)


def test_alert_receipt_accepts_the_tool_result(isolated_outbox):
    payload = {"order_id": "ORD-1005", "risk_score": 90}
    raw = mat.send_slack_alert("CH-FRAUD", "critical", payload)
    receipt = AlertReceipt.model_validate({**raw, "payload": payload})
    assert receipt.delivered and receipt.message_ts


def test_summaries_drop_pii():
    order = OrderSummary.from_record(gc.get_order_details("ORD-1005"))
    user = UserSummary.from_record(gc.get_user_profile("USR-105"))
    assert "shipping_address" not in order.model_dump() and "payment_method_last4" not in order.model_dump()
    assert "email" not in user.model_dump()
    assert order.items[0].condition == "damaged_on_arrival" and user.prior_fraud_flags == 1


# --- cross-field stories -----------------------------------------------------------------------

def _complete_report(order_id="ORD-1005"):
    order = gc.get_order_details(order_id)
    return RiskReport(
        status="complete", ticket_facts=TicketFacts(order_id=order_id), identity_check="match",
        order=OrderSummary.from_record(order), customer=UserSummary.from_record(gc.get_user_profile(order["user_id"])),
        fraud_audit=FraudAudit.model_validate(mat.audit_fraud_risk(order_id)), findings=["x"],
    )


def test_complete_report_requires_all_three_parts():
    _complete_report()
    with pytest.raises(ValidationError):
        RiskReport(status="complete", ticket_facts=TicketFacts(), identity_check="match")


def test_halt_reports_require_a_code_and_the_right_identity_check():
    RiskReport(status="unresolvable", error_code="ORDER_NOT_FOUND", ticket_facts=TicketFacts(order_id="ORD-9999"),
               identity_check="unverified")
    with pytest.raises(ValidationError):
        RiskReport(status="unresolvable", ticket_facts=TicketFacts(), identity_check="unverified")
    with pytest.raises(ValidationError):
        RiskReport(status="identity_mismatch", error_code="USER_ORDER_MISMATCH", ticket_facts=TicketFacts(),
                   identity_check="match")


def test_ticket_facts_normalise_ids_and_guard_reason_source():
    facts = TicketFacts(order_id=" ord-1005 ", claimed_user_id="")
    assert facts.order_id == "ORD-1005" and facts.claimed_user_id is None
    with pytest.raises(ValidationError):
        TicketFacts(reason_source="ticket")   # a source without a reason


def _decision(**overrides):
    base = dict(order_id="ORD-1001", user_id="USR-101", decision="AUTO_REFUND_APPROVED", refund_status="APPROVED",
                refund_attempted=True, refund=RefundOutcome.model_validate(gc.process_refund("ORD-1001", 35.0)),
                rationale=["ok"])
    return Decision(**{**base, **overrides})


def test_decision_cannot_report_a_refund_the_tool_did_not_make():
    _decision()
    escalated = RefundOutcome.model_validate(gc.process_refund("ORD-1002", 150.0))
    with pytest.raises(ValidationError):
        _decision(refund=escalated)                                   # says APPROVED, tool said ESCALATION_REQUIRED
    with pytest.raises(ValidationError):
        _decision(refund=None, refund_attempted=False)                # APPROVED with no evidence at all


def test_unattempted_refund_needs_a_recorded_reason():
    with pytest.raises(ValidationError):
        _decision(decision="ESCALATED_TO_HUMAN", refund_status="ESCALATION_REQUIRED", refund=None, refund_attempted=False)
    _decision(decision="ESCALATED_TO_HUMAN", refund_status="ESCALATION_REQUIRED", refund=None, refund_attempted=False,
              blocked_by=["risk_report:high"])


def test_none_status_only_for_halts():
    _decision(decision="NEEDS_MORE_INFO", refund_status="NONE", refund=None, refund_attempted=False,
              synthesized_by_code=True, halt_reason="ORDER_NOT_FOUND")
    with pytest.raises(ValidationError):
        _decision(decision="REJECTED", refund_status="NONE", refund=None, refund_attempted=False)
    with pytest.raises(ValidationError):   # synthesized without saying why
        _decision(decision="NEEDS_MORE_INFO", refund_status="NONE", refund=None, refund_attempted=False,
                  synthesized_by_code=True)


def test_ticket_facts_accept_natural_model_answers():
    """Incident 9: null reason_source and sentiment synonyms must not cost a retry."""
    facts = TicketFacts(order_id="ORD-2222", return_reason=None, reason_source=None, sentiment="neutral")
    assert facts.reason_source == "assumed" and facts.sentiment == "calm"
    assert TicketFacts(sentiment="Upset").sentiment == "frustrated" and TicketFacts(sentiment="furious").sentiment == "angry"
    with pytest.raises(ValidationError):
        TicketFacts(sentiment="ecstatic")     # unknown words still fail loudly


def test_ticket_facts_parse_amount_text_and_null_language():
    assert TicketFacts(demanded_amount="$300").demanded_amount == 300.0
    assert TicketFacts(demanded_amount="480 USD").demanded_amount == 480.0
    assert TicketFacts(demanded_amount="1,5").demanded_amount == 1.5
    assert TicketFacts(language=None).language == "en"
    with pytest.raises(ValidationError):
        TicketFacts(demanded_amount="a lot")


def test_no_refund_requested_means_nothing_was_considered():
    _decision(decision="NO_REFUND_REQUESTED", refund_status="NONE", refund=None, refund_attempted=False)
    with pytest.raises(ValidationError):
        _decision(decision="NO_REFUND_REQUESTED", refund_status="NONE", refund=None, refund_attempted=False, blocked_by=["risk_report:high"])
    with pytest.raises(ValidationError):
        _decision(decision="NO_REFUND_REQUESTED", refund_status="REJECTED", refund=None, refund_attempted=False, blocked_by=["x"])
