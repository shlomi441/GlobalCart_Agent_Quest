"""Shared fixtures. `isolated_outbox` is the important one: the kit computes
`outbox_path` relative to its own BASE_DIR, so an isolated outbox must live
*inside* the kit's outbox directory — a tmp_path elsewhere makes the kit raise."""

import uuid

import pytest

import crew  # noqa: F401
import multi_agent_tools as mat


@pytest.fixture
def isolated_outbox(monkeypatch):
    path = mat.BASE_DIR / "outbox" / f"alerts-test-{uuid.uuid4().hex[:8]}.jsonl"
    monkeypatch.setattr(mat, "OUTBOX_PATH", path)
    yield path
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# Oracle builders: what a perfect researcher / decision maker would hand over,
# built straight from the kit. Used by dispatcher tests now and by the
# oracle/mutation meta-tests in Phase 5.
# --------------------------------------------------------------------------- #

import mock_services as gc  # noqa: E402

from crew.policy import compute_blocks, derive_outcome, halt_plan, merited_amount, reason_from_condition, routing_amount  # noqa: E402
from crew.schemas import (  # noqa: E402
    Decision, FraudAudit, OrderSummary, PolicyCheck, RefundOutcome, RiskReport, TicketFacts, UserSummary,
)


def make_report(order_id, claimed_user_id=None, demanded=None, refund_requested=True):
    order = gc.get_order_details(order_id)
    if "error" in order:
        return RiskReport(status="unresolvable", error_code="ORDER_NOT_FOUND", identity_check="unverified",
                          ticket_facts=TicketFacts(order_id=order_id, demanded_amount=demanded), findings=["order not found"])
    reason = reason_from_condition(order["items"][0]["condition"])
    facts = TicketFacts(order_id=order_id, claimed_user_id=claimed_user_id, demanded_amount=demanded,
                        refund_requested=refund_requested, return_reason=reason,
                        reason_source="order_record" if reason else "assumed")
    owner = UserSummary.from_record(gc.get_user_profile(order["user_id"]))
    if claimed_user_id and claimed_user_id != order["user_id"]:
        claimant = gc.get_user_profile(claimed_user_id)
        return RiskReport(status="identity_mismatch", error_code="USER_ORDER_MISMATCH", identity_check="mismatch",
                          ticket_facts=facts, order=OrderSummary.from_record(order), customer=owner,
                          claimant=UserSummary.from_record(claimant) if "error" not in claimant else None,
                          findings=["claimed customer does not own the order"])
    audit = FraudAudit.model_validate(mat.audit_fraud_risk(order_id))
    return RiskReport(status="complete", identity_check="match", ticket_facts=facts,
                      order=OrderSummary.from_record(order), customer=owner, fraud_audit=audit,
                      findings=[f"risk {audit.risk_score}/100 {audit.risk_band}"])


def make_decision(report, prior_cases=(), attempt=True):
    """A perfect Agent 2, in code: check policy, compute blocks, pay only when nothing blocks."""
    if report.status != "complete":
        plan = halt_plan(report.status, report.error_code)
        who = report.claimant or report.customer   # a halted case is about the claimant, as in graph.synthesize_decision
        return Decision(order_id=report.order.order_id if report.order else None,
                        user_id=who.user_id if who else None,
                        decision=plan.decision, refund_status="NONE", refund_attempted=False,
                        demanded_amount=report.ticket_facts.demanded_amount,
                        requested_amount=routing_amount(report.ticket_facts.refund_requested,
                                                        merited_amount(report.order.total_amount, report.ticket_facts.demanded_amount)
                                                        if report.order else None),
                        rationale=[plan.reply_intent], synthesized_by_code=True, halt_reason=report.error_code)
    order_id, reason = report.order.order_id, report.ticket_facts.return_reason or "damaged_on_arrival"
    merited = merited_amount(report.order.total_amount, report.ticket_facts.demanded_amount)
    policy = PolicyCheck.model_validate(gc.check_return_policy(order_id, reason))
    blocks = compute_blocks(report, policy, list(prior_cases))
    refund = None
    if not blocks and attempt:
        refund = RefundOutcome.model_validate(gc.process_refund(order_id, merited, reason))
    status, code = derive_outcome(policy, refund, blocks)
    return Decision(order_id=order_id, user_id=report.customer.user_id, decision=code, refund_status=status,
                    refund_attempted=refund is not None, blocked_by=blocks,
                    demanded_amount=report.ticket_facts.demanded_amount, merited_amount=merited,
                    requested_amount=routing_amount(report.ticket_facts.refund_requested, merited),
                    policy=policy, refund=refund, cited_policies=policy.applicable_policies,
                    rationale=[f"verdict {policy.verdict}; blocks {blocks or 'none'}"])
