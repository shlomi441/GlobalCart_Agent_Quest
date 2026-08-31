"""Crew policy: the rules the crew adds on top of the kit's tools.

Everything here is a pure function — no model, no I/O, no kit imports — so each
rule can be pinned by a unit test with no API key. The dispatchers (dispatch.py)
and the graph (graph.py) *apply* these rules; this module *defines* them. When
the README says "where is this enforced", the answer is a function in this file
and the dispatcher line that calls it.
"""

from __future__ import annotations

import re
from string import Formatter
from typing import Any, Optional

from pydantic import Field

from crew.schemas import (
    Decision, DecisionCode, PolicyCheck, RefundOutcome, RefundStatus, ReportStatus, ReturnReason,
    RiskReport, Strict,
)

# --------------------------------------------------------------------------- #
# 1. Amounts — decision D5, the mechanical form of Part A's "never split" rule
# --------------------------------------------------------------------------- #

def merited_amount(order_total: float, demanded: Optional[float]) -> float:
    """What the claim is worth: what the customer paid, or less if they asked for less.

    A demand above the order total is capped at the total (the greed test);
    no stated amount means the whole order. This is the ONLY amount Agent 2 may
    pass to process_refund, so a claim above the cap cannot be shaved to fit —
    the shave is refused before the kit ever sees it.
    """
    if order_total <= 0:
        raise ValueError("order_total must be positive")
    if demanded is None or demanded <= 0:
        return round(order_total, 2)
    return round(min(demanded, order_total), 2)


def routing_amount(refund_requested: bool, merited: Optional[float]) -> float:
    """The amount the router should see: the amount actually in play. A "where is
    my order?" ticket puts no money in play, which is what lets the router reach
    #logistics-delays instead of Tier 2."""
    if not refund_requested or merited is None:
        return 0.0
    return merited


# --------------------------------------------------------------------------- #
# 2. Return reason — the order record beats the ticket, the ticket beats a guess
# --------------------------------------------------------------------------- #

_CONDITION_TO_REASON: dict[str, ReturnReason] = {
    "damaged_on_arrival": "damaged_on_arrival",
    "wrong_item": "wrong_item",
    "missing": "item_missing",
}


def reason_from_condition(condition: str) -> Optional[ReturnReason]:
    """Map an order item's recorded condition to a policy return reason.
    'new' maps to nothing: the record does not say why the customer wants a return."""
    return _CONDITION_TO_REASON.get(condition)


# --------------------------------------------------------------------------- #
# 3. Blocks — why the crew refuses to attempt a refund (decision D2)
# --------------------------------------------------------------------------- #

def compute_blocks(
    report: RiskReport,
    policy: Optional[PolicyCheck],
    prior_cases: list[dict[str, Any]],
) -> list[str]:
    """Return the reasons, in precedence order, that forbid calling process_refund.

    Precedence mirrors the kit's own process_refund: eligibility first, then risk
    and policy escalation, then the crew's long-term memory. The first entry is
    the headline reason; all of them are kept for the audit trail.
    """
    blocks: list[str] = []
    if policy is not None and not policy.eligible:
        blocks.append(f"policy:verdict:{policy.verdict}")
    if report.fraud_audit is not None and report.fraud_audit.blocks_automatic_refund:
        blocks.append(f"risk_report:{report.fraud_audit.risk_band}")
    if policy is not None and policy.requires_escalation:
        ids = [p for p in policy.applicable_policies if p.startswith("POL-ESC-")]
        blocks.append("policy:escalation:" + ",".join(ids))
    order_id = report.order.order_id if report.order else None
    for case in prior_cases:
        if case.get("order_id") == order_id and case.get("refund_status") == "APPROVED":
            blocks.append(f"memory:DUPLICATE_CLAIM:{case.get('refund_id')}")
            break
    return blocks


def derive_outcome(
    policy: Optional[PolicyCheck],
    refund: Optional[RefundOutcome],
    blocked_by: list[str],
    refund_requested: bool = True,
) -> tuple[RefundStatus, DecisionCode]:
    """The refund status and decision code, derived from evidence — never from the model.

    Order of truth: a real process_refund result beats everything; then "no money was
    asked for" (decision D9: nothing to approve, reject or escalate); then the recorded
    blocks; then a policy verdict; and if an eligible, unblocked claim somehow reached
    the end with no refund attempted, that is an incomplete decision and a human gets it.
    """
    if refund is None and not refund_requested:
        return ("NONE", "NO_REFUND_REQUESTED")
    if refund is not None:
        return {
            "APPROVED": ("APPROVED", "AUTO_REFUND_APPROVED"),
            "REJECTED": ("REJECTED", "REJECTED"),
            "ESCALATION_REQUIRED": ("ESCALATION_REQUIRED", "ESCALATED_TO_HUMAN"),
        }[refund.status]
    if blocked_by:
        head = blocked_by[0]
        if head.startswith("policy:verdict:") or head.startswith("memory:"):
            return ("REJECTED", "REJECTED")
        return ("ESCALATION_REQUIRED", "ESCALATED_TO_HUMAN")
    if policy is not None and not policy.eligible:
        return ("REJECTED", "REJECTED")
    return ("NONE", "ESCALATED_TO_HUMAN")


# --------------------------------------------------------------------------- #
# 4. Halts — what happens when a stage cannot complete (the stop condition's other half)
# --------------------------------------------------------------------------- #

class HaltPlan(Strict):
    decision: DecisionCode
    refund_status: RefundStatus = "NONE"
    consult_router: bool                       # False = the case is not established; never route, never alert
    forced_channel: Optional[str] = None       # crew override if the router does not escalate on its own
    reply_intent: str                          # one line the comms agent builds the reply around


def halt_plan(status: ReportStatus, error_code: Optional[str]) -> HaltPlan:
    """Map a non-complete report (or a failed later stage) to a defined outcome.

    The graph has no edge back to the researcher; every failure flows forward
    through one of these plans. That is the "escalate, do not re-dispatch" rule.
    """
    if status == "unresolvable":
        return HaltPlan(
            decision="NEEDS_MORE_INFO", consult_router=False,
            reply_intent="The order could not be found; ask the customer to confirm the order number. "
                         "No refund was considered.",
        )
    if status == "identity_mismatch":
        return HaltPlan(
            decision="ESCALATED_TO_HUMAN", consult_router=True, forced_channel="CH-FRAUD",
            reply_intent="Say the request for this order is under review by our team and we will follow up. "
                         "You may say the order could not be matched to the account details provided. "
                         "Never mention fraud, flags, or why the case is reviewed.",
        )
    return HaltPlan(  # "incomplete": an agent hit MAX_STEPS, returned invalid output, or a later stage failed
        decision="ESCALATED_TO_HUMAN", consult_router=True, forced_channel="CH-SUPPORT-T2",
        reply_intent="The request is under review by our team; a colleague will follow up. "
                     f"(internal cause: {error_code or 'unknown'})",
    )


# --------------------------------------------------------------------------- #
# 5. Reply hygiene — the runtime gate from Part A, widened for Part B's vocabulary
# --------------------------------------------------------------------------- #

#: (label, pattern). Labels make the leak report readable; patterns are matched case-insensitively.
HYGIENE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in [
        ("policy id", r"POL-[A-Z]+-\d+"),
        ("fraud rule id", r"\bFR-\d{2}\b"),
        ("channel id", r"\bCH-[A-Z0-9-]+\b"),
        ("channel name", r"#(?:fraud-security|finance-approvals|support-tier2|logistics-delays)"),
        ("tool name", r"\b(?:get_order_details|get_user_profile|check_return_policy|process_refund"
                      r"|audit_fraud_risk|get_escalation_route|send_slack_alert)\b"),
        ("fraud", r"\bfraud\w*\b"),
        ("flag", r"\bflag(?:ged|ging|s)?\b"),
        ("risk score/band", r"\brisk\s+(?:score|band|report|level|rating)\b|\bhigh[- ]risk\b"),
        ("suspicion", r"\bsuspicious\b|\bsuspect(?:ed)?\b"),
        ("rulebook", r"\brulebook\b|\btriggered\s+rules?\b"),
        ("hebrew: fraud/suspicion/flag/risk score", r"הונאה|מרמה|חשוד|חשד|דגל|ציון\s+סיכון"),
    ]
]


def find_leaks(text: str) -> list[str]:
    """Return the internal terms present in a customer-facing text, labelled."""
    leaks: list[str] = []
    for label, pattern in HYGIENE_PATTERNS:
        for match in pattern.finditer(text):
            leaks.append(f"{label}: {match.group(0)}")
    return leaks


# --------------------------------------------------------------------------- #
# 5a. Verified facts — the reply may only state what the evidence supports
# --------------------------------------------------------------------------- #

ORDER_ID_RE = re.compile(r"\bORD-\d+\b")
REFUND_ID_RE = re.compile(r"\bRF-\d+-\d+\b")
AMOUNT_RE = re.compile(r"\$\s?(\d+(?:[.,]\d+)?)|\b(\d+(?:[.,]\d+)?)\s?(?:USD|dollars?)\b", re.IGNORECASE)
TIMELINE_RE = re.compile(
    r"\b\d+\s*(?:-|–|to)\s*\d+\s*(?:business |working )?(?:days?|hours?)\b"
    r"|\b\d+\s+(?:business |working )?(?:days?|hours?)\b"
    r"|\ba few (?:business |working )?days\b|\bbusiness days\b|\bworking days\b|\bprocessing time\b"
    r"|ימי עסקים|תוך \d+ ימים",
    re.IGNORECASE,
)
PROMISE_RE = re.compile(r"\b(?:has been|have been|was|were|is|are|will be|has|have)(?:\s+\w+){0,2}\s+"
                        r"(?:approved|issued|credited|processed|refunded|paid)\b", re.IGNORECASE)


def reply_facts(report: RiskReport, decision: "Decision", prior_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything a customer reply is allowed to state as a fact, collected from the run's evidence."""
    facts = report.ticket_facts
    order_ids = {i for i in (facts.order_id, report.order.order_id if report.order else None) if i}
    refund_ids = {c.get("refund_id") for c in prior_cases if c.get("refund_id")}
    refund_ids |= {b.rsplit(":", 1)[1] for b in decision.blocked_by if b.startswith("memory:DUPLICATE_CLAIM:")}
    if decision.refund is not None and decision.refund.refund_id:
        refund_ids.add(decision.refund.refund_id)
    amounts = {facts.demanded_amount, decision.merited_amount, decision.requested_amount}
    if report.order:
        amounts.add(report.order.total_amount)
        amounts.update(i.unit_price for i in report.order.items)
        amounts.update(i.unit_price * i.qty for i in report.order.items)
    if decision.refund is not None:
        amounts.update({decision.refund.approved_amount, decision.refund.requested_amount})
    amounts.update(c.get("approved_amount") for c in prior_cases)
    return {"order_ids": order_ids, "refund_ids": refund_ids,
            "amounts": {float(a) for a in amounts if a is not None},
            "refund_approved": decision.refund_status == "APPROVED"}


def find_unverified_claims(text: str, *, order_ids: set[str], refund_ids: set[str], amounts: set[float],
                           refund_approved: bool) -> list[str]:
    """Facts in a customer-facing text that no tool result supports, labelled."""
    problems: list[str] = []
    for oid in sorted(set(ORDER_ID_RE.findall(text)) - order_ids):
        problems.append(f"order id {oid} is not this case's order")
    for rid in sorted(set(REFUND_ID_RE.findall(text)) - refund_ids):
        problems.append(f"refund id {rid} does not exist in the evidence")
    for match in AMOUNT_RE.finditer(text):
        raw = (match.group(1) or match.group(2)).replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if not any(abs(value - a) <= 0.005 for a in amounts):
            problems.append(f"amount {raw} USD does not appear in any tool result")
    for match in TIMELINE_RE.finditer(text):
        problems.append(f"timeline {match.group(0)!r} - no tool result supports a processing time")
    if not refund_approved:
        mentions_prior = bool(set(REFUND_ID_RE.findall(text)) & refund_ids)
        for match in PROMISE_RE.finditer(text):
            if not mentions_prior:
                problems.append(f"{match.group(0)!r} - no refund was approved in this case")
                break
    return problems


# --------------------------------------------------------------------------- #
# 5b. Identity — a name-only mismatch the tools cannot see
# --------------------------------------------------------------------------- #

def names_conflict(claimed: Optional[str], owner: Optional[str]) -> bool:
    """True when the name in the ticket cannot be the order owner's name.

    Deliberately conservative: only Latin-script names are compared (a Hebrew
    ticket for an English profile is not a conflict), and a token that is a
    prefix of the other side's token matches, so "Dani" fits "Daniel Peretz" and
    "Maya" fits "Maya Levi". The model's own identity_check still counts; this
    is the code-side cross-check that turns a missed conflict into a halt.
    """
    if not claimed or not owner:
        return False
    if not (claimed.isascii() and owner.isascii()):
        return False
    left = [t for t in re.findall(r"[a-z]+", claimed.lower()) if len(t) >= 3]
    right = [t for t in re.findall(r"[a-z]+", owner.lower()) if len(t) >= 3]
    if not left or not right:
        return False
    return not any(a.startswith(b) or b.startswith(a) for a in left for b in right)


# --------------------------------------------------------------------------- #
# 6. Grounding and alert helpers
# --------------------------------------------------------------------------- #

POLICY_ID = re.compile(r"POL-[A-Z]+-\d+")
RULE_ID = re.compile(r"\bFR-\d{2}\b")


def template_fields(template: str) -> set[str]:
    """The placeholders a channel template renders — the keys an alert payload must carry.
    Data-driven from the template text, so a channel change in the kit changes the check."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def fallback_reply(order_id: Optional[str], language: str = "en") -> str:
    """The reply used when Agent 3 cannot produce a clean one. Deliberately generic."""
    if language.lower().startswith("he"):
        ref = f" בנוגע להזמנה {order_id}" if order_id else ""
        return f"תודה שפנית אלינו{ref}. הפנייה שלך בבדיקה אצל הצוות שלנו ונחזור אליך בהקדם."
    ref = f" about order {order_id}" if order_id else ""
    return f"Thank you for contacting us{ref}. Your request is being reviewed by our team and we will follow up shortly."
