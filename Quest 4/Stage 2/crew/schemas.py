"""Handoff contracts for the GlobalCart Operations Crew.

Everything that crosses an agent boundary is one of the models below. There are
two kinds, and the distinction is the whole design (decision D4):

* ``*Output`` models — what a *model* (LLM) is asked to return. Small, and never
  the source of a business fact.
* Record models (``RiskReport``, ``Decision``, ``CommsResult``, ``CrewResult``)
  — assembled by *code* from an ``*Output`` plus the actual tool results, so
  every fact in a handoff is something a tool actually said.
  "The model narrates, the code records."

Reading order mirrors the flow:
    TicketFacts -> ResearcherOutput -> RiskReport
                -> DecisionOutput   -> Decision
                -> CommsOutput      -> CommsResult
                -> CrewResult, CrewState (the LangGraph state)

Validation rules live here as Pydantic validators, so a malformed handoff fails
loudly at the boundary instead of confusing the next agent quietly.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

# --------------------------------------------------------------------------- #
# Vocabularies. Every closed set in the system is a Literal, never a free string.
# --------------------------------------------------------------------------- #

AgentName = Literal["researcher", "decision", "comms"]
RiskBand = Literal["low", "medium", "high"]
ReturnReason = Literal["damaged_on_arrival", "wrong_item", "item_missing", "late_delivery", "changed_mind"]
Verdict = Literal["ELIGIBLE", "OUTSIDE_RETURN_WINDOW", "NON_RETURNABLE_CATEGORY", "ORDER_NOT_REFUNDABLE"]

#: Part A's decision vocabulary plus one value Part A never needed (decision D9): a ticket that asks for no
#: money has nothing to approve, reject or escalate — the router, not the decision, carries its action.
DecisionCode = Literal["AUTO_REFUND_APPROVED", "REJECTED", "ESCALATED_TO_HUMAN", "NEEDS_MORE_INFO", "NO_REFUND_REQUESTED"]
#: The kit's refund vocabulary plus NONE ("no refund could even be considered"),
#: which is exactly what the kit's own api_docs expect for the mismatch scenario.
RefundStatus = Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED", "NONE"]

ReportStatus = Literal["complete", "identity_mismatch", "unresolvable", "incomplete"]
IdentityCheck = Literal["match", "mismatch", "unverified"]
ReasonSource = Literal["ticket", "order_record", "assumed"]
Sentiment = Literal["calm", "frustrated", "angry"]


#: Words a model naturally uses for the three sentiment values (incident 9): coerced, not rejected.
SENTIMENT_SYNONYMS = {"neutral": "calm", "positive": "calm", "polite": "calm", "content": "calm",
                      "upset": "frustrated", "annoyed": "frustrated", "disappointed": "frustrated",
                      "impatient": "frustrated", "furious": "angry", "outraged": "angry", "hostile": "angry"}


class Strict(BaseModel):
    """Base for every contract: unknown keys are a bug, not a feature."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Agent 1 — what the model returns, and what code builds from it
# --------------------------------------------------------------------------- #

class TicketFacts(Strict):
    """What Agent 1 read in the ticket text. Claims to verify — never facts."""

    order_id: Optional[str] = None
    claimed_user_id: Optional[str] = None      # only if the ticket literally states an id
    claimed_name: Optional[str] = None
    refund_requested: bool = True              # False for "where is my order?"-type tickets
    demanded_amount: Optional[float] = Field(default=None, ge=0)  # the number the customer wrote, if any
    return_reason: Optional[ReturnReason] = None
    reason_source: ReasonSource = "assumed"    # the disclosed-assumption rule from Part A, made a field
    language: str = "en"                       # the reply must come back in this language
    sentiment: Sentiment = "calm"

    @field_validator("order_id", "claimed_user_id", mode="before")
    @classmethod
    def _normalise_ids(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @field_validator("demanded_amount", mode="before")
    @classmethod
    def _amount_from_text(cls, value: Any) -> Any:
        """'$300', '300 USD', '480.0' -> 300.0 / 480.0; anything without a number stays as is (and fails)."""
        if isinstance(value, str):
            match = re.search(r"\d+(?:[.,]\d+)?", value)
            return float(match.group(0).replace(",", ".")) if match else value
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _null_language_is_english(cls, value: Any) -> Any:
        return "en" if value is None or (isinstance(value, str) and not value.strip()) else value

    @field_validator("reason_source", mode="before")
    @classmethod
    def _null_source_is_assumed(cls, value: Any) -> Any:
        """Incident 9: a model with no reason naturally writes null here; that means 'assumed'."""
        return "assumed" if value is None else value

    @field_validator("sentiment", mode="before")
    @classmethod
    def _coerce_sentiment(cls, value: Any) -> Any:
        if isinstance(value, str):
            key = value.strip().lower()
            return SENTIMENT_SYNONYMS.get(key, key)
        return value

    @model_validator(mode="after")
    def _reason_source_needs_a_reason(self) -> "TicketFacts":
        if self.return_reason is None and self.reason_source != "assumed":
            raise ValueError("reason_source can only be 'ticket' or 'order_record' when return_reason is set")
        return self


class ResearcherOutput(Strict):
    """Agent 1's final answer. Facts about the world are NOT here — they are in the tool log."""

    ticket_facts: TicketFacts
    identity_check: IdentityCheck              # ticket's claimed identity vs. the order owner's profile
    findings: list[str] = Field(default_factory=list)  # short, grounded statements for the audit trail
    tools_called: list[str] = Field(default_factory=list)  # self-report; the loop compares it with reality


class OrderItem(Strict):
    sku: str
    name: str
    category: str
    qty: int
    unit_price: float
    condition: str


class OrderSummary(Strict):
    """The order record minus PII (shipping address, payment method) — downstream
    agents never need those, so they never see them."""

    order_id: str
    user_id: str
    status: str
    order_date: str
    delivery_date: Optional[str]
    total_amount: float
    currency: str
    items: list[OrderItem]
    address_changed_at: Optional[str]

    @classmethod
    def from_record(cls, order: dict[str, Any]) -> "OrderSummary":
        return cls(
            order_id=order["order_id"], user_id=order["user_id"], status=order["status"],
            order_date=order["order_date"], delivery_date=order.get("delivery_date"),
            total_amount=float(order["total_amount"]), currency=order["currency"],
            items=[OrderItem(**{k: item[k] for k in OrderItem.model_fields}) for item in order["items"]],
            address_changed_at=order.get("address_changed_at"),
        )


class UserSummary(Strict):
    """The customer profile minus contact details."""

    user_id: str
    name: str
    tier: str
    account_created_at: str
    prior_fraud_flags: int
    initial_fraud_score: int
    refund_history: list[dict[str, Any]]
    sentiment_history: str

    @classmethod
    def from_record(cls, user: dict[str, Any]) -> "UserSummary":
        return cls(**{k: user[k] for k in cls.model_fields})


class TriggeredRule(Strict):
    rule_id: str = Field(pattern=r"^FR-\d{2}$")
    name: str
    weight: int
    why: str


class FraudAudit(Strict):
    """`audit_fraud_risk`'s report, verbatim. `extra='forbid'` plus every field
    required means: if the kit ever changes its shape, we fail loudly instead of
    silently dropping evidence. This object is what "the risk report passes in
    full" means in the README."""

    order_id: str
    user_id: str
    risk_score: int = Field(ge=0, le=100)
    risk_band: RiskBand
    action_hint: str
    triggered_rules: list[TriggeredRule]
    evidence: dict[str, Any]
    blocks_automatic_refund: bool
    requires_security_channel: bool
    rulebook_version: str

    @model_validator(mode="after")
    def _engine_contract(self) -> "FraudAudit":
        if self.blocks_automatic_refund != (self.risk_band == "high"):
            raise ValueError("blocks_automatic_refund must be true exactly when risk_band is 'high'")
        if self.risk_score != min(100, sum(r.weight for r in self.triggered_rules)):
            raise ValueError("risk_score must equal the capped sum of triggered rule weights")
        return self


class RiskReport(Strict):
    """Agent 1 -> Agent 2. Assembled by code from ResearcherOutput + the tool log."""

    status: ReportStatus
    error_code: Optional[str] = None           # ORDER_NOT_FOUND, USER_ORDER_MISMATCH, IDENTITY_MISMATCH, MAX_STEPS_EXCEEDED, ...
    ticket_facts: TicketFacts
    identity_check: IdentityCheck
    order: Optional[OrderSummary] = None
    customer: Optional[UserSummary] = None     # the order owner's profile
    claimant: Optional[UserSummary] = None     # the ticket's claimed user, when that is somebody else
    fraud_audit: Optional[FraudAudit] = None   # the engine's report, verbatim
    findings: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)  # from the log, distinct, first-call order

    @model_validator(mode="after")
    def _status_and_payload_agree(self) -> "RiskReport":
        if self.status == "complete":
            missing = [n for n in ("order", "customer", "fraud_audit") if getattr(self, n) is None]
            if missing:
                raise ValueError(f"a complete report must carry {missing}")
            if self.error_code is not None:
                raise ValueError("a complete report cannot carry an error_code")
            if self.identity_check == "mismatch":
                raise ValueError("an identity mismatch is never a complete report")
        else:
            if self.error_code is None:
                raise ValueError(f"status '{self.status}' requires an error_code")
        if self.status == "unresolvable" and self.order is not None:
            raise ValueError("an unresolvable report has no established order")
        if self.status == "identity_mismatch" and self.identity_check != "mismatch":
            raise ValueError("status identity_mismatch requires identity_check == 'mismatch'")
        return self


# --------------------------------------------------------------------------- #
# Agent 2 — what the model returns, and what code builds from it
# --------------------------------------------------------------------------- #

class DecisionOutput(Strict):
    """Agent 2's final answer. Note what is NOT here: refund status, amount, id.
    Those are read from the tool log by code — the Decision maker cannot report a
    refund that was not made, because the fields are not its to write."""

    decision: DecisionCode                     # a claim; code derives the real one and compares
    rationale: list[str] = Field(min_length=1)
    cited_policies: list[str] = Field(default_factory=list)  # POL-* ids; grounded against tool results
    tools_called: list[str] = Field(default_factory=list)

    @field_validator("cited_policies")
    @classmethod
    def _policy_id_shape(cls, ids: list[str]) -> list[str]:
        bad = [i for i in ids if not (i.startswith("POL-") and i.count("-") == 2)]
        if bad:
            raise ValueError(f"not policy ids: {bad}")
        return ids


class PolicyCheck(BaseModel):
    """`check_return_policy`'s result. The kit returns a different key set per
    verdict, so this model types the keys the crew relies on and keeps the rest
    (`extra='allow'`) rather than lose them."""

    model_config = ConfigDict(extra="allow")

    order_id: str
    verdict: Verdict
    eligible: bool
    requires_escalation: bool
    applicable_policies: list[str]
    explanation: str
    escalation_reasons: list[str] = Field(default_factory=list)
    auto_refund_cap_usd: Optional[float] = None
    days_since_delivery: Optional[int] = None


class RefundOutcome(Strict):
    """`process_refund`'s result, verbatim."""

    status: Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]
    order_id: str
    user_id: str
    requested_amount: float
    approved_amount: float
    auto_refund_cap_usd: float
    applicable_policies: list[str]
    reasons: list[str]
    message: str
    refund_id: Optional[str] = None

    @model_validator(mode="after")
    def _approval_has_an_id(self) -> "RefundOutcome":
        approved = self.status == "APPROVED"
        if approved != (self.refund_id is not None and self.approved_amount > 0):
            raise ValueError("APPROVED must come with a refund_id and a positive approved_amount, and only then")
        return self


class Decision(Strict):
    """Agent 2 -> Agent 3. Assembled by code from DecisionOutput + the tool log.
    On a halt path (Agent 2 never ran) it is synthesized entirely by code."""

    order_id: Optional[str]
    user_id: Optional[str]
    decision: DecisionCode                      # derived from evidence by code (policy.derive_outcome)
    claimed_decision: Optional[DecisionCode] = None   # what the model said; kept as a measurement, never as the outcome
    refund_status: RefundStatus
    refund_attempted: bool
    blocked_by: list[str] = Field(default_factory=list)   # e.g. "risk_report:high", "policy:verdict:OUTSIDE_RETURN_WINDOW"
    demanded_amount: Optional[float] = None     # what the customer wrote
    merited_amount: Optional[float] = None      # what the claim is worth (policy.merited_amount)
    requested_amount: float = 0.0               # the amount in play — the router's input
    policy: Optional[PolicyCheck] = None
    refund: Optional[RefundOutcome] = None
    cited_policies: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(min_length=1)
    tools_called: list[str] = Field(default_factory=list)
    synthesized_by_code: bool = False
    halt_reason: Optional[str] = None

    @model_validator(mode="after")
    def _one_story(self) -> "Decision":
        """Decision, refund_status and the tool evidence must tell the same story."""
        if self.refund_attempted != (self.refund is not None):
            raise ValueError("refund_attempted must mirror the presence of a process_refund result")
        if self.refund is not None and self.refund.status != self.refund_status:
            raise ValueError("refund_status must equal what process_refund actually returned")
        if self.refund_status == "APPROVED" and self.refund is None:
            raise ValueError("APPROVED needs a process_refund result as evidence")
        expected = {
            "APPROVED": "AUTO_REFUND_APPROVED",
            "REJECTED": "REJECTED",
            "ESCALATION_REQUIRED": "ESCALATED_TO_HUMAN",
        }
        if self.refund_status in expected and self.decision != expected[self.refund_status]:
            raise ValueError(f"refund_status {self.refund_status} requires decision {expected[self.refund_status]}")
        if self.refund_status == "NONE":
            if self.decision not in ("NEEDS_MORE_INFO", "ESCALATED_TO_HUMAN", "NO_REFUND_REQUESTED"):
                raise ValueError("refund_status NONE is only compatible with NEEDS_MORE_INFO, ESCALATED_TO_HUMAN or NO_REFUND_REQUESTED")
        if self.decision == "NO_REFUND_REQUESTED" and (self.refund_status != "NONE" or self.blocked_by):
            raise ValueError("NO_REFUND_REQUESTED means nothing was considered: refund_status NONE and no blocks")
            if self.refund is not None:
                raise ValueError("refund_status NONE cannot carry a process_refund result")
        if self.refund is None and self.refund_status in ("ESCALATION_REQUIRED", "REJECTED") and not self.blocked_by:
            raise ValueError("a refund that was never attempted needs a recorded reason in blocked_by")
        if self.synthesized_by_code and self.halt_reason is None:
            raise ValueError("a code-synthesized decision must state its halt_reason")
        return self


# --------------------------------------------------------------------------- #
# Agent 3 — what the model returns, and what code builds from it
# --------------------------------------------------------------------------- #

class CommsOutput(Strict):
    """Agent 3's final answer: the customer reply only. The alert is a tool call,
    so its evidence lives in the tool log, not in prose."""

    customer_reply: str = Field(min_length=1)
    reply_language: str = "en"
    tools_called: list[str] = Field(default_factory=list)


class RouteResult(Strict):
    """`get_escalation_route`'s result, plus one crew field: `override_reason`,
    set when crew policy (decision D3) overrode a 'no escalation' answer."""

    escalation_required: bool
    channel_id: Optional[str]
    channel: Optional[str]
    owner_team: Optional[str]
    severity: Optional[str]
    priority: Optional[int]
    response_sla_minutes: Optional[int]
    template: Optional[str]
    matched_condition: str
    override_reason: Optional[str] = None

    @model_validator(mode="after")
    def _channel_iff_required(self) -> "RouteResult":
        if self.escalation_required != (self.channel_id is not None):
            raise ValueError("escalation_required must be true exactly when a channel_id is present")
        return self


class AlertReceipt(Strict):
    """`send_slack_alert`'s result, plus the payload the crew actually sent."""

    delivered: bool
    channel_id: str
    channel: str
    severity: str
    message_ts: str
    transport: str
    outbox_path: str
    webhook_status: Optional[int]
    message: str
    payload: dict[str, Any]


class CommsResult(Strict):
    """Agent 3 -> the outside world. Assembled by code from CommsOutput + the tool log."""

    route: Optional[RouteResult] = None        # None when routing was skipped by crew policy (no established case)
    alert: Optional[AlertReceipt] = None
    customer_reply: str = Field(min_length=1)
    reply_language: str = "en"
    tools_called: list[str] = Field(default_factory=list)
    fallback_used: bool = False                # True when code, not the model, produced the reply/alert

    @model_validator(mode="after")
    def _alert_iff_routed(self) -> "CommsResult":
        required = self.route is not None and self.route.escalation_required
        if self.alert is not None and not required:
            raise ValueError("an alert was sent for a case the router did not escalate")
        if self.alert is not None and self.alert.channel_id != self.route.channel_id:
            raise ValueError("the alert went to a different channel than the route")
        return self


# --------------------------------------------------------------------------- #
# Cross-cutting: the audit trail and the graph state
# --------------------------------------------------------------------------- #

class ToolCall(Strict):
    """One dispatched call. `synthetic` marks results the dispatcher generated
    itself (locks, guards) instead of running the tool."""

    agent: AgentName
    step: int
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    synthetic: bool = False

    @property
    def error(self) -> Optional[str]:
        return self.result.get("error") if isinstance(self.result, dict) else None


class AgentRun(Strict):
    """What the loop reports about one agent's run — the honesty metrics live here."""

    agent: AgentName
    steps: int
    format_retries: int
    error: Optional[str] = None                # MAX_STEPS_EXCEEDED, INVALID_OUTPUT, REPLY_HYGIENE_VIOLATION, ...
    claimed_tools: list[str] = Field(default_factory=list)
    executed_tools: list[str] = Field(default_factory=list)   # distinct, first-call order, real tools only
    honest: Optional[bool] = True              # claimed == executed; None when the agent produced no output to compare
    retry_details: list[str] = Field(default_factory=list)   # why each format/hygiene retry happened
    anomalies: list[str] = Field(default_factory=list)       # API/model oddities the loop absorbed (e.g. empty turns)


class CrewResult(Strict):
    """The crew's final product for one ticket."""

    ticket: str
    run_id: str
    risk_report: RiskReport
    decision: Decision
    comms: CommsResult
    tool_log: list[ToolCall]
    agent_runs: list[AgentRun]
    notes: list[str] = Field(default_factory=list)   # crew-level observations: fills, overrides, discrepancies
    halt_reason: Optional[str] = None

    def to_part_a(self) -> dict[str, Any]:
        """The Part A output contract, so the Part A regression suite reads Part B output unchanged."""
        distinct: list[str] = []
        for call in self.tool_log:
            if not call.synthetic and call.tool not in distinct:
                distinct.append(call.tool)
        refund = self.decision.refund
        approved = refund is not None and refund.status == "APPROVED"
        return {
            "reasoning_chain": [f"[researcher] {f}" for f in self.risk_report.findings]
            + [f"[decision] {r}" for r in self.decision.rationale],
            "action_taken": {
                "tools_called": distinct,
                "decision": self.decision.decision,
                "refund_amount": refund.approved_amount if approved else 0.0,
                "refund_id": refund.refund_id if approved else None,
            },
            "customer_response": self.comms.customer_reply,
        }


class CrewState(TypedDict, total=False):
    """The LangGraph state. Nodes read it and return partial updates; fields
    annotated with a reducer (`operator.add`) accumulate across nodes instead of
    being overwritten. This type *is* the answer to "what passes between the agents"."""

    ticket: str
    run_id: str
    prior_cases: list[dict[str, Any]]          # long-term memory (bonus): earlier runs on this order/customer
    risk_report: Optional[RiskReport]          # written by researcher
    merited_amount: Optional[float]            # written by triage
    requested_amount: float                    # written by triage
    decision: Optional[Decision]               # written by decision (or synthesized by triage on a halt)
    comms: Optional[CommsResult]               # written by comms
    halt_reason: Optional[str]                 # written by triage / any failing node
    tool_log: Annotated[list[ToolCall], operator.add]
    agent_runs: Annotated[list[AgentRun], operator.add]
    notes: Annotated[list[str], operator.add]
