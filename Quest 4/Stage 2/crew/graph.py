"""The crew graph: three agents, two code nodes, no way back.

    START -> researcher -> triage -+-> decision -> comms -> END
                                    \\--------------^
                                     (halt: mismatch / not found / incomplete)

`researcher`, `decision` and `comms` each run one AgentSpec through the shared
loop with that role's dispatcher. `triage` is pure code: it reads the ledger,
computes the amounts, and turns a non-complete report into a synthesized
Decision plus a halt — the stop condition's defined behaviour. There is no edge
from any node back to `researcher`, so an incomplete report can only flow
forward; `recursion_limit` (config) is the tripwire on top of that topology.

Three assemblers implement decision D4 ("the model narrates, the code records"):
every fact in a handoff is read from the dispatcher's log of real tool results;
the model's output contributes ticket facts, rationale and prose. Two fallback
paths keep the crew's obligations intact when a model fails: the customer always
gets a reply, and a required escalation is always alerted — by code, marked as
such (`fallback_used`, tool-log step 0).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Optional

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from crew.agent_loop import AgentRunResult, AgentSpec, run_agent
from crew.config import MODEL, RECURSION_LIMIT
from crew.dispatch import CommsDispatcher, DecisionDispatcher, ResearcherDispatcher, expected_facts
from crew.memory import Ledger
from crew.policy import (
    HaltPlan, compute_blocks, derive_outcome, fallback_reply, find_unverified_claims, halt_plan, merited_amount,
    names_conflict, reason_from_condition, reply_facts, routing_amount, template_fields,
)
from crew.schemas import (
    AgentName, AgentRun, AlertReceipt, CommsResult, CrewResult, CrewState, Decision, FraudAudit, OrderSummary,
    PolicyCheck, RefundOutcome, RiskReport, RouteResult, TicketFacts, UserSummary,
)

FALLBACK_STEP = 0   # tool-log step number reserved for calls the crew made itself, not the model


# --------------------------------------------------------------------------- #
# Briefs — what each agent is shown (decision D6: structured handoffs, never transcripts)
# --------------------------------------------------------------------------- #

def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))


def researcher_brief(ticket: str) -> str:
    return f"Customer ticket:\n\"\"\"\n{ticket}\n\"\"\""


def decision_brief(state: CrewState) -> str:
    return "Risk report from the Researcher, and the crew's computed amounts:\n" + _dump({
        "risk_report": state["risk_report"],
        "merited_amount": state.get("merited_amount"),
        "requested_amount": state.get("requested_amount"),
        "prior_cases": state.get("prior_cases", []),
    })


def comms_brief(state: CrewState, plan: Optional[HaltPlan]) -> str:
    return "Decision and risk report for this case:\n" + _dump({
        "decision": state["decision"],
        "risk_report": state["risk_report"],
        "halt": {"reason": state.get("halt_reason"), "reply_intent": plan.reply_intent} if plan else None,
    })


# --------------------------------------------------------------------------- #
# Assemblers — code records
# --------------------------------------------------------------------------- #

def build_risk_report(res: AgentRunResult, d: ResearcherDispatcher) -> tuple[RiskReport, list[str]]:
    """RiskReport from the researcher's output plus what its tools actually returned."""
    notes: list[str] = []
    out = res.output
    facts = out.ticket_facts if out else TicketFacts()
    if facts.order_id is None:
        looked_up = next((c.args.get("order_id") for c in d.log if c.tool in ("get_order_details", "audit_fraud_risk")), None)
        if looked_up:
            facts = facts.model_copy(update={"order_id": str(looked_up).strip().upper()})
            notes.append(f"[researcher] order_id recovered from the tool log: {facts.order_id}")

    def collect():
        orders = d.ok_results("get_order_details")
        order_rec = next((o for o in orders if o["order_id"] == facts.order_id), orders[-1] if orders else None)
        users = d.ok_results("get_user_profile")
        owner_rec = next((u for u in users if order_rec and u["user_id"] == order_rec["user_id"]), None)
        claimant_rec = next((u for u in users if facts.claimed_user_id and u["user_id"] == facts.claimed_user_id
                             and (order_rec is None or u["user_id"] != order_rec["user_id"])), None)
        audits = [a for a in d.ok_results("audit_fraud_risk") if order_rec and a["order_id"] == order_rec["order_id"]]
        errors = {c.result.get("error") for c in d.log if not c.synthetic and c.result.get("error")}
        return order_rec, owner_rec, claimant_rec, audits, errors

    order_rec, owner_rec, claimant_rec, audits, real_errors = collect()
    # Incident 15: the researcher's tools are deterministic lookups. If the model finished normally but skipped
    # one on an established order, the crew runs it (tool-log step 0) rather than halting a decidable case.
    if res.error is None and order_rec is not None:
        claimed = facts.claimed_user_id
        id_mismatch = bool(claimed) and claimed != order_rec["user_id"]
        if owner_rec is None and not id_mismatch:   # a mismatch case is about the claimant; the owner's profile is not needed
            d.call("get_user_profile", {"user_id": order_rec["user_id"]}, FALLBACK_STEP)
            notes.append("[researcher] owner profile was not read by the model; the crew read it (tool-log step 0)")
        if id_mismatch and "USER_ORDER_MISMATCH" not in real_errors:
            # Incident 17: the ticket names a customer who is not the owner. The kit's own verdict on that claim
            # is the evidence that matters; if the model audited without the id, the crew asks the engine itself.
            d.call("audit_fraud_risk", {"order_id": order_rec["order_id"], "user_id": claimed}, FALLBACK_STEP)
            notes.append(f"[researcher] the ticket claims {claimed} but the order belongs to {order_rec['user_id']}; "
                         "the crew asked the engine for its verdict on that claim (tool-log step 0)")
        elif not audits and "USER_ORDER_MISMATCH" not in real_errors:
            args = {"order_id": order_rec["order_id"]}
            if claimed:
                args["user_id"] = claimed
            d.call("audit_fraud_risk", args, FALLBACK_STEP)
            notes.append("[researcher] audit_fraud_risk was not called by the model; the crew ran it (tool-log step 0)")
        order_rec, owner_rec, claimant_rec, audits, real_errors = collect()

    identity, code = (out.identity_check if out else "unverified"), None
    if "USER_ORDER_MISMATCH" in real_errors:
        identity, code = "mismatch", "USER_ORDER_MISMATCH"
    elif order_rec and owner_rec and names_conflict(facts.claimed_name, owner_rec["name"]):
        if identity != "mismatch":
            notes.append(f"[researcher] name conflict caught by code: ticket says {facts.claimed_name!r}, owner is {owner_rec['name']!r}")
        identity, code = "mismatch", "IDENTITY_MISMATCH"
    elif identity == "mismatch":
        code = "IDENTITY_MISMATCH"

    if identity == "mismatch":
        status = "identity_mismatch"
    elif order_rec is None:
        status, code = "unresolvable", ("ORDER_NOT_FOUND" if facts.order_id else "NO_ORDER_ID")
    elif res.error:
        status, code = "incomplete", res.error
    elif owner_rec and audits:
        status = "complete"
    else:
        status, code = "incomplete", "MISSING_FRAUD_AUDIT" if owner_rec else "MISSING_CUSTOMER_PROFILE"
    if status != "complete" and res.error and code != res.error:
        notes.append(f"[researcher] agent ended with {res.error}: {res.detail}")

    report = RiskReport(
        status=status, error_code=code, ticket_facts=facts, identity_check=identity,
        order=OrderSummary.from_record(order_rec) if order_rec else None,
        customer=UserSummary.from_record(owner_rec) if owner_rec else None,
        claimant=UserSummary.from_record(claimant_rec) if claimant_rec else None,
        fraud_audit=FraudAudit.model_validate(audits[-1]) if audits else None,
        findings=out.findings if out else [f"researcher agent failed: {res.error}: {res.detail}"],
        tools_called=d.executed_tools(),
    )
    return report, notes


def synthesize_decision(report: RiskReport, plan: HaltPlan, merited: Optional[float], requested: float,
                        halt_reason: str) -> Decision:
    """The Decision for a halt path — no model ran, and the record says so."""
    return Decision(
        order_id=report.order.order_id if report.order else None,
        user_id=(report.claimant or report.customer).user_id if (report.claimant or report.customer) else None,
        decision=plan.decision, refund_status="NONE", refund_attempted=False,
        demanded_amount=report.ticket_facts.demanded_amount, merited_amount=merited, requested_amount=requested,
        rationale=[f"[crew] {report.status}: {halt_reason}", plan.reply_intent],
        synthesized_by_code=True, halt_reason=halt_reason,
    )


def build_decision(res: AgentRunResult, d: DecisionDispatcher, report: RiskReport, merited: float,
                   requested: float, prior_cases: list[dict[str, Any]]) -> tuple[Decision, list[str]]:
    """Decision from the tool evidence; the model's claim is recorded, then checked against it."""
    notes: list[str] = []
    out = res.output
    order_id = report.order.order_id
    checks = [r for r in d.ok_results("check_return_policy") if r.get("order_id") == order_id]
    if not checks and res.error is None:
        # Incident 18: the model decided from prior_cases alone and never asked the rulebook. The verdict is a
        # deterministic lookup the record must carry, so the crew runs it (tool-log step 0); the omission is noted.
        facts = report.ticket_facts
        reason = facts.return_reason or reason_from_condition(report.order.items[0].condition) or "changed_mind"
        d.call("check_return_policy", {"order_id": order_id, "reason": reason}, FALLBACK_STEP)
        notes.append("[decision] check_return_policy was not called by the model; the crew ran it (tool-log step 0)")
        checks = [r for r in d.ok_results("check_return_policy") if r.get("order_id") == order_id]
    policy = PolicyCheck.model_validate(checks[-1]) if checks else None
    refunds = [r for r in d.ok_results("process_refund") if r.get("order_id") == order_id]
    approved = [r for r in refunds if r.get("status") == "APPROVED"]
    refund = RefundOutcome.model_validate(approved[0] if approved else refunds[-1]) if refunds else None
    asked = report.ticket_facts.refund_requested
    blocks = compute_blocks(report, policy, prior_cases) if asked else []   # nothing was blocked: nothing was asked
    status, code = derive_outcome(policy, refund, blocks, asked)

    halt: Optional[str] = None
    if out is None:
        halt = res.error
        notes.append(f"[decision] agent failed ({res.error}: {res.detail}); outcome derived from tool evidence")
    elif status == "NONE" and code != "NO_REFUND_REQUESTED":
        halt = "DECISION_INCOMPLETE"
        notes.append("[decision] agent finished without establishing an outcome; escalating with what we have")
    claimed = out.decision if out else None
    if claimed is not None and claimed != code:
        notes.append(f"[decision] model claimed {claimed}; evidence says {code}; evidence wins")

    blob = json.dumps([c.result for c in d.log if not c.synthetic], ensure_ascii=False)
    cited = list(dict.fromkeys(out.cited_policies)) if out else []
    grounded = [p for p in cited if p in blob]
    if len(grounded) != len(cited):
        notes.append(f"[decision] dropped ungrounded citations: {sorted(set(cited) - set(grounded))}")

    decision = Decision(
        order_id=order_id, user_id=report.customer.user_id, decision=code, claimed_decision=claimed,
        refund_status=status, refund_attempted=refund is not None, blocked_by=blocks,
        demanded_amount=report.ticket_facts.demanded_amount, merited_amount=merited, requested_amount=requested,
        policy=policy, refund=refund, cited_policies=grounded,
        rationale=out.rationale if out else [f"[crew] decision agent failed: {res.error}: {res.detail}"],
        tools_called=d.executed_tools(), synthesized_by_code=out is None, halt_reason=halt,
    )
    return decision, notes


def alert_payload(report: RiskReport, decision: Decision, route: dict[str, Any]) -> dict[str, Any]:
    """A complete, state-true payload for a channel template — used when code must send the alert."""
    facts = expected_facts(report, decision)
    facts["triggered_rules"] = ", ".join(sorted(facts.get("triggered_rules", []))) or "none"
    audit = report.fraud_audit
    evidence = "; ".join(r.why for r in audit.triggered_rules) if audit and audit.triggered_rules else (report.error_code or "n/a")
    reasons = decision.blocked_by or ([decision.halt_reason] if decision.halt_reason else []) \
        or (decision.refund.reasons if decision.refund else [])
    policies = list(dict.fromkeys((decision.policy.applicable_policies if decision.policy else [])
                                  + (decision.refund.applicable_policies if decision.refund else [])))
    payload = {**facts, "evidence": evidence, "escalation_reason": "; ".join(reasons) or "n/a",
               "applicable_policies": ", ".join(policies) or "n/a"}
    for key in template_fields(route["template"]):
        payload.setdefault(key, "n/a")
    return payload


def build_comms(res: AgentRunResult, d: CommsDispatcher, report: RiskReport, decision: Decision) -> tuple[CommsResult, list[str]]:
    """CommsResult from the comms agent's output plus its tool log — with the crew's two fallbacks."""
    notes: list[str] = []
    fallback = False
    if d.established and d.route is None:
        notes.append("[comms] the model never consulted the router; the crew consulted it (tool-log step 0)")
        d.call("get_escalation_route", {}, FALLBACK_STEP)   # prepare() fills every argument from state
        fallback = True
    if d.route and d.route.get("escalation_required") and d.alert is None:
        payload = alert_payload(report, decision, d.route)
        sent = d.call("send_slack_alert", {"channel_id": d.route["channel_id"], "severity": d.route["severity"],
                                           "payload": payload}, FALLBACK_STEP)
        notes.append("[comms] required alert was not sent by the model; the crew sent it (tool-log step 0)"
                     if sent.get("delivered") else f"[comms] crew fallback alert failed: {sent}")
        fallback = True
    out = res.output
    if out is not None:
        reply, language = out.customer_reply, out.reply_language
    else:
        language = report.ticket_facts.language
        reply = fallback_reply(report.order.order_id if report.order else report.ticket_facts.order_id, language)
        notes.append(f"[comms] agent failed ({res.error}: {res.detail}); generic reply used")
        fallback = True
    comms = CommsResult(
        route=RouteResult.model_validate(d.route) if d.route else None,
        alert=AlertReceipt.model_validate({**d.alert, "payload": d.sent_payload or {}}) if d.alert else None,
        customer_reply=reply, reply_language=language, tools_called=d.executed_tools(), fallback_used=fallback,
    )
    return comms, notes


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #

class Crew:
    """Builds and runs the graph. `specs` = the three AgentSpecs (prompts + bundles + output models);
    `client` = the Anthropic SDK client or a FakeClient; `ledger` = long-term memory."""

    def __init__(self, specs: dict[AgentName, AgentSpec], client: Any, ledger: Optional[Ledger] = None,
                 model: str = MODEL, recursion_limit: int = RECURSION_LIMIT, verbose: bool = False):
        self.specs, self.client, self.model, self.verbose = specs, client, model, verbose
        self.ledger = ledger or Ledger()
        self.recursion_limit = recursion_limit
        self.app = self._build()

    # --- nodes ------------------------------------------------------------------------------
    def researcher(self, state: CrewState) -> dict[str, Any]:
        d = ResearcherDispatcher()
        res = run_agent(self.specs["researcher"], researcher_brief(state["ticket"]), d, self.client, self.model, self.verbose)
        report, notes = build_risk_report(res, d)
        return {"risk_report": report, "tool_log": d.log, "agent_runs": [res.run], "notes": notes + d.notes}

    def triage(self, state: CrewState) -> dict[str, Any]:
        report = state["risk_report"]
        facts, order = report.ticket_facts, report.order
        who = report.claimant or report.customer
        prior = self.ledger.recall(order.order_id if order else facts.order_id, who.user_id if who else None)
        merited = merited_amount(order.total_amount, facts.demanded_amount) if order else None
        requested = routing_amount(facts.refund_requested, merited)
        updates: dict[str, Any] = {"prior_cases": prior, "merited_amount": merited, "requested_amount": requested,
                                   "notes": [f"[triage] merited={merited} requested={requested} prior_cases={len(prior)}"]}
        if report.status != "complete":
            plan = halt_plan(report.status, report.error_code)
            updates["decision"] = synthesize_decision(report, plan, merited, requested, report.error_code)
            updates["halt_reason"] = report.error_code
            updates["notes"].append(f"[triage] halt: {report.status} ({report.error_code}) -> decision agent skipped")
        return updates

    @staticmethod
    def after_triage(state: CrewState) -> Literal["decision", "comms"]:
        return "comms" if state.get("halt_reason") else "decision"

    def decision(self, state: CrewState) -> dict[str, Any]:
        report, merited, prior = state["risk_report"], state["merited_amount"], state.get("prior_cases", [])
        d = DecisionDispatcher(report, merited, prior)
        res = run_agent(self.specs["decision"], decision_brief(state), d, self.client, self.model, self.verbose)
        decision, notes = build_decision(res, d, report, merited, state["requested_amount"], prior)
        updates: dict[str, Any] = {"decision": decision, "tool_log": d.log, "agent_runs": [res.run], "notes": notes + d.notes}
        if decision.halt_reason:
            updates["halt_reason"] = decision.halt_reason
        return updates

    def comms(self, state: CrewState) -> dict[str, Any]:
        report, decision, halt = state["risk_report"], state["decision"], state.get("halt_reason")
        plan = halt_plan(report.status if report.status != "complete" else "incomplete", halt) if halt else None
        d = CommsDispatcher(report, decision, plan)
        known = reply_facts(report, decision, state.get("prior_cases", []))
        verified = lambda text: find_unverified_claims(text, **known)   # noqa: E731 - this run's evidence, closed over
        res = run_agent(self.specs["comms"], comms_brief(state, plan), d, self.client, self.model, self.verbose,
                        reply_checks=[verified])
        comms, notes = build_comms(res, d, report, decision)
        return {"comms": comms, "tool_log": d.log, "agent_runs": [res.run], "notes": notes + d.notes}

    # --- wiring -----------------------------------------------------------------------------
    def _build(self):
        g = StateGraph(CrewState)
        g.add_node("researcher", self.researcher)
        g.add_node("triage", self.triage)
        g.add_node("decision", self.decision)
        g.add_node("comms", self.comms)
        g.add_edge(START, "researcher")
        g.add_edge("researcher", "triage")
        g.add_conditional_edges("triage", self.after_triage, {"decision": "decision", "comms": "comms"})
        g.add_edge("decision", "comms")
        g.add_edge("comms", END)
        return g.compile()

    def mermaid(self) -> str:
        """The README diagram, generated from the compiled graph — it cannot drift from the code."""
        return self.app.get_graph().draw_mermaid()

    # --- running ---------------------------------------------------------------------------
    def run(self, ticket: str, run_id: Optional[str] = None, remember: bool = True) -> CrewResult:
        run_id = run_id or uuid.uuid4().hex[:8]
        initial: CrewState = {"ticket": ticket, "run_id": run_id, "prior_cases": [], "risk_report": None,
                              "merited_amount": None, "requested_amount": 0.0, "decision": None, "comms": None,
                              "halt_reason": None, "tool_log": [], "agent_runs": [], "notes": []}
        try:
            final = self.app.invoke(initial, config={"recursion_limit": self.recursion_limit})
        except GraphRecursionError as exc:
            final = self._emergency(initial, f"RECURSION_LIMIT: {exc}")
        result = CrewResult(ticket=ticket, run_id=run_id, risk_report=final["risk_report"], decision=final["decision"],
                            comms=final["comms"], tool_log=final["tool_log"], agent_runs=final["agent_runs"],
                            notes=final["notes"], halt_reason=final.get("halt_reason"))
        if remember:
            self.ledger.remember(result)
        return result

    def _emergency(self, state: CrewState, reason: str) -> dict[str, Any]:
        """The graph was cut off by the recursion tripwire: escalate with what exists, by code."""
        report = RiskReport(status="incomplete", error_code="RECURSION_LIMIT", ticket_facts=TicketFacts(),
                            identity_check="unverified", findings=[reason])
        plan = halt_plan("incomplete", "RECURSION_LIMIT")
        decision = synthesize_decision(report, plan, None, 0.0, "RECURSION_LIMIT")
        d = CommsDispatcher(report, decision, plan)
        failed = AgentRunResult(output=None, error="RECURSION_LIMIT", detail=reason,
                                run=AgentRun(agent="comms", steps=0, format_retries=0, error="RECURSION_LIMIT"))
        comms, notes = build_comms(failed, d, report, decision)
        return {**state, "risk_report": report, "decision": decision, "comms": comms, "halt_reason": "RECURSION_LIMIT",
                "tool_log": list(state["tool_log"]) + d.log, "agent_runs": list(state["agent_runs"]) + [failed.run],
                "notes": list(state["notes"]) + [f"[crew] {reason}"] + notes + d.notes}
