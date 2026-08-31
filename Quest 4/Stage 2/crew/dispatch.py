"""Dispatchers: the only path from a model's tool call to a real kit function.

One dispatcher per agent per run. Each is built from that agent's kit bundle and
nothing else, so a call outside the lane cannot reach a function even if the
model names it (OUT_OF_LANE). On top of the lane, every dispatcher applies the
Part A guards (REPEATED_CALL, UNKNOWN_TOOL, TOOL_EXECUTION_ERROR) and its own
role-specific locks. Every refusal is a structured `{"error": CODE, "message":
...}` result — data the model reads and adapts to, never an exception — and it is
logged as a `ToolCall` with `synthetic=True` so the audit trail shows what the
model tried and what stopped it.

The locks are the enforceable half of the crew's conduct rules (decision D8:
prompts request, code enforces):

    researcher   ID_PROBING_BLOCKED
    decision     WRONG_ORDER, BLOCKED_BY_RISK_REPORT, SEQUENCING_VIOLATION,
                 BLOCKED_BY_POLICY_VERDICT, BLOCKED_BY_POLICY_ESCALATION,
                 DUPLICATE_CLAIM, AMOUNT_NOT_MERITED
    comms        ROUTING_NOT_APPLICABLE, ARGUMENT_MISMATCH, ALERT_NOT_AUTHORIZED,
                 DUPLICATE_ALERT, ROUTE_MISMATCH, PAYLOAD_MISMATCH
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, Optional

import multi_agent_tools as mat  # the crew package bootstrap put the kit on sys.path

from crew.config import REPEAT_CALL_LIMIT
from crew.policy import POLICY_ID, RULE_ID, HaltPlan, template_fields
from crew.schemas import AgentName, Decision, RiskReport, ToolCall


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


def forced_route(channel_id: str, reason: str) -> dict[str, Any]:
    """A route the crew names itself (decision D3 / fallback paths), in the router's own shape."""
    channel = next(c for c in mat.get_escalation_channels()["channels"] if c["channel_id"] == channel_id)
    return {
        "escalation_required": True, "channel_id": channel["channel_id"], "channel": channel["name"],
        "owner_team": channel["owner_team"], "severity": channel["severity"], "priority": channel["priority"],
        "response_sla_minutes": channel["response_sla_minutes"], "template": channel["template"],
        "matched_condition": reason, "override_reason": reason,
    }


def expected_facts(report: RiskReport, decision: Decision) -> dict[str, Any]:
    """The facts an alert payload may state — and must state correctly."""
    audit, order, policy, refund = report.fraud_audit, report.order, decision.policy, decision.refund
    who = report.claimant or report.customer            # decision D3: the claimant is the party of interest
    facts: dict[str, Any] = {"requested_amount": float(decision.requested_amount)}
    if order:
        facts.update(order_id=order.order_id, order_status=order.status, order_date=order.order_date)
    if who:
        facts["user_id"] = who.user_id
    if audit and report.status != "identity_mismatch":
        facts.update(risk_score=audit.risk_score, risk_band=audit.risk_band,
                     triggered_rules={r.rule_id for r in audit.triggered_rules})
    else:   # incidents 5 and 12: no engine report - or a report about the order's OWNER while the alert is about
            # the claimant - means there is no score and no band for this case; a number here is a fabrication
        facts.update(risk_score="n/a", risk_band="n/a")
    if policy:
        facts["verdict"] = policy.verdict
    cap = (policy.auto_refund_cap_usd if policy else None) or (refund.auto_refund_cap_usd if refund else None)
    if cap is not None:
        facts["auto_refund_cap"] = float(cap)
    return facts


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class Dispatcher:
    """Lane + guards. Subclasses add `prepare` (fill omitted args), `lock` (refuse) and `after` (post-process)."""

    agent: AgentName

    def __init__(self, bundle: list[dict[str, Any]], registry: Optional[dict[str, Callable[..., Any]]] = None):
        registry = registry or mat.TOOL_REGISTRY
        self.allowed: list[str] = [t["name"] for t in bundle]
        self.registry: dict[str, Callable[..., Any]] = {name: registry[name] for name in self.allowed}
        self.log: list[ToolCall] = []
        self.notes: list[str] = []
        self._counts: Counter = Counter()

    # --- the single entry point -------------------------------------------------------------
    def call(self, name: str, args: dict[str, Any], step: int) -> dict[str, Any]:
        args = dict(args)
        if name not in self.registry:
            owner = mat.TOOL_OWNERSHIP.get(name)
            if owner:
                out = _err("OUT_OF_LANE", f"'{name}' belongs to the {owner} agent, not to you. "
                                          f"Your tools are: {', '.join(self.allowed)}.")
            else:
                out = _err("UNKNOWN_TOOL", f"No tool named '{name}'. Your tools are: {', '.join(self.allowed)}.")
            return self._record(step, name, args, out, synthetic=True)

        args = self.prepare(name, args)
        blocked = self.lock(name, args)          # role locks first: their messages are the actionable ones
        if blocked is not None:
            return self._record(step, name, args, blocked, synthetic=True)

        key = (name, json.dumps(args, sort_keys=True, default=str))
        self._counts[key] += 1
        if self._counts[key] > REPEAT_CALL_LIMIT:
            return self._record(step, name, args, _err(
                "REPEATED_CALL", f"This exact call was already made {REPEAT_CALL_LIMIT} times. "
                                 "Reuse the earlier result instead of calling again."), synthetic=True)

        try:
            out = self.registry[name](**args)
        except Exception as exc:  # a malformed argument must not crash the crew
            out = _err("TOOL_EXECUTION_ERROR", f"{type(exc).__name__}: {exc}")
        out = self.after(name, args, out)
        return self._record(step, name, args, out, synthetic=False)

    # --- hooks ----------------------------------------------------------------------------
    def prepare(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return args

    def lock(self, name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        return None

    def after(self, name: str, args: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
        return out

    # --- bookkeeping --------------------------------------------------------------------------
    def _record(self, step: int, name: str, args: dict[str, Any], out: dict[str, Any], synthetic: bool) -> dict[str, Any]:
        self.log.append(ToolCall(agent=self.agent, step=step, tool=name, args=args, result=out, synthetic=synthetic))
        return out

    def executed_tools(self) -> list[str]:
        """Distinct tool names the model reached for through its lane, first-call order."""
        seen: list[str] = []
        for call in self.log:
            if call.tool in self.registry and call.tool not in seen:
                seen.append(call.tool)
        return seen

    def real_results(self, name: str) -> list[dict[str, Any]]:
        """Results the kit actually produced for `name` (synthetic refusals excluded)."""
        return [c.result for c in self.log if c.tool == name and not c.synthetic]

    def ok_results(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.real_results(name) if "error" not in r]


# --------------------------------------------------------------------------- #
# Agent 1
# --------------------------------------------------------------------------- #

class ResearcherDispatcher(Dispatcher):
    agent = "researcher"

    def __init__(self, registry=None):
        super().__init__(mat.RESEARCHER_TOOLS, registry)

    def _errors(self, code: str) -> list[ToolCall]:
        return [c for c in self.log if not c.synthetic and c.result.get("error") == code]

    def lock(self, name, args):
        mismatch = self._errors("USER_ORDER_MISMATCH")
        if mismatch and name != "get_user_profile":
            first = mismatch[0]
            return _err("ID_PROBING_BLOCKED",
                        f"audit_fraud_risk already reported USER_ORDER_MISMATCH for order "
                        f"{first.args.get('order_id')} and customer {first.args.get('user_id')}. Do not try other "
                        "ids and do not re-audit. You may still read a customer profile. Report "
                        "identity_check='mismatch' and finish.")
        not_found = {c.args.get("order_id") for c in self._errors("ORDER_NOT_FOUND")}
        if not_found:
            if name == "audit_fraud_risk":
                return _err("ID_PROBING_BLOCKED", f"Order {', '.join(sorted(map(str, not_found)))} does not exist; "
                                                  "there is nothing to audit. Report the case as unresolvable and finish.")
            if name == "get_order_details" and args.get("order_id") not in not_found:
                return _err("ID_PROBING_BLOCKED", f"Order {', '.join(sorted(map(str, not_found)))} was not found. "
                                                  "Do not guess other order numbers; ask the customer to confirm the "
                                                  "number. Report the case as unresolvable and finish.")
        return None


# --------------------------------------------------------------------------- #
# Agent 2
# --------------------------------------------------------------------------- #

class DecisionDispatcher(Dispatcher):
    agent = "decision"

    def __init__(self, report: RiskReport, merited_amount: float, prior_cases: list[dict[str, Any]], registry=None):
        super().__init__(mat.DECISION_TOOLS, registry)
        if report.order is None:
            raise ValueError("the decision agent only runs on an established order")
        self.report = report
        self.order_id = report.order.order_id
        self.merited = merited_amount
        self.prior_cases = prior_cases

    def _policy(self) -> Optional[dict[str, Any]]:
        checks = [r for r in self.ok_results("check_return_policy") if r.get("order_id") == self.order_id]
        return checks[-1] if checks else None

    def lock(self, name, args):
        if args.get("order_id") != self.order_id:
            return _err("WRONG_ORDER", f"This case is about order {self.order_id} only.")
        if name != "process_refund":
            return None
        if not self.report.ticket_facts.refund_requested:
            return _err("NO_REFUND_REQUESTED", "The customer did not ask for money on this ticket. Do not pay; "
                                               "the decision is NO_REFUND_REQUESTED.")

        audit = self.report.fraud_audit
        if audit is not None and audit.blocks_automatic_refund:
            rules = ", ".join(r.rule_id for r in audit.triggered_rules)
            return _err("BLOCKED_BY_RISK_REPORT",
                        f"The risk report ({audit.risk_score}/100, band '{audit.risk_band}', rules {rules}) blocks "
                        "any automatic refund. Do not attempt a refund; the decision is ESCALATED_TO_HUMAN.")
        policy = self._policy()
        if policy is None:
            return _err("SEQUENCING_VIOLATION", "Call check_return_policy for this order before process_refund.")
        if not policy.get("eligible"):
            return _err("BLOCKED_BY_POLICY_VERDICT",
                        f"check_return_policy returned {policy.get('verdict')}; this channel cannot pay. "
                        "Do not attempt a refund; the decision is REJECTED.")
        if policy.get("requires_escalation"):
            return _err("BLOCKED_BY_POLICY_ESCALATION",
                        "check_return_policy requires human review: " + "; ".join(policy.get("escalation_reasons", []))
                        + ". Do not attempt a refund; the decision is ESCALATED_TO_HUMAN.")
        for case in self.prior_cases:
            if case.get("order_id") == self.order_id and case.get("refund_status") == "APPROVED":
                return _err("DUPLICATE_CLAIM",
                            f"Order {self.order_id} was already refunded ({case.get('refund_id')} on "
                            f"{case.get('timestamp', 'an earlier date')}). Do not pay twice; the decision is REJECTED "
                            "and the customer should be told about the existing refund.")
        amount = _num(args.get("amount"))
        if amount is None or abs(amount - self.merited) > 0.005:
            return _err("AMOUNT_NOT_MERITED",
                        f"The merited amount for this claim is {self.merited:.2f} USD (order total "
                        f"{self.report.order.total_amount:.2f}; customer asked for "
                        f"{self.report.ticket_facts.demanded_amount}). Request exactly {self.merited:.2f} and let "
                        "process_refund arbitrate the cap. Never reduce a claim to fit under your authority.",
                        merited_amount=self.merited)
        return None


# --------------------------------------------------------------------------- #
# Agent 3
# --------------------------------------------------------------------------- #

class CommsDispatcher(Dispatcher):
    agent = "comms"

    def __init__(self, report: RiskReport, decision: Decision, plan: Optional[HaltPlan] = None, registry=None):
        super().__init__(mat.COMMS_TOOLS, registry)
        self.report, self.decision, self.plan = report, decision, plan
        self.established = plan is None or plan.consult_router
        self.route: Optional[dict[str, Any]] = None       # the effective route (after any crew override)
        self.alert: Optional[dict[str, Any]] = None       # the delivered alert's result
        self.sent_payload: Optional[dict[str, Any]] = None

    # what the router must be told, computed from state — the model may not disagree with it
    def expected_route_args(self) -> dict[str, Any]:
        audit, order, policy = self.report.fraud_audit, self.report.order, self.decision.policy
        who = self.report.claimant or self.report.customer   # decision D3: the claimant's flags on a mismatch
        return {
            "risk_band": audit.risk_band if audit else "low",
            "requested_amount": float(self.decision.requested_amount),
            "prior_fraud_flags": int(who.prior_fraud_flags) if who else 0,
            "order_status": order.status if order else "delivered",
            "verdict": policy.verdict if policy else "ELIGIBLE",
        }

    def expected_facts(self) -> dict[str, Any]:
        return expected_facts(self.report, self.decision)

    def prepare(self, name, args):
        if name == "get_escalation_route":
            for key, value in self.expected_route_args().items():
                if key not in args:
                    args[key] = value
                    self.notes.append(f"get_escalation_route: filled {key}={value!r} from state")
        return args

    def lock(self, name, args):
        if name == "get_escalation_route":
            if not self.established:
                return _err("ROUTING_NOT_APPLICABLE", "No established order or customer: there is nothing to route. "
                                                      "Reply to the customer; do not route and do not alert.")
            wrong = {}
            for key, expected in self.expected_route_args().items():
                given = args.get(key)
                same = (abs(_num(given) - expected) <= 0.005) if isinstance(expected, float) and _num(given) is not None \
                    else (str(given) == str(expected))
                if not same:
                    wrong[key] = {"given": given, "expected": expected}
            if wrong:
                return _err("ARGUMENT_MISMATCH", "Route with the facts from the reports, not from memory. "
                                                 "Corrections: " + json.dumps(wrong, default=str), expected=self.expected_route_args())
            return None

        if name == "send_slack_alert":
            if not self.established or self.route is None:
                return _err("ALERT_NOT_AUTHORIZED", "Call get_escalation_route first; alert only if it returns "
                                                    "escalation_required=true.")
            if not self.route.get("escalation_required"):
                return _err("ALERT_NOT_AUTHORIZED", "The router returned escalation_required=false. Do not alert; "
                                                    "send the customer reply only.")
            if self.alert is not None:
                return _err("DUPLICATE_ALERT", "An alert was already delivered for this case. Do not send another.")
            if args.get("channel_id") != self.route["channel_id"] or args.get("severity") != self.route["severity"]:
                return _err("ROUTE_MISMATCH", f"Send to channel_id={self.route['channel_id']!r} with "
                                              f"severity={self.route['severity']!r}, exactly as the router said.")
            payload = args.get("payload")
            if not isinstance(payload, dict):
                return _err("PAYLOAD_MISMATCH", "payload must be a JSON object of structured facts.")
            problems = self._payload_problems(payload)
            if problems:
                return _err("PAYLOAD_MISMATCH", "The alert payload must state the case facts exactly. " +
                            json.dumps(problems, default=str), expected=json.loads(json.dumps(self.expected_facts(), default=list)))
        return None

    def _payload_problems(self, payload: dict[str, Any]) -> dict[str, Any]:
        problems: dict[str, Any] = {}
        missing = sorted(template_fields(self.route["template"]) - set(payload))
        if missing:
            problems["missing"] = missing
        facts = self.expected_facts()
        for key, expected in facts.items():
            if key not in payload:
                continue
            given = payload[key]
            if key == "triggered_rules":
                ok = set(RULE_ID.findall(json.dumps(given, ensure_ascii=False))) == expected
            elif isinstance(expected, float):
                ok = _num(given) is not None and abs(_num(given) - expected) <= 0.005
            elif isinstance(expected, int):
                ok = _num(given) is not None and _num(given) == expected
            else:
                ok = str(given).strip().lower() == str(expected).lower()
            if not ok:
                problems[key] = {"given": given, "expected": expected if not isinstance(expected, set) else sorted(expected)}
        if "applicable_policies" in payload:
            known = set((self.decision.policy.applicable_policies if self.decision.policy else [])
                        + (self.decision.refund.applicable_policies if self.decision.refund else []))
            cited = set(POLICY_ID.findall(json.dumps(payload["applicable_policies"])))
            if cited - known:
                problems["applicable_policies"] = {"ungrounded": sorted(cited - known), "known": sorted(known)}
        return problems

    def after(self, name, args, out):
        if name == "get_escalation_route" and "error" not in out:
            if not out["escalation_required"] and self.plan is not None and self.plan.forced_channel:
                out = self._override(out, self.plan.forced_channel)
            self.route = out
        if name == "send_slack_alert" and out.get("delivered"):
            self.alert, self.sent_payload = out, args.get("payload")
        return out

    def _override(self, out: dict[str, Any], channel_id: str) -> dict[str, Any]:
        """ The router has no input for an identity mismatch or a crew failure,
        so crew policy names the channel and says so in the result the model sees."""
        reason = f"crew policy: {self.decision.halt_reason or 'halt'} -> {channel_id} (router matched nothing)"
        self.notes.append(reason)
        return forced_route(channel_id, reason)
