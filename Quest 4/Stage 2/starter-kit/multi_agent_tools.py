"""Extended tool box for Quest #04, Part B — the Multi-Agent Crew.

Part A gave your single agent four tools. Part B adds three, and — just as
importantly — splits all seven into **per-role bundles**. A crew works because
each specialist can only reach for the tools its job requires:

======================  ==========================================================
``RESEARCHER_TOOLS``    ``get_order_details``, ``get_user_profile``,
                        ``audit_fraud_risk``
``DECISION_TOOLS``      ``check_return_policy``, ``process_refund``
``COMMS_TOOLS``         ``get_escalation_route``, ``send_slack_alert``
======================  ==========================================================

The Comms agent has no access to ``process_refund``. The Researcher cannot
approve money. That separation is not decoration — it is the cheapest guardrail
in the system, and the brief asks you to implement it.

Everything is still deterministic and offline. ``send_slack_alert`` appends to
``outbox/alerts.jsonl`` by default; set ``SLACK_WEBHOOK_URL`` if you want to
post to a real Slack incoming webhook for the demo video.

Usage
-----
>>> import multi_agent_tools as mat
>>> report = mat.audit_fraud_risk("ORD-1005", "USR-105")
>>> report["risk_band"]
'high'
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from mock_services import (
    TOOL_SCHEMAS as PART_A_SCHEMAS,
    check_return_policy,
    get_order_details,
    get_policies,
    get_user_profile,
    process_refund,
    reference_date,
)

__all__ = [
    "audit_fraud_risk",
    "get_escalation_route",
    "send_slack_alert",
    "read_outbox",
    "RESEARCHER_TOOLS",
    "DECISION_TOOLS",
    "COMMS_TOOLS",
    "TOOL_SCHEMAS",
    "TOOL_REGISTRY",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTBOX_PATH = BASE_DIR / "outbox" / "alerts.jsonl"


@lru_cache(maxsize=None)
def _load(filename: str) -> Dict[str, Any]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture {path}.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _error(code: str, message: str) -> Dict[str, Any]:
    """Same structured-error shape that mock_services returns."""
    return {"error": code, "message": message}


def _parse(value: Optional[str]) -> Optional[date]:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def _find_order(order_id: str) -> Optional[Dict[str, Any]]:
    order = get_order_details(order_id)
    return None if "error" in order else order


def _find_user(user_id: str) -> Optional[Dict[str, Any]]:
    user = get_user_profile(user_id)
    return None if "error" in user else user


def get_fraud_rules() -> Dict[str, Any]:
    """Return the raw fraud rulebook, so an agent can quote a ``rule_id``."""
    return _load("fraud_rules.json")


def get_escalation_channels() -> Dict[str, Any]:
    """Return the raw escalation routing table."""
    return _load("escalation_channels.json")


def _band_for(score: int) -> Dict[str, Any]:
    for band in get_fraud_rules()["risk_bands"]:
        if band["min_score"] <= score <= band["max_score"]:
            return band
    return get_fraud_rules()["risk_bands"][-1]


# --------------------------------------------------------------------------- #
# Tool 5 — the Researcher's fraud engine
# --------------------------------------------------------------------------- #

def audit_fraud_risk(order_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Run the GlobalCart fraud rulebook over one order and its customer.

    When to use this tool:
        This is the Researcher / Fraud Auditor agent's job. Call it after you
        have fetched the order and the profile. It is a **deterministic rule
        engine**, not a judgement call — do not try to reason your way to a risk
        score yourself, and do not overrule the band you get back. Pass the
        whole report downstream to the Decision agent; the ``triggered_rules``
        and ``evidence`` fields are what makes the final decision auditable.

    Args:
        order_id: The order under investigation, e.g. ``"ORD-1005"``.
        user_id: The customer to score. Optional — if omitted, it is taken from
            the order. Pass it explicitly when you want to assert that the
            ticket's claimed customer really owns the order.

    Returns:
        On success, a dict with keys:
            ``order_id``, ``user_id``, ``risk_score`` (0-100),
            ``risk_band`` (``low`` / ``medium`` / ``high``), ``action_hint``,
            ``triggered_rules`` (list of ``{rule_id, name, weight, why}``),
            ``evidence`` (the raw signals the rules fired on),
            ``blocks_automatic_refund`` (bool — true when the band is high),
            ``requires_security_channel`` (bool) and ``rulebook_version``.
        On failure, ``{"error": ..., "message": ...}`` with code
        ``ORDER_NOT_FOUND``, ``USER_NOT_FOUND`` or ``USER_ORDER_MISMATCH``.
    """
    if not isinstance(order_id, str):
        raise TypeError("order_id must be a string")

    order = _find_order(order_id.strip())
    if order is None:
        return _error("ORDER_NOT_FOUND", f"No order found with id '{order_id}'.")

    resolved_user_id = (user_id or order["user_id"]).strip()
    user = _find_user(resolved_user_id)
    if user is None:
        return _error("USER_NOT_FOUND", f"No user found with id '{resolved_user_id}'.")
    if resolved_user_id != order["user_id"]:
        return _error(
            "USER_ORDER_MISMATCH",
            f"Order {order['order_id']} belongs to {order['user_id']}, not to "
            f"{resolved_user_id}. Treat this as a red flag and escalate.",
        )

    rules = {r["rule_id"]: r for r in get_fraud_rules()["rules"]}
    today = reference_date()
    triggered: List[Dict[str, Any]] = []

    def fire(rule_id: str, why: str) -> None:
        rule = rules[rule_id]
        triggered.append(
            {"rule_id": rule_id, "name": rule["name"], "weight": rule["weight"], "why": why}
        )

    # --- gather signals ---------------------------------------------------- #
    history = user.get("refund_history", [])
    window_60 = [
        claim
        for claim in history
        if (parsed := _parse(claim.get("date"))) and (today - parsed).days <= 60
    ]
    refunded_60 = round(sum(float(c["amount"]) for c in window_60), 2)
    account_age = (today - _parse(user["account_created_at"])).days
    delivery = _parse(order.get("delivery_date"))
    changed = _parse(order.get("address_changed_at"))
    days_before_delivery = (delivery - changed).days if (delivery and changed) else None
    missing_items = [i["sku"] for i in order["items"] if i.get("condition") == "missing"]

    evidence = {
        "reference_date": str(today),
        "order_total_usd": order["total_amount"],
        "order_status": order["status"],
        "claims_in_last_60_days": len(window_60),
        "refunded_usd_in_last_60_days": refunded_60,
        "account_age_days": account_age,
        "prior_fraud_flags": user.get("prior_fraud_flags", 0),
        "initial_fraud_score": user.get("initial_fraud_score", 0),
        "address_changed_at": order.get("address_changed_at"),
        "address_change_days_before_delivery": days_before_delivery,
        "missing_item_skus": missing_items,
    }

    # --- evaluate rules ---------------------------------------------------- #
    r1 = rules["FR-01"]["params"]
    if len(window_60) >= r1["min_claims"]:
        fire("FR-01", f"{len(window_60)} claims in the last {r1['window_days']} days")

    r2 = rules["FR-02"]["params"]
    if days_before_delivery is not None and 0 <= days_before_delivery <= r2["days_before_delivery"]:
        fire("FR-02", f"address changed {days_before_delivery} day(s) before delivery")

    r3 = rules["FR-03"]["params"]
    if account_age < r3["max_account_age_days"] and order["total_amount"] >= r3["min_order_total_usd"]:
        fire("FR-03", f"account is {account_age} days old and order is {order['total_amount']:.2f} USD")

    if user.get("prior_fraud_flags", 0) >= rules["FR-04"]["params"]["min_flags"]:
        fire("FR-04", f"{user['prior_fraud_flags']} prior fraud flag(s) on record")

    if user.get("initial_fraud_score", 0) >= rules["FR-05"]["params"]["min_score"]:
        fire("FR-05", f"inherited fraud score is {user['initial_fraud_score']}")

    if order["total_amount"] >= rules["FR-06"]["params"]["min_order_total_usd"]:
        fire("FR-06", f"order total is {order['total_amount']:.2f} USD")

    if missing_items:
        fire("FR-07", f"claim of missing item(s): {', '.join(missing_items)}")

    r8 = rules["FR-08"]["params"]
    if refunded_60 >= r8["min_total_usd"]:
        fire("FR-08", f"{refunded_60:.2f} USD refunded in the last {r8['window_days']} days")

    score = min(sum(t["weight"] for t in triggered), get_fraud_rules()["scoring"]["max_score"])
    band = _band_for(score)

    return {
        "order_id": order["order_id"],
        "user_id": resolved_user_id,
        "risk_score": score,
        "risk_band": band["band"],
        "action_hint": band["action_hint"],
        "triggered_rules": triggered,
        "evidence": evidence,
        "blocks_automatic_refund": band["band"] == "high",
        "requires_security_channel": band["band"] == "high" or user.get("prior_fraud_flags", 0) > 0,
        "rulebook_version": get_fraud_rules()["version"],
    }


# --------------------------------------------------------------------------- #
# Tool 6 — the Comms agent's routing helper
# --------------------------------------------------------------------------- #

def get_escalation_route(
    risk_band: str = "low",
    requested_amount: float = 0.0,
    prior_fraud_flags: int = 0,
    order_status: str = "delivered",
    verdict: str = "ELIGIBLE",
) -> Dict[str, Any]:
    """Pick the escalation channel a case should be handed to.

    When to use this tool:
        This is the Communications & Escalation agent's job. Call it once the
        Decision agent has produced a verdict, to find out *where* the handoff
        goes before you write the alert. Channels are evaluated in priority
        order and the first match wins, so you get exactly one destination.

    Args:
        risk_band: ``low``, ``medium`` or ``high`` — from
            :func:`audit_fraud_risk`.
        requested_amount: The refund amount in USD that was asked for.
        prior_fraud_flags: From the customer profile.
        order_status: From the order.
        verdict: The policy verdict, e.g. ``ELIGIBLE``,
            ``OUTSIDE_RETURN_WINDOW``, ``NON_RETURNABLE_CATEGORY``.

    Returns:
        A dict with ``escalation_required`` (bool), ``channel_id``, ``channel``
        (the Slack channel name), ``severity``, ``priority``,
        ``response_sla_minutes``, ``template`` and ``matched_condition``. When
        nothing warrants escalation, ``escalation_required`` is ``False`` and
        ``channel_id`` is ``None`` — in that case do **not** send an alert.
    """
    channels = {c["channel_id"]: c for c in get_escalation_channels()["channels"]}
    cap = get_policies()["auto_refund_cap_usd"]

    def route(channel_id: str, matched: str) -> Dict[str, Any]:
        channel = channels[channel_id]
        return {
            "escalation_required": True,
            "channel_id": channel["channel_id"],
            "channel": channel["name"],
            "owner_team": channel["owner_team"],
            "severity": channel["severity"],
            "priority": channel["priority"],
            "response_sla_minutes": channel["response_sla_minutes"],
            "template": channel["template"],
            "matched_condition": matched,
        }

    if risk_band == "high" or prior_fraud_flags > 0:
        return route(
            "CH-FRAUD",
            f"risk_band == '{risk_band}'" if risk_band == "high" else "prior_fraud_flags > 0",
        )
    if requested_amount >= 250:
        return route("CH-FINANCE", "requested_amount >= 250")
    if requested_amount > cap:
        return route("CH-SUPPORT-T2", f"requested_amount > auto_refund_cap ({cap:.2f})")
    if verdict in {"NON_RETURNABLE_CATEGORY", "OUTSIDE_RETURN_WINDOW"}:
        return route("CH-SUPPORT-T2", f"verdict == '{verdict}'")
    if order_status in {"delayed", "processing"}:
        return route("CH-LOGISTICS", f"order_status == '{order_status}'")

    return {
        "escalation_required": False,
        "channel_id": None,
        "channel": None,
        "owner_team": None,
        "severity": None,
        "priority": None,
        "response_sla_minutes": None,
        "template": None,
        "matched_condition": "no escalation condition matched — resolve automatically",
    }


# --------------------------------------------------------------------------- #
# Tool 7 — the outbound alert
# --------------------------------------------------------------------------- #

def send_slack_alert(
    channel_id: str,
    severity: str,
    payload: Dict[str, Any],
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a structured alert to an external channel. **Side effect.**

    When to use this tool:
        Only the Communications & Escalation agent may call this, and only after
        :func:`get_escalation_route` returned ``escalation_required: true``.
        Sending an alert for a case that resolved cleanly is noise, and graders
        read the outbox.

        By default the alert is written to ``outbox/alerts.jsonl`` — one JSON
        object per line — so the whole quest works offline with no Slack
        workspace. If the ``SLACK_WEBHOOK_URL`` environment variable is set, the
        message is also POSTed to that incoming webhook, which is what you want
        for the demo video.

    Args:
        channel_id: A channel id from ``escalation_channels.json``, e.g.
            ``"CH-FRAUD"``. Take it from :func:`get_escalation_route`.
        severity: ``low``, ``medium``, ``high`` or ``critical``.
        payload: The structured facts behind the alert — order id, user id, risk
            score, triggered rule ids, requested amount. Keep it machine
            readable; this is the record a human will act on.
        message: Optional human-readable body. If omitted, one is rendered from
            the channel's template using ``payload``.

    Returns:
        A dict with ``delivered`` (bool), ``channel_id``, ``channel``,
        ``severity``, ``message_ts`` (a deterministic pseudo-timestamp),
        ``transport`` (``outbox`` or ``outbox+webhook``), ``outbox_path``,
        ``webhook_status`` (``None`` when no webhook is configured) and
        ``message``.
        Or ``{"error": ..., "message": ...}`` with code ``CHANNEL_NOT_FOUND``
        or ``INVALID_SEVERITY``.
    """
    channels = {c["channel_id"]: c for c in get_escalation_channels()["channels"]}
    if channel_id not in channels:
        return _error(
            "CHANNEL_NOT_FOUND",
            f"'{channel_id}' is not a known channel. Valid ids: {', '.join(sorted(channels))}.",
        )
    levels = get_escalation_channels()["severity_levels"]
    if severity not in levels:
        return _error("INVALID_SEVERITY", f"severity must be one of {', '.join(levels)}.")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    channel = channels[channel_id]
    if message is None:
        try:
            message = channel["template"].format(**{**_template_defaults(), **payload})
        except (KeyError, IndexError):
            message = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    # Deterministic pseudo-timestamp: reference date plus a stable digest of the
    # payload. Real Slack would give you a wall-clock ts; we keep it reproducible
    # so that a test asserting on message_ts still passes tomorrow. (Note we use
    # hashlib rather than the built-in hash(), which is salted per process.)
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = int(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
    message_ts = f"{reference_date().strftime('%Y%m%d')}.{digest:06d}"

    record = {
        "message_ts": message_ts,
        "channel_id": channel_id,
        "channel": channel["name"],
        "severity": severity,
        "payload": payload,
        "message": message,
    }

    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTBOX_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    webhook_status: Optional[int] = None
    transport = "outbox"
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        transport = "outbox+webhook"
        body = json.dumps({"text": f"[{severity.upper()}] {channel['name']}\n{message}"}).encode()
        request = urllib.request.Request(
            webhook, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                webhook_status = response.status
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            webhook_status = getattr(exc, "code", -1)

    return {
        "delivered": True,
        "channel_id": channel_id,
        "channel": channel["name"],
        "severity": severity,
        "message_ts": message_ts,
        "transport": transport,
        "outbox_path": str(OUTBOX_PATH.relative_to(BASE_DIR)),
        "webhook_status": webhook_status,
        "message": message,
    }


def _template_defaults() -> Dict[str, Any]:
    """Placeholders so a partially-filled payload still renders a template."""
    return {
        "order_id": "unknown",
        "user_id": "unknown",
        "risk_score": "n/a",
        "risk_band": "n/a",
        "triggered_rules": "none",
        "requested_amount": "n/a",
        "auto_refund_cap": get_policies()["auto_refund_cap_usd"],
        "evidence": "n/a",
        "applicable_policies": "n/a",
        "verdict": "n/a",
        "escalation_reason": "n/a",
        "order_status": "n/a",
        "order_date": "n/a",
    }


def read_outbox() -> List[Dict[str, Any]]:
    """Read back everything :func:`send_slack_alert` has written.

    Not a tool for your agents — a helper for you and for whoever grades the
    submission. Use it in your demo video to prove the alert actually left the
    system.
    """
    if not OUTBOX_PATH.exists():
        return []
    with OUTBOX_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# Per-role tool schemas — Separation of Concerns, enforced by construction
# --------------------------------------------------------------------------- #

_AUDIT_SCHEMA = {
    "name": "audit_fraud_risk",
    "description": (
        "Run the deterministic GlobalCart fraud rulebook over one order and its "
        "customer. Returns a risk score out of 100, a risk band (low/medium/high), "
        "the list of rules that fired with the evidence behind each, and whether "
        "the band blocks an automatic refund. Use this instead of reasoning about "
        "risk yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order under investigation, e.g. 'ORD-1005'."},
            "user_id": {
                "type": "string",
                "description": "Customer to score. Optional; defaults to the order's owner.",
            },
        },
        "required": ["order_id"],
    },
}

_ROUTE_SCHEMA = {
    "name": "get_escalation_route",
    "description": (
        "Decide which escalation channel a case should be handed to, given the "
        "risk band, the requested amount, prior fraud flags, the order status and "
        "the policy verdict. Returns exactly one destination, or "
        "escalation_required=false when the case can be resolved automatically. "
        "Call this before send_slack_alert."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_band": {"type": "string", "enum": ["low", "medium", "high"]},
            "requested_amount": {"type": "number", "description": "Refund amount asked for, in USD."},
            "prior_fraud_flags": {"type": "integer", "description": "From the customer profile."},
            "order_status": {"type": "string", "description": "From the order."},
            "verdict": {
                "type": "string",
                "description": "Policy verdict, e.g. ELIGIBLE or OUTSIDE_RETURN_WINDOW.",
            },
        },
        "required": ["risk_band"],
    },
}

_ALERT_SCHEMA = {
    "name": "send_slack_alert",
    "description": (
        "Send a structured alert to an external escalation channel. This has a "
        "side effect: the alert is appended to outbox/alerts.jsonl and, if a "
        "webhook is configured, POSTed to Slack. Call it only after "
        "get_escalation_route returned escalation_required=true."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Channel id from get_escalation_route, e.g. 'CH-FRAUD'.",
            },
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "payload": {
                "type": "object",
                "description": (
                    "Structured facts behind the alert: order_id, user_id, risk_score, "
                    "risk_band, triggered_rules, requested_amount."
                ),
            },
            "message": {
                "type": "string",
                "description": "Optional human-readable body. Rendered from the channel template if omitted.",
            },
        },
        "required": ["channel_id", "severity", "payload"],
    },
}

_BY_NAME = {s["name"]: s for s in PART_A_SCHEMAS}

#: Agent 1 — The Researcher & Fraud Auditor. Reads the world, scores the risk.
#: Deliberately has no ability to approve money or to message anyone.
RESEARCHER_TOOLS: List[Dict[str, Any]] = [
    _BY_NAME["get_order_details"],
    _BY_NAME["get_user_profile"],
    _AUDIT_SCHEMA,
]

#: Agent 2 — The Decision Maker / Operations Lead. Consults policy and decides.
#: Cannot investigate from scratch and cannot talk to the customer.
DECISION_TOOLS: List[Dict[str, Any]] = [
    _BY_NAME["check_return_policy"],
    _BY_NAME["process_refund"],
]

#: Agent 3 — The Communications & Escalation Manager. Routes and notifies.
#: Cannot approve money. This is the separation the brief asks you to enforce.
COMMS_TOOLS: List[Dict[str, Any]] = [_ROUTE_SCHEMA, _ALERT_SCHEMA]

#: Every tool in the system. Useful for a single-agent baseline or for a
#: supervisor node — but do not hand this whole list to all three agents.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    *PART_A_SCHEMAS,
    _AUDIT_SCHEMA,
    _ROUTE_SCHEMA,
    _ALERT_SCHEMA,
]

TOOL_REGISTRY = {
    "get_order_details": get_order_details,
    "get_user_profile": get_user_profile,
    "check_return_policy": check_return_policy,
    "process_refund": process_refund,
    "audit_fraud_risk": audit_fraud_risk,
    "get_escalation_route": get_escalation_route,
    "send_slack_alert": send_slack_alert,
}

#: name -> which agent is allowed to call it. Use it to build your crews, or to
#: assert in a test that no agent reached outside its lane.
TOOL_OWNERSHIP = {
    "get_order_details": "researcher",
    "get_user_profile": "researcher",
    "audit_fraud_risk": "researcher",
    "check_return_policy": "decision",
    "process_refund": "decision",
    "get_escalation_route": "comms",
    "send_slack_alert": "comms",
}


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import pprint

    report = audit_fraud_risk("ORD-1005", "USR-105")
    pprint.pprint(report)
    print()
    pprint.pprint(
        get_escalation_route(
            risk_band=report["risk_band"],
            requested_amount=480.0,
            prior_fraud_flags=report["evidence"]["prior_fraud_flags"],
        )
    )
