#!/usr/bin/env python3
"""Sanity check for the Quest #04 Part B tool box.

Covers everything Part A covered, plus the fraud engine, the escalation router
and the alert outbox. Run it before you write a single line of crew code.

    cd "Quest 4 - Stage 2/starter-kit"
    python3 examples/verify_scenarios.py

Standard library only. No pip install required. Writes and then cleans up
``outbox/alerts.jsonl``.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import mock_services as gc  # noqa: E402
import multi_agent_tools as mat  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(label: str, actual, expected) -> None:
    global PASSED
    if actual == expected:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL  {label}\n          expected: {expected!r}\n          actual:   {actual!r}")


print(f"reference date: {gc.reference_date()}\n")

# --------------------------------------------------------------------------- #
print("Part A regression — the four original tools still behave")
check("VIP $35 approves", gc.process_refund("ORD-1001", 35.0)["status"], "APPROVED")
check("$150 escalates", gc.process_refund("ORD-1002", 150.0)["status"], "ESCALATION_REQUIRED")
check("60 days rejects", gc.check_return_policy("ORD-1003", "changed_mind")["verdict"], "OUTSIDE_RETURN_WINDOW")
check("gift card rejects", gc.check_return_policy("ORD-1008")["verdict"], "NON_RETURNABLE_CATEGORY")

# --------------------------------------------------------------------------- #
print("\nScenario B1 — the headline fraud case: ORD-1005 / USR-105")
audit = mat.audit_fraud_risk("ORD-1005", "USR-105")
fired = [r["rule_id"] for r in audit["triggered_rules"]]
check("risk score", audit["risk_score"], 90)
check("risk band", audit["risk_band"], "high")
check("rules fired", sorted(fired), ["FR-01", "FR-02", "FR-04", "FR-05", "FR-08"])
check("blocks automatic refund", audit["blocks_automatic_refund"], True)
check("security channel required", audit["requires_security_channel"], True)
check("evidence: 3 claims in 60 days", audit["evidence"]["claims_in_last_60_days"], 3)
check("evidence: address changed 2 days pre-delivery", audit["evidence"]["address_change_days_before_delivery"], 2)

route = mat.get_escalation_route(
    risk_band=audit["risk_band"],
    requested_amount=480.0,
    prior_fraud_flags=audit["evidence"]["prior_fraud_flags"],
)
check("routes to fraud channel", route["channel_id"], "CH-FRAUD")
check("channel name", route["channel"], "#fraud-security")
check("severity is critical", route["severity"], "critical")
check("escalation required", route["escalation_required"], True)

# --------------------------------------------------------------------------- #
print("\nScenario B2 — new account, high value, item 'missing': ORD-1012 / USR-109")
audit2 = mat.audit_fraud_risk("ORD-1012")
fired2 = [r["rule_id"] for r in audit2["triggered_rules"]]
check("risk score", audit2["risk_score"], 60)
check("risk band", audit2["risk_band"], "high")
check("rules fired", sorted(fired2), ["FR-02", "FR-03", "FR-06", "FR-07"])
check("user resolved from order", audit2["user_id"], "USR-109")
check("account age", audit2["evidence"]["account_age_days"], 8)

# --------------------------------------------------------------------------- #
print("\nScenario B3 — clean case: no escalation at all")
audit3 = mat.audit_fraud_risk("ORD-1001")
check("risk score is zero", audit3["risk_score"], 0)
check("risk band is low", audit3["risk_band"], "low")
check("no rules fired", audit3["triggered_rules"], [])
clean = mat.get_escalation_route(risk_band="low", requested_amount=35.0, prior_fraud_flags=0)
check("no escalation", clean["escalation_required"], False)
check("no channel", clean["channel_id"], None)

# --------------------------------------------------------------------------- #
print("\nRouting table — first match wins, in priority order")
check(
    "prior flag alone reaches fraud",
    mat.get_escalation_route(risk_band="low", requested_amount=10.0, prior_fraud_flags=1)["channel_id"],
    "CH-FRAUD",
)
check(
    "$250+ reaches finance",
    mat.get_escalation_route(risk_band="low", requested_amount=250.0)["channel_id"],
    "CH-FINANCE",
)
check(
    "over cap but under $250 reaches tier 2",
    mat.get_escalation_route(risk_band="medium", requested_amount=150.0)["channel_id"],
    "CH-SUPPORT-T2",
)
check(
    "rejected claim reaches tier 2",
    mat.get_escalation_route(risk_band="low", requested_amount=0.0, verdict="OUTSIDE_RETURN_WINDOW")["channel_id"],
    "CH-SUPPORT-T2",
)
check(
    "delayed shipment reaches logistics",
    mat.get_escalation_route(risk_band="low", requested_amount=0.0, order_status="delayed")["channel_id"],
    "CH-LOGISTICS",
)

# --------------------------------------------------------------------------- #
print("\nAlert outbox — the side effect actually lands")
outbox = BASE / "outbox" / "alerts.jsonl"
before = len(mat.read_outbox())
sent = mat.send_slack_alert(
    channel_id="CH-FRAUD",
    severity="critical",
    payload={
        "order_id": "ORD-1005",
        "user_id": "USR-105",
        "risk_score": audit["risk_score"],
        "risk_band": audit["risk_band"],
        "triggered_rules": ", ".join(fired),
        "requested_amount": 480.0,
        "evidence": "3 claims in 60 days; address re-routed 2 days pre-delivery",
    },
)
check("alert delivered", sent["delivered"], True)
check("transport is offline outbox", sent["transport"], "outbox")
check("no webhook configured", sent["webhook_status"], None)
check("outbox grew by one", len(mat.read_outbox()) - before, 1)
check("rendered template mentions the order", "ORD-1005" in sent["message"], True)
check("rendered template mentions the score", "90/100" in sent["message"], True)

ts_first = sent["message_ts"]
ts_again = mat.send_slack_alert(
    channel_id="CH-FRAUD", severity="critical", payload={"order_id": "ORD-1005", "risk_score": 90}
)["message_ts"]
ts_repeat = mat.send_slack_alert(
    channel_id="CH-FRAUD", severity="critical", payload={"risk_score": 90, "order_id": "ORD-1005"}
)["message_ts"]
check("message_ts is deterministic for identical payloads", ts_again, ts_repeat)
check("message_ts differs for different payloads", ts_first != ts_again, True)

check("unknown channel", mat.send_slack_alert("CH-NOPE", "high", {})["error"], "CHANNEL_NOT_FOUND")
check("bad severity", mat.send_slack_alert("CH-FRAUD", "urgent-ish", {})["error"], "INVALID_SEVERITY")

# --------------------------------------------------------------------------- #
print("\nGuardrails — mismatched customer and missing data")
check(
    "order/user mismatch is caught",
    mat.audit_fraud_risk("ORD-1001", "USR-105")["error"],
    "USER_ORDER_MISMATCH",
)
check("unknown order", mat.audit_fraud_risk("ORD-9999")["error"], "ORDER_NOT_FOUND")

# --------------------------------------------------------------------------- #
print("\nSeparation of concerns — each agent gets only its own tools")
def names(bundle):
    return sorted(t["name"] for t in bundle)

check("researcher tools", names(mat.RESEARCHER_TOOLS), ["audit_fraud_risk", "get_order_details", "get_user_profile"])
check("decision tools", names(mat.DECISION_TOOLS), ["check_return_policy", "process_refund"])
check("comms tools", names(mat.COMMS_TOOLS), ["get_escalation_route", "send_slack_alert"])
check("comms cannot refund", "process_refund" in names(mat.COMMS_TOOLS), False)
check("researcher cannot refund", "process_refund" in names(mat.RESEARCHER_TOOLS), False)
check("researcher cannot message", "send_slack_alert" in names(mat.RESEARCHER_TOOLS), False)
check("seven tools in total", len(mat.TOOL_SCHEMAS), 7)
check("registry matches schemas", sorted(mat.TOOL_REGISTRY), names(mat.TOOL_SCHEMAS))
check(
    "bundles partition the tool set",
    names(mat.RESEARCHER_TOOLS + mat.DECISION_TOOLS + mat.COMMS_TOOLS),
    names(mat.TOOL_SCHEMAS),
)

# --------------------------------------------------------------------------- #
# Leave the workspace as we found it.
if outbox.exists():
    outbox.unlink()

print()
if FAILED:
    print(f"{PASSED} passed, {len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print(f"All {PASSED} checks passed. The crew's tool box is behaving as documented.")
