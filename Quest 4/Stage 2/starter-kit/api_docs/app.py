"""Interactive API documentation for the Quest #04 Part B tool box.

All four Part A tools, plus the three the crew adds — grouped by which agent
owns them, so the separation of concerns is visible in the docs themselves.

    pip install -r requirements.txt
    uvicorn api_docs.app:app --reload --port 8000

Then open <http://127.0.0.1:8000/docs> and use **Try it out**. Every endpoint
ships with pre-filled examples taken from ``examples/scenarios.md``.

Extra endpoints worth knowing:
  * ``/crew/tool-bundles``  — the per-agent tool schemas (this is the guardrail)
  * ``/crew/outbox``        — read back every alert your crew has sent
  * ``/tools/schemas``      — all seven schemas at once
  * ``/scenarios``          — the Part B scenarios as JSON
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import mock_services as gc  # noqa: E402
import multi_agent_tools as mat  # noqa: E402

Reason = Literal[
    "damaged_on_arrival",
    "wrong_item",
    "item_missing",
    "late_delivery",
    "changed_mind",
]
RiskBand = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high", "critical"]

DESCRIPTION = """
The **GlobalCart Operations API** — the tool box for Quest #04, Part B: the
Multi-Agent Crew.

### The crew and its lanes

```
[Ticket] -> Agent 1 Researcher -> Agent 2 Decision -> Agent 3 Comms -> [Reply + Alert]
```

| Agent | Tools it may call | Tools it may **not** call |
|-------|-------------------|---------------------------|
| 1 · Researcher & Fraud Auditor | `get_order_details`, `get_user_profile`, `audit_fraud_risk` | anything that spends money or messages anyone |
| 2 · Decision Maker / Ops Lead | `check_return_policy`, `process_refund` | the fraud engine, the alert channel |
| 3 · Comms & Escalation Manager | `get_escalation_route`, `send_slack_alert` | **`process_refund`** |

`GET /crew/tool-bundles` returns exactly these three bundles as JSON Schema.
Wiring your crew from that endpoint — rather than handing all seven tools to all
three agents — is the cheapest guardrail in the system, and the brief asks for it.

### Three things that will bite you

* **A claim can be eligible and still must not be paid.** `ORD-1005` passes
  `check_return_policy` but scores 90/100 on the fraud rulebook. An agent that
  reads only the policy verdict approves a fraudulent payout.
* **`audit_fraud_risk` is a rule engine, not an opinion.** It is deterministic.
  Do not ask the model to estimate a risk score, and do not let it overrule the
  band it gets back.
* **Do not alert on clean tickets.** When `get_escalation_route` returns
  `escalation_required: false`, sending an alert anyway is a defect. Whoever
  grades your submission will read `outbox/alerts.jsonl`.

### Determinism

All date arithmetic is measured against `policies.json -> reference_date`
(**2026-08-05**). `send_slack_alert` returns a `message_ts` derived from a
SHA-256 digest of the payload, so it is stable across processes and runs.
Override the date with the `QUEST4_REFERENCE_DATE` environment variable.

### Slack

`send_slack_alert` appends to `outbox/alerts.jsonl` and nothing else, unless you
set `SLACK_WEBHOOK_URL` — then it also POSTs to that incoming webhook. Offline is
the default so nobody needs a workspace to finish the quest.
"""

TAGS_METADATA = [
    {"name": "Agent 1 · Researcher", "description": "Read the world and score the risk. No spending, no messaging."},
    {"name": "Agent 2 · Decision", "description": "Consult policy and decide. The refund cap is enforced here."},
    {"name": "Agent 3 · Comms", "description": "Route the escalation and send the alert. Cannot approve money."},
    {"name": "Reference", "description": "The rulebooks: policies, fraud rules, escalation channels."},
    {"name": "Crew integration", "description": "Per-agent tool bundles, the outbox, and the test scenarios."},
]

app = FastAPI(
    title="GlobalCart Operations API — Quest #04, Part B",
    description=DESCRIPTION,
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
    contact={"name": "Place IL — Quest #04"},
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class OrderRequest(BaseModel):
    order_id: str = Field(..., description="Order identifier.", examples=["ORD-1005"])


class UserRequest(BaseModel):
    user_id: str = Field(..., description="Customer identifier.", examples=["USR-105"])


class PolicyRequest(BaseModel):
    order_id: str = Field(..., description="Order to evaluate.", examples=["ORD-1005"])
    reason: Reason = Field("damaged_on_arrival", description="Why the customer is asking.")


class RefundRequest(BaseModel):
    order_id: str = Field(..., description="Order to refund.", examples=["ORD-1005"])
    amount: float = Field(..., gt=0, description="Refund amount in USD.", examples=[480.0])
    reason: Reason = Field("damaged_on_arrival", description="Why the refund is being issued.")


class AuditRequest(BaseModel):
    order_id: str = Field(..., description="Order under investigation.", examples=["ORD-1005"])
    user_id: Optional[str] = Field(
        None,
        description=(
            "Customer to score. Optional — defaults to the order's owner. Pass it "
            "explicitly to assert that the ticket's claimed customer really owns "
            "the order; a mismatch returns USER_ORDER_MISMATCH."
        ),
        examples=["USR-105"],
    )


class RouteRequest(BaseModel):
    risk_band: RiskBand = Field("low", description="From audit_fraud_risk.")
    requested_amount: float = Field(0.0, ge=0, description="Refund amount asked for, in USD.", examples=[480.0])
    prior_fraud_flags: int = Field(0, ge=0, description="From the customer profile.", examples=[1])
    order_status: str = Field("delivered", description="From the order.")
    verdict: str = Field("ELIGIBLE", description="The policy verdict.")


class AlertRequest(BaseModel):
    channel_id: str = Field(..., description="Channel id from get_escalation_route.", examples=["CH-FRAUD"])
    severity: Severity = Field(..., description="Alert severity.", examples=["critical"])
    payload: Dict[str, Any] = Field(
        ...,
        description="Structured facts behind the alert. Keep it machine readable.",
        examples=[
            {
                "order_id": "ORD-1005",
                "user_id": "USR-105",
                "risk_score": 90,
                "risk_band": "high",
                "triggered_rules": "FR-01, FR-02, FR-04, FR-05, FR-08",
                "requested_amount": 480.0,
                "evidence": "3 claims in 60 days; address re-routed 2 days pre-delivery",
            }
        ],
    )
    message: Optional[str] = Field(
        None, description="Optional body. Rendered from the channel template if omitted."
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["ORDER_NOT_FOUND"])
    message: str = Field(..., examples=["No order found with id 'ORD-9999'."])


ERROR_STATUS = {
    "ORDER_NOT_FOUND": 404,
    "USER_NOT_FOUND": 404,
    "CHANNEL_NOT_FOUND": 404,
    "USER_ORDER_MISMATCH": 409,
    "INVALID_AMOUNT": 422,
    "INVALID_REASON": 422,
    "INVALID_SEVERITY": 422,
}


def _unwrap(result: Dict[str, Any]) -> Dict[str, Any]:
    """Map a structured error dict onto the matching HTTP status, body unchanged."""
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=ERROR_STATUS.get(result["error"], 400), detail=result)
    return result


# --------------------------------------------------------------------------- #
# Agent 1 — Researcher & Fraud Auditor
# --------------------------------------------------------------------------- #

ORDER_DETAILS_RESPONSE_EXAMPLE = {
    "order_id": "ORD-1005",
    "user_id": "USR-105",
    "status": "delivered",
    "order_date": "2026-07-24",
    "delivery_date": "2026-07-31",
    "total_amount": 480.0,
    "currency": "USD",
    "channel": "web",
    "items": [
        {
            "sku": "SKU-TABL-09",
            "name": "GlobalCart Tablet Pro 11",
            "category": "electronics",
            "qty": 1,
            "unit_price": 480.0,
            "condition": "damaged_on_arrival",
        }
    ],
    "shipping_address": {
        "line1": "3 Ha-Namal St",
        "city": "Ashdod",
        "country": "IL",
        "postal_code": "7761001",
    },
    "address_changed_at": "2026-07-29",
    "payment_method_last4": "7734",
}

@app.post(
    "/tools/get_order_details",
    tags=["Agent 1 · Researcher"],
    summary="Look up an order by id",
    responses={
        200: {"content": {"application/json": {"example": ORDER_DETAILS_RESPONSE_EXAMPLE}}},
        404: {"model": ErrorResponse},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "fraud_case": {"summary": "B1 — the headline fraud case", "value": {"order_id": "ORD-1005"}},
                        "new_account_case": {"summary": "B2 — new account, $890, item missing", "value": {"order_id": "ORD-1012"}},
                        "clean_case": {"summary": "B3 — clean VIP ticket", "value": {"order_id": "ORD-1001"}},
                    }
                }
            },
            "required": True,
        }
    },
)
def post_order_details(payload: OrderRequest) -> Dict[str, Any]:
    """Shipping status, dates, total, items and their condition, address, and
    whether the address was changed after the order was placed.

    `address_changed_at` is a fraud signal — rule `FR-02` fires on it.
    """
    return _unwrap(gc.get_order_details(payload.order_id))


@app.post(
    "/tools/get_user_profile",
    tags=["Agent 1 · Researcher"],
    summary="Look up a customer profile by user id",
    responses={404: {"model": ErrorResponse}},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "risky": {"summary": "B1 — score 61, 1 flag, 3 claims", "value": {"user_id": "USR-105"}},
                        "brand_new": {"summary": "B2 — account 8 days old", "value": {"user_id": "USR-109"}},
                        "clean_vip": {"summary": "B3 — VIP, clean history", "value": {"user_id": "USR-101"}},
                    }
                }
            },
            "required": True,
        }
    },
)
def post_user_profile(payload: UserRequest) -> Dict[str, Any]:
    """Tier, account age, lifetime value, refund history, prior fraud flags and
    inherited fraud score. Rules `FR-01`, `FR-03`, `FR-04`, `FR-05` and `FR-08`
    all read from here."""
    return _unwrap(gc.get_user_profile(payload.user_id))


@app.post(
    "/tools/audit_fraud_risk",
    tags=["Agent 1 · Researcher"],
    summary="Run the fraud rulebook over an order and its customer",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse, "description": "Customer does not own the order"}},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "high_risk_repeat_claims": {
                            "summary": "B1 — expect 90/100, high, 5 rules",
                            "value": {"order_id": "ORD-1005", "user_id": "USR-105"},
                        },
                        "high_risk_new_account": {
                            "summary": "B2 — expect 60/100, high, 4 different rules",
                            "value": {"order_id": "ORD-1012"},
                        },
                        "clean": {"summary": "B3 — expect 0/100, low, no rules", "value": {"order_id": "ORD-1001"}},
                        "mismatch": {
                            "summary": "B5 — customer does not own the order (409)",
                            "value": {"order_id": "ORD-1001", "user_id": "USR-105"},
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
def post_audit_fraud_risk(payload: AuditRequest) -> Dict[str, Any]:
    """A **deterministic** additive-weight rule engine over `fraud_rules.json`.

    Returns `risk_score` out of 100, a `risk_band`, every rule that fired with
    the reason it fired, and the raw `evidence` behind it. `blocks_automatic_refund`
    is true when the band is high.

    Pass the whole report to the Decision agent — `triggered_rules` and
    `evidence` are what make the final decision auditable. Do not ask a model to
    estimate a score instead of calling this.
    """
    return _unwrap(mat.audit_fraud_risk(payload.order_id, payload.user_id))


# --------------------------------------------------------------------------- #
# Agent 2 — Decision Maker
# --------------------------------------------------------------------------- #

@app.post(
    "/tools/check_return_policy",
    tags=["Agent 2 · Decision"],
    summary="Decide whether a claim is still eligible",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "eligible_but_risky": {
                            "summary": "B1 — ELIGIBLE, yet requires_escalation is true",
                            "value": {"order_id": "ORD-1005", "reason": "damaged_on_arrival"},
                        },
                        "outside_window": {"summary": "60 days since delivery", "value": {"order_id": "ORD-1003", "reason": "changed_mind"}},
                        "non_returnable": {"summary": "Digital gift card", "value": {"order_id": "ORD-1008", "reason": "changed_mind"}},
                    }
                }
            },
            "required": True,
        }
    },
)
def post_check_return_policy(payload: PolicyRequest) -> Dict[str, Any]:
    """Applies the return window, VIP overrides, non-returnable categories and
    order status, and returns the verdict plus the `policy_id`s behind it.

    Watch `requires_escalation`: a claim can be `ELIGIBLE` and still need a human.
    """
    return _unwrap(gc.check_return_policy(payload.order_id, payload.reason))


@app.post(
    "/tools/process_refund",
    tags=["Agent 2 · Decision"],
    summary="Issue a refund — or refuse and demand escalation",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "escalation_required": {
                            "summary": "B1 — $480 → ESCALATION_REQUIRED",
                            "value": {"order_id": "ORD-1005", "amount": 480.0, "reason": "damaged_on_arrival"},
                        },
                        "small_but_still_escalates": {
                            "summary": "B1 — even $25 escalates for this customer",
                            "value": {"order_id": "ORD-1005", "amount": 25.0, "reason": "damaged_on_arrival"},
                        },
                        "approved": {
                            "summary": "B3 — $35 on a clean VIP order → APPROVED",
                            "value": {"order_id": "ORD-1001", "amount": 35.0, "reason": "damaged_on_arrival"},
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
def post_process_refund(payload: RefundRequest) -> Dict[str, Any]:
    """The only money-moving endpoint, and the only agent that may reach it is
    Agent 2.

    Returns `APPROVED`, `REJECTED` or `ESCALATION_REQUIRED`. Requests above the
    automatic cap, and requests from customers whose risk profile demands review,
    always come back as `ESCALATION_REQUIRED` with `approved_amount: 0.0`.
    """
    return _unwrap(gc.process_refund(payload.order_id, payload.amount, payload.reason))


# --------------------------------------------------------------------------- #
# Agent 3 — Comms & Escalation
# --------------------------------------------------------------------------- #

@app.post(
    "/tools/get_escalation_route",
    tags=["Agent 3 · Comms"],
    summary="Pick the one channel this case should be handed to",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "fraud": {
                            "summary": "B1 — high risk → #fraud-security",
                            "value": {"risk_band": "high", "requested_amount": 480.0, "prior_fraud_flags": 1},
                        },
                        "finance": {
                            "summary": "$250+ at low risk → #finance-approvals",
                            "value": {"risk_band": "low", "requested_amount": 250.0, "prior_fraud_flags": 0},
                        },
                        "tier2": {
                            "summary": "Over the cap, under $250 → #support-tier2",
                            "value": {"risk_band": "medium", "requested_amount": 150.0, "prior_fraud_flags": 0},
                        },
                        "logistics": {
                            "summary": "Delayed shipment, no refund → #logistics-delays",
                            "value": {"risk_band": "low", "requested_amount": 0.0, "order_status": "delayed"},
                        },
                        "no_escalation": {
                            "summary": "B3 — clean case → escalation_required: false",
                            "value": {"risk_band": "low", "requested_amount": 35.0, "prior_fraud_flags": 0},
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
def post_get_escalation_route(payload: RouteRequest) -> Dict[str, Any]:
    """Channels are evaluated in ascending `priority`, so exactly one
    destination comes back.

    When `escalation_required` is `false`, send the customer reply and **no**
    alert. Alerting on clean tickets is a defect.
    """
    return mat.get_escalation_route(
        risk_band=payload.risk_band,
        requested_amount=payload.requested_amount,
        prior_fraud_flags=payload.prior_fraud_flags,
        order_status=payload.order_status,
        verdict=payload.verdict,
    )


@app.post(
    "/tools/send_slack_alert",
    tags=["Agent 3 · Comms"],
    summary="Send a structured alert to an external channel (side effect)",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "fraud_alert": {
                            "summary": "B1 — the alert the video should show",
                            "value": {
                                "channel_id": "CH-FRAUD",
                                "severity": "critical",
                                "payload": {
                                    "order_id": "ORD-1005",
                                    "user_id": "USR-105",
                                    "risk_score": 90,
                                    "risk_band": "high",
                                    "triggered_rules": "FR-01, FR-02, FR-04, FR-05, FR-08",
                                    "requested_amount": 480.0,
                                    "evidence": "3 claims in 60 days; address re-routed 2 days pre-delivery",
                                },
                            },
                        },
                        "finance_alert": {
                            "summary": "Refund approval needed",
                            "value": {
                                "channel_id": "CH-FINANCE",
                                "severity": "high",
                                "payload": {"order_id": "ORD-1002", "user_id": "USR-102", "requested_amount": 150.0},
                            },
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
def post_send_slack_alert(payload: AlertRequest) -> Dict[str, Any]:
    """Appends one JSON object to `outbox/alerts.jsonl`, and POSTs to Slack as
    well if `SLACK_WEBHOOK_URL` is set.

    Call it only after `get_escalation_route` returned
    `escalation_required: true`. If `message` is omitted, one is rendered from
    the channel's template using `payload`.
    """
    return _unwrap(
        mat.send_slack_alert(payload.channel_id, payload.severity, payload.payload, payload.message)
    )


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #

@app.get("/policies", tags=["Reference"], summary="The GlobalCart rulebook")
def policies() -> Dict[str, Any]:
    """Every `policy_id` your Decision agent can cite."""
    return gc.get_policies()


@app.get("/fraud-rules", tags=["Reference"], summary="The fraud rulebook")
def fraud_rules() -> Dict[str, Any]:
    """The eight weighted rules, the risk bands, and where each signal comes from."""
    return mat.get_fraud_rules()


@app.get("/escalation-channels", tags=["Reference"], summary="The escalation routing table")
def escalation_channels() -> Dict[str, Any]:
    """The four channels, their trigger conditions, SLAs and message templates."""
    return mat.get_escalation_channels()


@app.get("/orders", tags=["Reference"], summary="List every order id")
def list_orders() -> Dict[str, List[str]]:
    """Convenience endpoint for exploring the dataset."""
    return {"order_ids": gc.list_order_ids()}


# --------------------------------------------------------------------------- #
# Crew integration
# --------------------------------------------------------------------------- #

@app.get(
    "/crew/tool-bundles",
    tags=["Crew integration"],
    summary="Per-agent tool schemas — wire your crew from this",
)
def tool_bundles() -> Dict[str, Any]:
    """The three bundles, as JSON Schema, plus the ownership map.

    Build each agent's tool list from its own bundle rather than handing all
    seven tools to all three agents. That is the separation of concerns the brief
    asks for, and it is enforceable in a test.
    """
    return {
        "researcher": {"tools": mat.RESEARCHER_TOOLS, "count": len(mat.RESEARCHER_TOOLS)},
        "decision": {"tools": mat.DECISION_TOOLS, "count": len(mat.DECISION_TOOLS)},
        "comms": {"tools": mat.COMMS_TOOLS, "count": len(mat.COMMS_TOOLS)},
        "ownership": mat.TOOL_OWNERSHIP,
    }


@app.get("/tools/schemas", tags=["Crew integration"], summary="All seven tool schemas")
def tool_schemas() -> Dict[str, Any]:
    """Exactly what a model sees when it decides which tool to call. If an agent
    picks the wrong tool, the fix is usually in these descriptions."""
    return {"tools": mat.TOOL_SCHEMAS, "count": len(mat.TOOL_SCHEMAS)}


@app.get("/crew/outbox", tags=["Crew integration"], summary="Read back every alert sent")
def outbox() -> Dict[str, Any]:
    """Everything `send_slack_alert` has written to `outbox/alerts.jsonl`.

    Use this in your demo video to prove the alert actually left the system —
    and to check you are not alerting on clean tickets.
    """
    alerts = mat.read_outbox()
    return {"count": len(alerts), "alerts": alerts}


@app.get("/scenarios", tags=["Crew integration"], summary="The Part B scenarios, as JSON")
def scenarios() -> Dict[str, Any]:
    """The regression suite from `examples/scenarios.md`, machine-readable."""
    return {
        "reference_date": str(gc.reference_date()),
        "scenarios": [
            {
                "id": "B1",
                "name": "Headline fraud case — repeat claims and a re-routed address",
                "ticket": "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars, this keeps happening.",
                "order_id": "ORD-1005",
                "user_id": "USR-105",
                "expected_risk_score": 90,
                "expected_risk_band": "high",
                "expected_rules": ["FR-01", "FR-02", "FR-04", "FR-05", "FR-08"],
                "expected_refund_status": "ESCALATION_REQUIRED",
                "expected_channel": "CH-FRAUD",
            },
            {
                "id": "B2",
                "name": "New account, high value, item never arrived",
                "ticket": "I ordered a laptop (ORD-1012), the box arrived but it was empty. I need the 890 dollars back.",
                "order_id": "ORD-1012",
                "user_id": "USR-109",
                "expected_risk_score": 60,
                "expected_risk_band": "high",
                "expected_rules": ["FR-02", "FR-03", "FR-06", "FR-07"],
                "expected_refund_status": "ESCALATION_REQUIRED",
                "expected_channel": "CH-FRAUD",
            },
            {
                "id": "B3",
                "name": "Clean case — resolve automatically, send no alert",
                "ticket": "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box.",
                "order_id": "ORD-1001",
                "user_id": "USR-101",
                "expected_risk_score": 0,
                "expected_risk_band": "low",
                "expected_rules": [],
                "expected_refund_status": "APPROVED",
                "expected_channel": None,
            },
            {
                "id": "B5",
                "name": "Consistency trap — claimed customer does not own the order",
                "ticket": "Order ORD-1001, this is Ronen (USR-105), refund me.",
                "order_id": "ORD-1001",
                "user_id": "USR-105",
                "expected_error": "USER_ORDER_MISMATCH",
                "expected_refund_status": "NONE",
                "expected_channel": "CH-FRAUD",
            },
        ],
    }


@app.get("/healthz", tags=["Crew integration"], summary="Liveness probe")
def healthz() -> Dict[str, Any]:
    """Confirms the server is up and every fixture loaded."""
    return {
        "status": "ok",
        "reference_date": str(gc.reference_date()),
        "orders_loaded": len(gc.list_order_ids()),
        "fraud_rules_loaded": len(mat.get_fraud_rules()["rules"]),
        "channels_loaded": len(mat.get_escalation_channels()["channels"]),
        "tools": [s["name"] for s in mat.TOOL_SCHEMAS],
        "slack_webhook_configured": bool(__import__("os").environ.get("SLACK_WEBHOOK_URL")),
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
