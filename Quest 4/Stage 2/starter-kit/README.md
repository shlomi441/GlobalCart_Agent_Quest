# GlobalCart Starter Kit — Quest #04, Part B

The tool box for the **Multi-Agent Operations Crew**. This kit is
**self-contained**: it includes everything from Part A plus the fraud engine, the
escalation router and the outbound alert channel. You do not need the Part A
folder alongside it.

> **The tools and the data are not yours to change.** Your crew is what gets
> graded.

---

## What's in the box

```
starter-kit/
├── mock_services.py          Part A: the four original tools (unchanged)
├── multi_agent_tools.py      Part B: fraud engine, router, alert + per-role bundles
├── requirements.txt          only needed for the API docs server
├── data/
│   ├── orders.json           14 orders
│   ├── users.json            9 customers
│   ├── policies.json         the refund rulebook
│   ├── fraud_rules.json      8 weighted rules, 3 risk bands
│   └── escalation_channels.json   4 channels, SLAs, message templates
├── examples/
│   ├── scenarios.md          the crew scenarios with expected outcomes
│   └── verify_scenarios.py   51 assertions — run this first
├── outbox/                   where send_slack_alert writes (gitignored content)
└── api_docs/
    └── app.py                FastAPI + Swagger UI, grouped by agent
```

---

## Quickstart

```bash
cd "Quest 4 - Stage 2/starter-kit"
python3 examples/verify_scenarios.py     # expect: All 51 checks passed.

pip install -r requirements.txt
uvicorn api_docs.app:app --reload --port 8000
```

Open **<http://127.0.0.1:8000/docs>**. The endpoints are grouped by which agent
owns them, and every one has pre-filled examples for the scenarios in the brief.
Two endpoints are worth visiting first:

* **`GET /crew/tool-bundles`** — the three per-agent tool bundles. Wire your crew
  from this.
* **`GET /crew/outbox`** — read back every alert your crew has sent.

---

## The crew

```
[Customer ticket]
      │
      ▼
Agent 1 — Researcher & Fraud Auditor          RESEARCHER_TOOLS
      │   get_order_details · get_user_profile · audit_fraud_risk
      │   ↓ risk report: score, band, triggered rules, evidence
      ▼
Agent 2 — Decision Maker / Ops Lead           DECISION_TOOLS
      │   check_return_policy · process_refund
      │   ↓ final operational decision
      ▼
Agent 3 — Comms & Escalation Manager          COMMS_TOOLS
      │   get_escalation_route · send_slack_alert
      ▼
[Customer reply]  +  [Alert in outbox/alerts.jsonl]
```

### Build each agent from its own bundle

```python
import multi_agent_tools as mat

researcher_tools = mat.RESEARCHER_TOOLS   # 3 schemas
decision_tools   = mat.DECISION_TOOLS     # 2 schemas
comms_tools      = mat.COMMS_TOOLS        # 2 schemas
```

Agent 3 has **no** access to `process_refund`. Agent 1 cannot approve money or
message anyone. That is the separation of concerns the brief asks for, and it is
worth asserting in a test:

```python
assert "process_refund" not in {t["name"] for t in mat.COMMS_TOOLS}
```

`mat.TOOL_OWNERSHIP` maps every tool name to the agent that owns it.
`mat.TOOL_SCHEMAS` is all seven at once — useful for a single-agent baseline, but
do not hand it to all three agents.

---

## The three new tools

| Tool | Owner | Returns |
|---|---|---|
| `audit_fraud_risk(order_id, user_id=None)` | Agent 1 | `risk_score` 0-100, `risk_band`, `triggered_rules[]`, `evidence`, `blocks_automatic_refund` |
| `get_escalation_route(risk_band, requested_amount, prior_fraud_flags, order_status, verdict)` | Agent 3 | One channel, or `escalation_required: false` |
| `send_slack_alert(channel_id, severity, payload, message=None)` | Agent 3 | `delivered`, `message_ts`, `transport`, rendered `message` |

Plus three read-only helpers for you: `get_fraud_rules()`,
`get_escalation_channels()`, `read_outbox()`.

### The fraud rulebook

Eight additive weighted rules over `fraud_rules.json`. Bands: **low** 0-29,
**medium** 30-59, **high** 60-100.

| Rule | Weight | Fires when |
|---|---|---|
| `FR-01` repeat_refund_claims | 25 | 3+ claims in the trailing 60 days |
| `FR-02` last_minute_address_change | 20 | Address changed ≤7 days before delivery |
| `FR-03` new_account_high_value | 20 | Account <30 days old **and** order ≥$500 |
| `FR-04` prior_fraud_flag | 15 | Any prior fraud flag on record |
| `FR-05` high_initial_fraud_score | 15 | Inherited score ≥60 |
| `FR-06` high_value_claim | 10 | Order total ≥$500 |
| `FR-07` item_never_arrived_claim | 10 | An item's condition is `missing` |
| `FR-08` refund_amount_velocity | 15 | ≥$250 refunded in the trailing 60 days |

`audit_fraud_risk` is a **deterministic rule engine, not an opinion**. Do not ask
a model to estimate a risk score, and do not let it overrule the band it gets
back. Pass the whole report downstream — `triggered_rules` and `evidence` are
what make the final decision auditable.

### The escalation router

Channels are evaluated in ascending `priority`, so exactly one destination comes
back.

| Priority | Channel | Severity | SLA | Fires on |
|---|---|---|---|---|
| 1 | `#fraud-security` | critical | 15 min | `risk_band == "high"` or any prior fraud flag |
| 2 | `#finance-approvals` | high | 2 h | Requested amount ≥ $250 |
| 3 | `#support-tier2` | medium | 4 h | Over the cap but under $250, or a rejected claim |
| 4 | `#logistics-delays` | low | 8 h | Order `delayed` / `processing`, no refund asked |

When nothing matches, `escalation_required` is `false` and `channel_id` is
`None`. **Send no alert in that case.** An over-eager Comms agent that pings
`#fraud-security` on every ticket is a real failure mode, and the outbox is where
it shows up.

### The alert channel

`send_slack_alert` appends one JSON object per line to `outbox/alerts.jsonl`, and
nothing else — the whole quest works offline. For the demo video, point it at a
real Slack incoming webhook:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

The response then reports `transport: "outbox+webhook"` and a `webhook_status`.
The offline record is written either way. Read it back with:

```python
for alert in mat.read_outbox():
    print(alert["channel"], alert["severity"], alert["payload"]["order_id"])
```

---

## Guardrails you are expected to build

The brief asks you to explain three mechanisms in your `README.md`. Here is what
the kit gives you and what remains yours:

| Mechanism | The kit provides | You must provide |
|---|---|---|
| **Financial authority** | `process_refund` returns `ESCALATION_REQUIRED` above the cap — it cannot be talked into `APPROVED` | Your crew must *detect* that and report it honestly, never claim a refund was issued |
| **Separation of concerns** | Three per-role tool bundles + `TOOL_OWNERSHIP` | Wire each agent from its own bundle, and say so in the README |
| **Infinite loops** | Structured errors instead of exceptions, so a failure is a readable signal | A `max_iterations` / stop condition on your crew, and a defined behaviour when Agent 1's report is incomplete: escalate, do not re-dispatch forever |
| **Memory context** | Every tool returns flat, JSON-serialisable dicts, ready for a Pydantic model | Define the schemas that carry state from Agent 1 → 2 → 3, and document what each stage adds |

A suggested shape for the handoff — the brief recommends Pydantic or TypedDict:

```python
from pydantic import BaseModel
from typing import Any, Literal

class RiskReport(BaseModel):          # Agent 1 -> Agent 2
    order_id: str
    user_id: str
    risk_score: int
    risk_band: Literal["low", "medium", "high"]
    triggered_rules: list[dict[str, Any]]
    blocks_automatic_refund: bool

class Decision(BaseModel):            # Agent 2 -> Agent 3
    order_id: str
    verdict: str
    refund_status: Literal["APPROVED", "REJECTED", "ESCALATION_REQUIRED"]
    approved_amount: float
    applicable_policies: list[str]
    rationale: str
```

---

## Errors

Same contract as Part A: business failures return a dict, they do not raise.

| Code | Meaning |
|---|---|
| `ORDER_NOT_FOUND` / `USER_NOT_FOUND` | No such record |
| `USER_ORDER_MISMATCH` | The ticket's claimed customer does not own the order — a red flag; escalate, do not retry with a different id |
| `CHANNEL_NOT_FOUND` | Unknown `channel_id` |
| `INVALID_SEVERITY` | Not one of `low` / `medium` / `high` / `critical` |
| `INVALID_AMOUNT` / `INVALID_REASON` | Bad refund input |

## Determinism

Date arithmetic uses `policies.json -> reference_date` (**2026-08-05**), override
with `QUEST4_REFERENCE_DATE`. `send_slack_alert`'s `message_ts` is a SHA-256
digest of the payload, so it is stable across processes and runs — two identical
payloads produce the same timestamp.

---

## Before you submit

| Scenario | Expected |
|---|---|
| `ORD-1005` / `USR-105` | 90/100 **high**, fires `FR-01 FR-02 FR-04 FR-05 FR-08`, refund **ESCALATION_REQUIRED**, alert to `#fraud-security` |
| `ORD-1012` / `USR-109` | 60/100 **high** from a *different* rule set (`FR-02 FR-03 FR-06 FR-07`) |
| `ORD-1001` / `USR-101` | 0/100 **low**, refund **APPROVED**, **no alert at all** |
| `ORD-1001` + `USR-105` | `USER_ORDER_MISMATCH` — escalate, do not loop |

And do not regress on Part A: `ORD-1002` at $150 still escalates, `ORD-1003` is
still outside the window, `ORD-1008` is still non-returnable.

Questions about the brief go to the Place IL Quest channel.
