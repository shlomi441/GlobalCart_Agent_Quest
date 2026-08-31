# Test scenarios — Quest #04, Part B

Part A's nine tickets still apply — your crew must not regress on them. What
follows is what Part B adds: fraud scoring, routing, and the outbound alert.

Verify the fixtures and the engines with:

```bash
python3 examples/verify_scenarios.py
```

All date arithmetic is measured against `policies.json -> reference_date`
(**2026-08-05**), so nothing drifts over time.

---

## The crew, and who is allowed to touch what

```
[Customer ticket]
      │
      ▼
Agent 1 — Researcher & Fraud Auditor      RESEARCHER_TOOLS
      │   get_order_details, get_user_profile, audit_fraud_risk
      │   → risk report: score, band, triggered rules, evidence
      ▼
Agent 2 — Decision Maker / Ops Lead        DECISION_TOOLS
      │   check_return_policy, process_refund
      │   → final operational decision
      ▼
Agent 3 — Comms & Escalation Manager       COMMS_TOOLS
      │   get_escalation_route, send_slack_alert
      ▼
[Customer reply]  +  [Alert in outbox/alerts.jsonl]
```

`TOOL_OWNERSHIP` in `multi_agent_tools.py` encodes this. Agent 3 has **no**
access to `process_refund`; Agent 1 cannot approve money or message anyone. A
test that asserts no agent reached outside its lane is worth writing.

---

## B1. The headline case — `ORD-1005` / `USR-105`

> **Ticket:** "This is Ronen, order ORD-1005. The tablet screen was smashed on
> arrival. Refund me the full 480 dollars, this keeps happening."

This is the scenario the brief asks you to demo on video.

| Stage | Expected |
|---|---|
| Agent 1 — `audit_fraud_risk("ORD-1005", "USR-105")` | `risk_score: 90`, `risk_band: high`, `blocks_automatic_refund: true` |
| Rules that fire | `FR-01` (3 claims/60d) · `FR-02` (address re-routed 2 days pre-delivery) · `FR-04` (1 prior flag) · `FR-05` (score 61) · `FR-08` ($451 refunded/60d) |
| Agent 2 — `check_return_policy` | `ELIGIBLE` on its merits, but `requires_escalation: true` |
| Agent 2 — `process_refund("ORD-1005", 480.0)` | **ESCALATION_REQUIRED**, `approved_amount: 0.0` |
| Agent 3 — `get_escalation_route(risk_band="high", requested_amount=480.0, prior_fraud_flags=1)` | `CH-FRAUD` / `#fraud-security`, severity `critical`, 15-minute SLA |
| Agent 3 — `send_slack_alert(...)` | one line appended to `outbox/alerts.jsonl` |
| Customer reply | Neutral, no accusation of fraud, states the claim is under review. **Never tell the customer they were flagged.** |

The trap: the claim is genuinely eligible. An agent that only reads
`check_return_policy` approves it. Only the fraud report stops the payout.

## B2. New account, high value, "item never arrived" — `ORD-1012` / `USR-109`

> **Ticket:** "I ordered a laptop (ORD-1012), the box arrived but it was empty.
> I need the 890 dollars back."

| | |
|---|---|
| Customer | Account created 2026-07-28 — **8 days old**, first ever order |
| Order | $890, delivered 2026-08-02, item condition `missing`, address changed 2026-08-01 |
| Expected | `risk_score: 60`, `risk_band: high`; fires `FR-02`, `FR-03`, `FR-06`, `FR-07` |
| Route | `CH-FRAUD` |

Note that this one reaches `high` from a *different* combination of rules than
B1. A crew that hard-codes "high risk means repeat claims" gets this wrong.

## B3. Clean case — no escalation at all

`ORD-1001` / `USR-101`: `risk_score: 0`, `risk_band: low`, zero rules fired.
`get_escalation_route` returns `escalation_required: false` and
`channel_id: null`.

**Do not send an alert here.** An over-eager Comms agent that pings
`#fraud-security` on every ticket is a real failure mode, and the outbox is
where a grader will see it.

## B4. Routing table — first match wins

`get_escalation_route` walks channels in ascending `priority` and returns the
first match, so exactly one destination comes back.

| Input | Channel | Why |
|---|---|---|
| `risk_band="high"` | `CH-FRAUD` | priority 1 |
| `prior_fraud_flags=1`, band `low`, $10 | `CH-FRAUD` | a prior flag alone is enough |
| band `low`, $250 | `CH-FINANCE` | large payout, low risk |
| band `medium`, $150 | `CH-SUPPORT-T2` | over the $50 cap, under $250 |
| `verdict="OUTSIDE_RETURN_WINDOW"` | `CH-SUPPORT-T2` | a human should review the rejection |
| `order_status="delayed"`, no refund | `CH-LOGISTICS` | chase the carrier |
| band `low`, $35, clean | — | `escalation_required: false` |

## B5. Loop and consistency traps

| Trap | What should happen |
|---|---|
| `audit_fraud_risk("ORD-1001", "USR-105")` | `USER_ORDER_MISMATCH` — the ticket's claimed customer does not own the order. Escalate; do not retry with a different id. |
| `audit_fraud_risk("ORD-9999")` | `ORDER_NOT_FOUND`. Ask the customer to confirm the number. Do not loop. |
| Agent 2 gets an incomplete report from Agent 1 | Stop on a defined condition and escalate, rather than sending Agent 1 back repeatedly. Set `max_iterations` on your crew. |
| Agent 3 receives `escalation_required: false` | Send the customer reply and **no** alert. |

Your `README.md` has to explain the stop condition you chose and where it is
enforced in code.

---

## Reading back what you sent

```python
import multi_agent_tools as mat
for alert in mat.read_outbox():
    print(alert["channel"], alert["severity"], alert["payload"]["order_id"])
```

`send_slack_alert` writes to `outbox/alerts.jsonl` by default so the whole quest
works with no Slack workspace. For the demo video, set a real incoming webhook:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

The response then reports `transport: "outbox+webhook"` and a
`webhook_status`. The offline record is written either way.

---

## Suggested demo script for the Loom video (3 minutes)

1. **B1 end to end** — show all three agents handing off, the risk report, the
   refund being refused, and the alert landing in `#fraud-security`.
2. **B3** — the same crew resolving a clean ticket with no alert at all.
3. **30 seconds on your guardrails** — where `max_iterations` lives, and why the
   Comms agent cannot reach `process_refund`.
