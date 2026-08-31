#!/usr/bin/env bash
# demo.sh - Quest #04 Part B, the 3-minute demo for macOS / Linux. Run from "Quest 4/Stage 2" with the venv active:
#     chmod +x demo.sh && ./demo.sh
# Identical flow to demo.ps1: no narration, every important point is a caption card held on screen long
# enough to read. Adjust SPEED (1.0 = as written, 1.3 = 30% longer holds) if the recording feels rushed.
# To hide your real path in the recording, set the prompt first:   export PS1='GlobalCart Ops Crew> '
set -o pipefail
export PYTHONUTF8=1
PYTHON="${PYTHON:-python3}"
SPEED="${SPEED:-1.0}"

RONEN="This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars, this keeps happening."
MAYA="Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. I've been shopping with you for years, can you sort this out?"

CYAN=$'\033[36m'; YELLOW=$'\033[33m'; GREEN=$'\033[32m'; MAGENTA=$'\033[35m'; RESET=$'\033[0m'

# card SECONDS COLOR LINE...   - a boxed caption held for SECONDS * SPEED
card() {
    local seconds="$1" color="$2"; shift 2
    local width=0 line
    for line in "$@"; do (( ${#line} > width )) && width=${#line}; done
    width=$((width + 4))
    local rule; rule=$(printf '=%.0s' $(seq 1 "$width"))
    printf '\n%s%s%s\n' "$color" "$rule" "$RESET"
    for line in "$@"; do printf '%s  %s%s\n' "$color" "$line" "$RESET"; done
    printf '%s%s%s\n\n' "$color" "$rule" "$RESET"
    sleep "$(awk -v s="$seconds" -v f="$SPEED" 'BEGIN { printf "%.1f", s * f }')"
}
cmd() { printf '%s> %s%s\n' "$YELLOW" "$1" "$RESET"; }

# ---------------------------------------------------------------- 0. clean slate
clear
rm -f starter-kit/outbox/alerts.jsonl memory/ledger.jsonl
mkdir -p runs

card 9 "$CYAN" \
  "GlobalCart Operations Crew  -  Quest #04, Part B" \
  "" \
  "Three specialist agents on LangGraph, each with its own prompt and its own kit tool bundle:" \
  "  researcher  get_order_details / get_user_profile / audit_fraud_risk" \
  "  decision    check_return_policy / process_refund" \
  "  comms       get_escalation_route / send_slack_alert" \
  "The kit (tools + data) is untouched. The model narrates; the code records."

# ---------------------------------------------------------------- 1. the kit
cmd "$PYTHON starter-kit/examples/verify_scenarios.py"
"$PYTHON" starter-kit/examples/verify_scenarios.py | tail -n 1
sleep 3

# ---------------------------------------------------------------- 2. B1
card 8 "$CYAN" \
  "B1 - the headline case: ORD-1005" \
  "A claim that is eligible on paper (check_return_policy says ELIGIBLE)" \
  "while the fraud engine says 90/100, band high." \
  "" \
  "Watch for:  the decision agent never calls process_refund" \
  "            the alert goes to #fraud-security through the webhook" \
  "            the customer is told only that the request is under review"
cmd "$PYTHON -m crew.run --save runs/demo_b1.txt \"$RONEN\""
"$PYTHON" -m crew.run --save runs/demo_b1.txt "$RONEN"

card 10 "$CYAN" \
  "What just happened" \
  "  researcher : order + owner + fraud engine -> 90/100 high, rules FR-01 FR-02 FR-04 FR-05 FR-08" \
  "  decision   : ESCALATED_TO_HUMAN, attempted=False, blocked_by risk_report:high + policy escalation" \
  "               (had the model tried, the dispatcher refuses: BLOCKED_BY_RISK_REPORT)" \
  "  comms      : route CH-FRAUD critical -> one alert, transport=outbox+webhook" \
  "  reply      : 'under review by our team' - no fraud, no flags, no scores, no rule ids"

cmd "cat starter-kit/outbox/alerts.jsonl   # the offline record of the alert"
"$PYTHON" - <<'PY'
import json
for line in open("starter-kit/outbox/alerts.jsonl", encoding="utf-8"):
    a = json.loads(line)
    print(f"\033[32mchannel={a['channel']}  severity={a['severity']}  message_ts={a['message_ts']}\033[0m")
    print(a["message"])
PY
card 2 "$MAGENTA" "The same alert is now in Slack  ->  switching to #fraud-security" "(press Enter here when back)"
read -r

# ---------------------------------------------------------------- 3. B3
card 6 "$CYAN" \
  "B3 - the clean case: ORD-1001 (VIP, damaged on arrival, 35 USD)" \
  "Same crew, same code path. Expected: approve, and send NO alert."
cmd "$PYTHON -m crew.run --save runs/demo_b3.txt \"$MAYA\""
"$PYTHON" -m crew.run --save runs/demo_b3.txt "$MAYA"

card 8 "$CYAN" \
  "What just happened" \
  "  decision : AUTO_REFUND_APPROVED, refund RF-1001-3500 read from process_refund's own result" \
  "  comms    : router says escalation_required=false -> route None, alert none" \
  "  outbox   : still exactly one line (a comms agent that alerts anyway is blocked: ALERT_NOT_AUTHORIZED)"
cmd "wc -l < starter-kit/outbox/alerts.jsonl"
echo "$(wc -l < starter-kit/outbox/alerts.jsonl | tr -d ' ') line(s) in the outbox"
sleep 3

# ---------------------------------------------------------------- 4. guardrails
card 12 "$CYAN" \
  "Guardrails 1/3 - the stop condition (crew/graph.py, crew/config.py, crew/policy.py)" \
  "  The graph below is generated from the compiled code. There is no edge back to researcher:" \
  "  an incomplete report can only flow forward (triage -> comms with a code-synthesized decision)." \
  "  LangGraph recursion_limit=10 is the tripwire above that; MAX_STEPS=8 per agent; one format retry." \
  "  Every failure has a defined destination: not found -> ask the customer, mismatch -> #fraud-security," \
  "  anything else -> #support-tier2 (policy.halt_plan). Even the tripwire ends in a Tier-2 alert, by code."
cmd "$PYTHON -m crew.run --mermaid"
"$PYTHON" -m crew.run --mermaid
sleep 5

card 12 "$CYAN" \
  "Guardrails 2/3 - financial authority (crew/dispatch.py, crew/graph.py, crew/schemas.py)" \
  "  Refund status is READ from process_refund's result, never reported by the model" \
  "  (graph.build_decision + the Decision validator reject any other story)." \
  "  DecisionDispatcher locks, before the kit ever sees a call:" \
  "    BLOCKED_BY_RISK_REPORT   no payout attempt when the engine's band is high" \
  "    AMOUNT_NOT_MERITED       a 52 USD claim can never be shaved to 50 to fit the cap" \
  "    DUPLICATE_CLAIM          the ledger already holds an approved refund for this order" \
  "  Reply gates: no internal vocabulary; no ids, amounts or timelines the evidence does not support."

card 7 "$CYAN" \
  "Guardrails 3/3 - separation of concerns (crew/dispatch.py, crew/agents.py)" \
  "  Each agent's dispatcher is built from its own kit bundle; a foreign tool is unreachable (OUT_OF_LANE)." \
  "  The comms agent cannot refund - asserted in the tests and right here:"
cmd "$PYTHON -c \"import crew, multi_agent_tools as m; assert 'process_refund' not in {t['name'] for t in m.COMMS_TOOLS}; print('COMMS_TOOLS does not contain process_refund: OK')\""
"$PYTHON" -c "import crew, multi_agent_tools as m; assert 'process_refund' not in {t['name'] for t in m.COMMS_TOOLS}; print('COMMS_TOOLS does not contain process_refund: OK')"
sleep 3

card 7 "$GREEN" \
  "124 offline tests (no API key)  |  18 live scenarios x Sonnet 5 and Haiku 4.5  |  18 incidents documented" \
  "README: architecture and data flow, memory context, guardrails with file::function, compatibility matrix"
