# demo.ps1 - Quest #04 Part B, the 3-minute demo. Run from "Quest 4\Stage 2" with the venv active.
# No narration: every important point is a caption card held on screen long enough to read.
# Adjust $SPEED (1.0 = as written, 1.3 = 30% longer holds) if the recording feels rushed.
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$SPEED = 1.0

$RONEN = "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars, this keeps happening."
$MAYA  = "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. I've been shopping with you for years, can you sort this out?"

function Card([string[]]$lines, [double]$seconds, [string]$color = "Cyan") {
    $width = ($lines | Measure-Object -Property Length -Maximum).Maximum + 4
    Write-Host ""
    Write-Host ("=" * $width) -ForegroundColor $color
    foreach ($l in $lines) { Write-Host ("  " + $l) -ForegroundColor $color }
    Write-Host ("=" * $width) -ForegroundColor $color
    Write-Host ""
    Start-Sleep -Seconds ([math]::Round($seconds * $SPEED))
}
function Cmd([string]$text) { Write-Host ("> " + $text) -ForegroundColor Yellow }

# ---------------------------------------------------------------- 0. clean slate
Clear-Host
Remove-Item starter-kit\outbox\alerts.jsonl -ErrorAction SilentlyContinue
Remove-Item memory\ledger.jsonl -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force runs | Out-Null

Card @(
  "GlobalCart Operations Crew  -  Quest #04, Part B",
  "",
  "Three specialist agents on LangGraph, each with its own prompt and its own kit tool bundle:",
  "  researcher  get_order_details / get_user_profile / audit_fraud_risk",
  "  decision    check_return_policy / process_refund",
  "  comms       get_escalation_route / send_slack_alert",
  "The kit (tools + data) is untouched. The model narrates; the code records."
) 9

# ---------------------------------------------------------------- 1. the kit
Cmd "python starter-kit\examples\verify_scenarios.py"
python starter-kit\examples\verify_scenarios.py | Select-Object -Last 1
Start-Sleep -Seconds 3

# ---------------------------------------------------------------- 2. B1
Card @(
  "B1 - the headline case: ORD-1005",
  "A claim that is eligible on paper (check_return_policy says ELIGIBLE)",
  "while the fraud engine says 90/100, band high.",
  "",
  "Watch for:  the decision agent never calls process_refund",
  "            the alert goes to #fraud-security through the webhook",
  "            the customer is told only that the request is under review"
) 8
Cmd "python -m crew.run --save runs\demo_b1.txt `"$RONEN`""
python -m crew.run --save runs\demo_b1.txt $RONEN

Card @(
  "What just happened",
  "  researcher : order + owner + fraud engine -> 90/100 high, rules FR-01 FR-02 FR-04 FR-05 FR-08",
  "  decision   : ESCALATED_TO_HUMAN, attempted=False, blocked_by risk_report:high + policy escalation",
  "               (had the model tried, the dispatcher refuses: BLOCKED_BY_RISK_REPORT)",
  "  comms      : route CH-FRAUD critical -> one alert, transport=outbox+webhook",
  "  reply      : 'under review by our team' - no fraud, no flags, no scores, no rule ids"
) 10

Cmd "Get-Content starter-kit\outbox\alerts.jsonl   # the offline record of the alert"
Get-Content -Encoding UTF8 starter-kit\outbox\alerts.jsonl | ForEach-Object {
    $a = $_ | ConvertFrom-Json
    Write-Host ("channel=" + $a.channel + "  severity=" + $a.severity + "  message_ts=" + $a.message_ts) -ForegroundColor Green
    Write-Host $a.message
}
Card @("The same alert is now in Slack  ->  switching to #fraud-security", "(press Enter here when back)") 2 "Magenta"
$null = Read-Host

# ---------------------------------------------------------------- 3. B3
Card @(
  "B3 - the clean case: ORD-1001 (VIP, damaged on arrival, 35 USD)",
  "Same crew, same code path. Expected: approve, and send NO alert."
) 6
Cmd "python -m crew.run --save runs\demo_b3.txt `"$MAYA`""
python -m crew.run --save runs\demo_b3.txt $MAYA

Card @(
  "What just happened",
  "  decision : AUTO_REFUND_APPROVED, refund RF-1001-3500 read from process_refund's own result",
  "  comms    : router says escalation_required=false -> route None, alert none",
  "  outbox   : still exactly one line (a comms agent that alerts anyway is blocked: ALERT_NOT_AUTHORIZED)"
) 8
Cmd "(Get-Content starter-kit\outbox\alerts.jsonl | Measure-Object -Line).Lines"
"$((Get-Content starter-kit\outbox\alerts.jsonl | Measure-Object -Line).Lines) line(s) in the outbox"
Start-Sleep -Seconds 3

# ---------------------------------------------------------------- 4. guardrails
Card @(
  "Guardrails 1/3 - the stop condition (crew/graph.py, crew/config.py, crew/policy.py)",
  "  The graph below is generated from the compiled code. There is no edge back to researcher:",
  "  an incomplete report can only flow forward (triage -> comms with a code-synthesized decision).",
  "  LangGraph recursion_limit=10 is the tripwire above that; MAX_STEPS=8 per agent; one format retry.",
  "  Every failure has a defined destination: not found -> ask the customer, mismatch -> #fraud-security,",
  "  anything else -> #support-tier2 (policy.halt_plan). Even the tripwire ends in a Tier-2 alert, by code."
) 12
Cmd "python -m crew.run --mermaid"
python -m crew.run --mermaid
Start-Sleep -Seconds 5

Card @(
  "Guardrails 2/3 - financial authority (crew/dispatch.py, crew/graph.py, crew/schemas.py)",
  "  Refund status is READ from process_refund's result, never reported by the model",
  "  (graph.build_decision + the Decision validator reject any other story).",
  "  DecisionDispatcher locks, before the kit ever sees a call:",
  "    BLOCKED_BY_RISK_REPORT   no payout attempt when the engine's band is high",
  "    AMOUNT_NOT_MERITED       a 52 USD claim can never be shaved to 50 to fit the cap",
  "    DUPLICATE_CLAIM          the ledger already holds an approved refund for this order",
  "  Reply gates: no internal vocabulary; no ids, amounts or timelines the evidence does not support."
) 12

Card @(
  "Guardrails 3/3 - separation of concerns (crew/dispatch.py, crew/agents.py)",
  "  Each agent's dispatcher is built from its own kit bundle; a foreign tool is unreachable (OUT_OF_LANE).",
  "  The comms agent cannot refund - asserted in the tests and right here:"
) 7
Cmd "python -c `"import crew, multi_agent_tools as m; assert 'process_refund' not in {t['name'] for t in m.COMMS_TOOLS}; print('COMMS_TOOLS does not contain process_refund: OK')`""
python -c "import crew, multi_agent_tools as m; assert 'process_refund' not in {t['name'] for t in m.COMMS_TOOLS}; print('COMMS_TOOLS does not contain process_refund: OK')"
Start-Sleep -Seconds 3

Card @(
  "124 offline tests (no API key)  |  18 live scenarios  |  18 incidents documented",
  "README: architecture and data flow, memory context, guardrails with file::function"
) 7 "Green"
