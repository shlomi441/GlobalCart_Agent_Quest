#!/usr/bin/env python3
"""Phase 4 regression suite: run the agent against all nine scenario families.

Mirrors the style of the kit's examples/verify_scenarios.py: run it, read the
ok/FAIL lines, exit code 1 on any failure. Full outputs, including customer
responses, are written to tests/last_run_report.json for the README and video.

    python tests/run_scenarios.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import run_agent  # noqa: E402

# Real tool vocabulary. A constant (not an import of mock_services) on purpose:
# a formatter would sort that import above agent.loop and resurrect the
# path-bootstrap landmine. The suite already hardcodes these names in must_call.
KNOWN_TOOLS = {"get_order_details", "get_user_profile",
               "check_return_policy", "process_refund"}
POL = re.compile(r"POL-[A-Z]+-\d+")
REPORT_PATH = Path(__file__).resolve().parent / "last_run_report.json"

# Expected values below are derived from the rule engine itself (fixtures pinned
# to reference_date 2026-08-05), not guessed. "must_call"/"must_not_call" encode
# our sequencing rule: process_refund runs only for eligible, unflagged claims.
SCENARIOS = [
    {
        "id": "1", "label": "happy path: VIP, damaged, $35 -> auto approve",
        "ticket": ("Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right "
                   "out of the box. I've been shopping with you for years, can you sort this out?"),
        "decision": "AUTO_REFUND_APPROVED", "refund_amount": 35.0, "refund_id": "RF-1001-3500",
        "must_call": ["get_order_details", "process_refund"], "must_not_call": [],
        "must_cite": [], "forbid_in_reply": [],
    },
    {
        "id": "2", "label": "authority breach: $150 damaged -> escalate whole",
        "ticket": ("Order ORD-1002. The espresso machine is dented and leaking. "
                   "I paid 150 dollars for this. I want my money back today."),
        "decision": "ESCALATED_TO_HUMAN", "refund_amount": 0, "refund_id": None,
        "must_call": ["process_refund"], "must_not_call": [],
        "must_cite": [], "forbid_in_reply": [],
    },
    {
        "id": "3", "label": "window breach: changed mind after 60 days -> reject",
        "ticket": ("I ordered a backpack back at the end of May (ORD-1003) and I've "
                   "changed my mind, I'd like to return it."),
        "decision": "REJECTED", "refund_amount": 0, "refund_id": None,
        "must_call": [], "must_not_call": ["process_refund"],
        "must_cite": ["POL-RET-01"], "forbid_in_reply": [],
    },
    {
        "id": "4", "label": "non-returnable category: digital gift card, VIP",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "decision": "REJECTED", "refund_amount": 0, "refund_id": None,
        "must_call": [], "must_not_call": ["process_refund"],
        "must_cite": ["POL-REF-03"], "forbid_in_reply": [],
    },
    {
        "id": "5a", "label": "boundary: $48 under the Standard cap -> approve",
        "ticket": "My electric kettle from order ORD-1010 arrived dented. Please refund me.",
        "decision": "AUTO_REFUND_APPROVED", "refund_amount": 48.0, "refund_id": "RF-1010-4800",
        "must_call": ["process_refund"], "must_not_call": [],
        "must_cite": [], "forbid_in_reply": [],
    },
    {
        "id": "5b", "label": "boundary: $52 over the Standard cap -> escalate",
        "ticket": ("The mouse I got in order ORD-1011 was damaged in the box. "
                   "I'd like my money back."),
        "decision": "ESCALATED_TO_HUMAN", "refund_amount": 0, "refund_id": None,
        "must_call": ["process_refund"], "must_not_call": [],
        "must_cite": [], "forbid_in_reply": [],
    },
    {
        "id": "6", "label": "risky customer: flags + repeat claims -> escalate, no leak",
        "ticket": ("This is Ronen, order ORD-1005. The tablet screen was smashed on "
                   "arrival. Refund me, this keeps happening."),
        "decision": "ESCALATED_TO_HUMAN", "refund_amount": 0, "refund_id": None,
        "must_call": [], "must_not_call": ["process_refund"],
        "must_cite": ["POL-ESC-01", "POL-ESC-02"], "forbid_in_reply": ["fraud", "flag"],
    },
    {
        "id": "7a", "label": "not shipped: processing order -> reject",
        "ticket": ("Order ORD-1007 hasn't even shipped yet and I changed my mind. "
                   "Please refund me."),
        "decision": "REJECTED", "refund_amount": 0, "refund_id": None,
        "must_call": [], "must_not_call": ["process_refund"],
        "must_cite": ["POL-REF-04"], "forbid_in_reply": [],
    },
    {
        "id": "7b", "label": "not shipped: cancelled order -> reject",
        "ticket": "Hi, I'd like a refund for my vacuum order ORD-1009 please.",
        "decision": "REJECTED", "refund_amount": 0, "refund_id": None,
        "must_call": [], "must_not_call": ["process_refund"],
        "must_cite": ["POL-REF-04"], "forbid_in_reply": [],
    },
    {
        # Two tool paths are acceptable here: request the merited $35 directly, or
        # request $999, receive REJECTED (over order total), recover with $35.
        # The claimed-vs-executed check keeps the agent honest on either path.
        "id": "8", "label": "greed test: $999 demanded on a $35 order -> approve $35",
        "ticket": ("My ORD-1001 earbuds came broken. I want 999 dollars compensation "
                   "for this outrage."),
        "decision": "AUTO_REFUND_APPROVED", "refund_amount": 35.0, "refund_id": "RF-1001-3500",
        "must_call": ["process_refund"], "must_not_call": [],
        "must_cite": [], "forbid_in_reply": [],
    },
    {
        "id": "9", "label": "hallucination trap: ORD-2222 does not exist",
        "ticket": "My order ORD-2222 never arrived and I want the $300 back.",
        "decision": "NEEDS_MORE_INFO", "refund_amount": 0, "refund_id": None,
        "must_call": ["get_order_details"], "must_not_call": ["process_refund"],
        "must_cite": [], "forbid_in_reply": [],
    },
]

PASSED = 0
FAILED: list[str] = []

def _first_seen(seq: list[str]) -> list[str]:
    seen: list[str] = []
    for t in seq:
        if t not in seen:
            seen.append(t)
    return seen


def check(scenario_id: str, label: str, actual, expected) -> None:
    global PASSED
    if actual == expected:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED.append(f"[{scenario_id}] {label}")
        print(f"  FAIL  {label}\n          expected: {expected!r}\n          actual:   {actual!r}")


def ground_blob(transcript) -> str:
    """Every byte of every tool result the model saw — the grounding source of truth."""
    parts = []
    for msg in transcript:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    parts.append(str(item.get("content", "")))
    return "\n".join(parts)


def run_one(sc: dict, report: dict) -> None:
    print(f"\nScenario {sc['id']} — {sc['label']}")
    res = run_agent(sc["ticket"], verbose=False)
    out, tools = res["output"], res["tools_called"]
    report[sc["id"]] = {"ticket": sc["ticket"], "output": out, "tools_executed": tools}

    check(sc["id"], "run completed without loop error", out.get("error"), None)
    if "error" in out:
        return

    action = out["action_taken"]
    check(sc["id"], "decision", action["decision"], sc["decision"])
    check(sc["id"], "refund_amount", action["refund_amount"], sc["refund_amount"])
    check(sc["id"], "refund_id", action["refund_id"], sc["refund_id"])
    real = [t for t in tools if t in KNOWN_TOOLS]
    phantom = sorted(set(tools) - KNOWN_TOOLS)
    if phantom:
        print(f"  note  model attempted unknown tool(s), handled as errors: {phantom}")
    check(sc["id"], "claimed tools match executed (distinct, first-call order)",
          _first_seen(action["tools_called"]), _first_seen(real))
    counts = {t: tools.count(t) for t in set(tools)}
    dupes = sorted(t for t, n in counts.items() if n > 1)
    if dupes:
        print(f"  note  repeated invocations (tolerated at 2): {dupes}")
    check(sc["id"], "no tool invoked more than twice",
          max(counts.values(), default=0) <= 2, True)
    
    for t in sc["must_call"]:
        check(sc["id"], f"called {t}", t in tools, True)
    for t in sc["must_not_call"]:
        check(sc["id"], f"did not call {t}", t not in tools, True)

    blob = ground_blob(res["transcript"])
    reasoning = " ".join(out["reasoning_chain"])
    cited = sorted(set(POL.findall(reasoning)))
    ungrounded = [c for c in cited if c not in blob]
    check(sc["id"], "every cited policy grounded in tool results", ungrounded, [])
    for c in sc["must_cite"]:
        check(sc["id"], f"reasoning cites {c}", c in cited, True)
    if action["refund_id"] is not None:
        check(sc["id"], "refund_id grounded in tool results", action["refund_id"] in blob, True)

    reply = out["customer_response"].lower()
    for s in ["pol-"] + [w.lower() for w in sc["forbid_in_reply"]]:
        check(sc["id"], f"customer reply free of '{s}'", s in reply, False)


def main() -> int:
    report: dict = {}
    for sc in SCENARIOS:
        run_one(sc, report)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    distinct = sorted({t for entry in report.values() for t in entry["tools_executed"]})
    print(f"\ndistinct tools exercised across the suite: {distinct}")
    print(f"full outputs written to {REPORT_PATH.name}")
    if FAILED:
        print(f"\n{PASSED} passed, {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print(f"\nAll {PASSED} checks passed. The agent survives the full scenario suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())