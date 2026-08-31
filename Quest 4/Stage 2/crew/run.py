"""CLI: run the crew on one ticket.

    python -m crew.run "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me the full 480 dollars."
    python -m crew.run --model claude-haiku-4-5 "<ticket>"
    python -m crew.run --json "<ticket>"        # the full CrewResult as JSON
    python -m crew.run --mermaid                # the graph, as the README will show it

Prints an audit view first (who ran, what they called, what stopped them, what
was decided, what was sent) and then the customer reply.
"""

from __future__ import annotations

import argparse
import json
import sys

from crew.agents import build_specs, live_client
from crew.config import MODEL
from crew.graph import Crew
from crew.schemas import CrewResult

DEFAULT_TICKET = ("This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. "
                  "Refund me the full 480 dollars, this keeps happening.")


def audit_view(result: CrewResult) -> str:
    lines = [f"run {result.run_id}  halt={result.halt_reason}", ""]
    for run in result.agent_runs:
        calls = [c for c in result.tool_log if c.agent == run.agent]
        lines.append(f"[{run.agent}] steps={run.steps} retries={run.format_retries} error={run.error} honest={run.honest}")
        for c in calls:
            tag = "crew" if c.step == 0 else f"s{c.step}"
            status = c.result.get("error") or c.result.get("status") or c.result.get("verdict") \
                or c.result.get("channel_id") or ("delivered" if c.result.get("delivered") else "ok")
            lines.append(f"    {tag:>4} {c.tool}({json.dumps(c.args, ensure_ascii=False)[:90]}) -> {status}{' [blocked]' if c.synthetic else ''}")
    r, d, m = result.risk_report, result.decision, result.comms
    lines += ["", f"risk report : {r.status} " + (f"{r.fraud_audit.risk_score}/100 {r.fraud_audit.risk_band} "
              f"{[x.rule_id for x in r.fraud_audit.triggered_rules]}" if r.fraud_audit else f"({r.error_code})"),
              f"decision    : {d.decision} refund_status={d.refund_status} attempted={d.refund_attempted} "
              f"blocked_by={d.blocked_by} merited={d.merited_amount} claimed={d.claimed_decision}",
              f"route       : {m.route.channel_id if m.route else None}"
              + (f" ({m.route.override_reason})" if m.route and m.route.override_reason else ""),
              f"alert       : " + (f"{m.alert.channel_id} {m.alert.severity} ts={m.alert.message_ts} transport={m.alert.transport}"
                                   if m.alert else "none") + (" [crew fallback]" if m.fallback_used else ""),
              "", "customer reply:", m.customer_reply]
    if result.notes:
        lines += ["", "notes:"] + [f"  - {n}" for n in result.notes]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the GlobalCart operations crew on one ticket.")
    parser.add_argument("ticket", nargs="*", help="the customer ticket text")
    parser.add_argument("--mode", choices=["text_json", "tool"], default=None, help="final-answer mode (default: OUTPUT_MODE env)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--json", action="store_true", help="print the full CrewResult as JSON")
    parser.add_argument("--verbose", action="store_true", help="print every tool call as it happens")
    parser.add_argument("--no-remember", action="store_true", help="do not write this run to the ledger")
    parser.add_argument("--mermaid", action="store_true", help="print the graph diagram and exit")
    parser.add_argument("--save", metavar="PATH", help="write the audit view to PATH and the full JSON to PATH.json (UTF-8)")
    args = parser.parse_args(argv)

    specs = build_specs(args.mode) if args.mode else build_specs()
    if args.mermaid:
        print(Crew(specs, client=None).mermaid())
        return 0
    ticket = " ".join(args.ticket).strip() or DEFAULT_TICKET
    crew = Crew(specs, live_client(), model=args.model, verbose=args.verbose)
    result = crew.run(ticket, remember=not args.no_remember)
    print(audit_view(result))
    if args.save:
        from pathlib import Path
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(audit_view(result) + "\n", encoding="utf-8")
        out.with_suffix(out.suffix + ".json").write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved {out} and {out.with_suffix(out.suffix + '.json')}")
    if args.json:
        print()
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
