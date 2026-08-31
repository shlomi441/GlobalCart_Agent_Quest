"""Phase 3: the long-term ledger records runs and recalls them by order or customer."""

import json

import crew  # noqa: F401

from crew.memory import Ledger
from crew.schemas import AgentRun, CommsResult, CrewResult
from conftest import make_decision, make_report


def _result(order_id, run_id="r1", **kw):
    report = make_report(order_id, **kw)
    return CrewResult(ticket=f"ticket about {order_id}", run_id=run_id, risk_report=report, decision=make_decision(report),
                      comms=CommsResult(customer_reply="ok"), tool_log=[], agent_runs=[])


def test_remember_then_recall_by_order_or_customer(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.recall("ORD-1001", "USR-101") == []
    rec = ledger.remember(_result("ORD-1001"))
    assert rec["refund_status"] == "APPROVED" and rec["refund_id"] == "RF-1001-3500" and rec["approved_amount"] == 35.0
    assert "ticket" not in rec and len(rec["ticket_sha"]) == 12          # no ticket text, just a fingerprint
    ledger.remember(_result("ORD-1002", run_id="r2", demanded=150.0))
    assert [r["run_id"] for r in ledger.recall("ORD-1001", None)] == ["r1"]
    assert [r["run_id"] for r in ledger.recall(None, "USR-102")] == ["r2"]
    assert [r["run_id"] for r in ledger.recall("ORD-1006", "USR-101")] == ["r1"]   # same customer, other order
    assert ledger.recall(None, None) == []
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["refund_status"] == "ESCALATION_REQUIRED"


def test_halted_runs_are_remembered_with_their_reason(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    res = _result("ORD-1001", claimed_user_id="USR-105")
    rec = ledger.remember(res.model_copy(update={"halt_reason": "USER_ORDER_MISMATCH"}))
    assert rec["refund_status"] == "NONE" and rec["halt_reason"] == "USER_ORDER_MISMATCH" and rec["user_id"] == "USR-105"
