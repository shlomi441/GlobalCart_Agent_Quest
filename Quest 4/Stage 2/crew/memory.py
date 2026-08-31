"""Long-term memory: a ledger of past runs (the guide's third memory type — bonus).

Working memory lives inside each agent's loop; handoff memory is the graph
state; this is what survives between runs. It is deliberately minimal — one
JSON line per run — and it has teeth: `triage` reads it into `prior_cases`, and
`DecisionDispatcher` refuses to pay an order the ledger already shows as
refunded (`DUPLICATE_CLAIM`). That is cross-run idempotency at the crew layer,
which Part A's README argued belongs outside the model.

Only facts already present in the CrewResult are written; the record carries no
ticket text and no PII beyond the ids the kit itself uses.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from crew.config import LEDGER_PATH
from crew.schemas import CrewResult


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else LEDGER_PATH

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def recall(self, order_id: Optional[str], user_id: Optional[str]) -> list[dict[str, Any]]:
        """Earlier runs about this order or this customer, oldest first."""
        if not order_id and not user_id:
            return []
        return [r for r in self.all()
                if (order_id and r.get("order_id") == order_id) or (user_id and r.get("user_id") == user_id)]

    def remember(self, result: CrewResult) -> dict[str, Any]:
        refund, alert = result.decision.refund, result.comms.alert
        approved = refund is not None and refund.status == "APPROVED"
        record = {
            "run_id": result.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticket_sha": hashlib.sha256(result.ticket.encode("utf-8")).hexdigest()[:12],
            "order_id": result.decision.order_id,
            "user_id": result.decision.user_id,
            "decision": result.decision.decision,
            "refund_status": result.decision.refund_status,
            "refund_id": refund.refund_id if approved else None,
            "approved_amount": refund.approved_amount if approved else 0.0,
            "channel_id": alert.channel_id if alert else None,
            "message_ts": alert.message_ts if alert else None,
            "halt_reason": result.halt_reason,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
