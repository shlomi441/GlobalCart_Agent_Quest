"""The output contract. The prompt requests this shape; this file enforces it."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal[
    "AUTO_REFUND_APPROVED",
    "REJECTED",
    "ESCALATED_TO_HUMAN",
    "NEEDS_MORE_INFO",
]


class ActionTaken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools_called: list[str]
    decision: Decision
    refund_amount: float = Field(ge=0)
    refund_id: Optional[str] = None

    @model_validator(mode="after")
    def _decision_payload_consistent(self) -> "ActionTaken":
        """A decision and its payload must tell the same story."""
        if self.decision == "AUTO_REFUND_APPROVED":
            if not self.refund_id or self.refund_amount <= 0:
                raise ValueError(
                    "AUTO_REFUND_APPROVED requires a non-null refund_id "
                    "and refund_amount > 0"
                )
        else:
            if self.refund_id is not None or self.refund_amount != 0:
                raise ValueError(
                    f"decision {self.decision} requires refund_amount 0 "
                    "and refund_id null"
                )
        return self


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_chain: list[str] = Field(min_length=1)
    action_taken: ActionTaken
    customer_response: str = Field(min_length=1)