"""A scripted stand-in for the Anthropic client, so the loop and the graph run offline.

A script is a list of turns. A turn is either a list of (tool_name, args) pairs —
the model calls tools — or a string — the model's final text. The response
objects mimic exactly the attributes the loop reads (`content` blocks with
`.type/.name/.input/.id/.text`, and `stop_reason`). Every request is recorded in
`.requests`, which lets a test assert what the model was *offered*: the tools
list at the API boundary is the lane made visible.
"""

import json
from types import SimpleNamespace


def final(obj) -> str:
    """A final-answer turn: the JSON the model would type."""
    return json.dumps(obj, ensure_ascii=False)


class OddTurn:
    """A raw response for edge cases: any stop_reason with any blocks (e.g. Haiku's empty 'tool_use' turn)."""

    def __init__(self, stop_reason, blocks=()):
        self.stop_reason, self.blocks = stop_reason, list(blocks)


EMPTY_TOOL_TURN = OddTurn("tool_use")


class FakeClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self._n = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.turns:
            raise AssertionError("fake model script exhausted — the loop asked for more turns than scripted")
        turn = self.turns.pop(0)
        if isinstance(turn, OddTurn):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=t) for t in turn.blocks], stop_reason=turn.stop_reason)
        if isinstance(turn, str):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=turn)], stop_reason="end_turn")
        blocks = []
        for name, args in turn:
            self._n += 1
            blocks.append(SimpleNamespace(type="tool_use", id=f"toolu_{self._n}", name=name, input=dict(args)))
        return SimpleNamespace(content=blocks, stop_reason="tool_use")
