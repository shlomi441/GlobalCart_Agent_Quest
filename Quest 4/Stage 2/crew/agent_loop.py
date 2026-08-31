"""The agent loop: model turn -> tool dispatch -> repeat, until a validated final answer.

This is Part A's `loop.py` with three things made injectable, so one loop serves
three agents and runs offline:

* the **spec** (prompt, tool bundle, output model, hygiene field) — one per agent;
* the **dispatcher** (dispatch.py) — the only path from a model's tool call to a
  real function; it owns the lane, the guards and the locks;
* the **client** — the real Anthropic SDK, or `tests/fake_client.FakeClient`
  replaying scripted turns with no API key.

Everything the loop learns is returned as data (`AgentRunResult`); the loop never
raises for a model or tool problem. Guards carried over from Part A: MAX_STEPS,
one format retry, the runtime reply-hygiene gate. The `(tool, args)` repeat guard
and UNKNOWN_TOOL containment moved into the dispatcher, where the lane lives.

Two output modes (decision D8 — evaluate on weaker models):
* ``text_json``: the model ends its turn with one JSON object (Part A's way);
* ``tool``: the model ends by calling a pseudo-tool whose input_schema is the
  output model's JSON schema, so the API itself constrains the shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, ValidationError

from crew.config import MAX_FORMAT_RETRIES, MAX_STEPS, MAX_TOKENS, MODEL
from crew.policy import find_leaks
from crew.schemas import AgentName, AgentRun

OutputMode = Literal["text_json", "tool"]


@dataclass
class AgentSpec:
    name: AgentName
    system_prompt: str
    tools: list[dict[str, Any]]                 # the kit bundle for this role, nothing else
    output_model: Type[BaseModel]
    hygiene_field: Optional[str] = None         # output field that must pass the reply-hygiene gate
    output_mode: OutputMode = "text_json"
    max_steps: int = MAX_STEPS
    max_format_retries: int = MAX_FORMAT_RETRIES

    @property
    def finish_tool(self) -> str:
        return f"finish_{self.name}"


@dataclass
class AgentRunResult:
    output: Optional[BaseModel]                 # the validated output model, or None on failure
    error: Optional[str]                        # MAX_STEPS_EXCEEDED | INVALID_OUTPUT | REPLY_HYGIENE_VIOLATION
    detail: Optional[str]
    run: AgentRun
    transcript: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _first_seen(names: list[str]) -> list[str]:
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Assistant content blocks as plain dicts, so transcripts are JSON-serialisable
    whether they came from the SDK (pydantic objects) or from the fake client."""
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    return {k: v for k, v in vars(block).items() if v is not None}


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic emits nested models as `$defs` + `$ref`; tool input schemas are safer inlined."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk({k: v for k, v in defs[name].items() if k != "title"})
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def finish_tool_schema(spec: AgentSpec) -> dict[str, Any]:
    return {
        "name": spec.finish_tool,
        "description": ("Submit your final answer. Call this exactly once, alone, after every "
                        "tool result you need is in. Its input is your complete final report."),
        "input_schema": _inline_refs(spec.output_model.model_json_schema()),
    }


def _compact_errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors())


def _parse_final(spec: AgentSpec, data: Any):
    """Validate a candidate final answer. Returns (model, None) or (None, problem)."""
    if isinstance(data, str):
        raw = data.strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"not valid JSON: {exc}"
    try:
        return spec.output_model.model_validate(data), None
    except ValidationError as exc:
        return None, _compact_errors(exc)


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

def run_agent(spec: AgentSpec, user_message: str, dispatcher, client, model: str = MODEL,
              verbose: bool = False, reply_checks: Optional[list] = None) -> AgentRunResult:
    """Run one agent to completion. `dispatcher` must expose `.call(name, args, step) -> dict`
    and `.executed_tools() -> list[str]`; `client` must expose `.messages.create(...)`.
    `reply_checks` are extra callables `text -> list[problem]` applied to the hygiene field
    (the comms node passes the verified-facts check built from this run's evidence)."""
    tools = list(spec.tools) + ([finish_tool_schema(spec)] if spec.output_mode == "tool" else [])
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    format_retries = 0
    steps = 0
    empty_turns = 0
    retry_details: list[str] = []
    anomalies: list[str] = []

    def finish(output, error, detail):
        claimed = _first_seen([t for t in getattr(output, "tools_called", []) if t != spec.finish_tool])
        executed = dispatcher.executed_tools()
        run = AgentRun(agent=spec.name, steps=steps, format_retries=format_retries, error=error,
                       claimed_tools=claimed, executed_tools=executed,
                       honest=(claimed == executed) if output is not None else None,
                       retry_details=retry_details, anomalies=anomalies)
        return AgentRunResult(output=output, error=error, detail=detail, run=run, transcript=messages)

    def retry_or_fail(err_code: str, problem: str, retry_msg: str, raw: Any):
        nonlocal format_retries
        if format_retries >= spec.max_format_retries:
            return finish(None, err_code, problem)
        format_retries += 1
        retry_details.append(f"{err_code}: {problem}")
        if verbose:
            print(f"[{spec.name}] format retry {format_retries}: {problem}")
        messages.append({"role": "user", "content": retry_msg})
        return None

    def validate_candidate(candidate: Any):
        """Shared by both output modes. Returns an AgentRunResult, or None to keep looping."""
        result, problem = _parse_final(spec, candidate)
        if result is None:
            return retry_or_fail(
                "INVALID_OUTPUT", problem,
                "Your final answer failed validation: " + problem + ". Reply again with ONLY the "
                "corrected final answer - same content, fixed structure. No prose, no markdown fences.",
                candidate)
        if spec.hygiene_field:
            leaks = find_leaks(getattr(result, spec.hygiene_field))
            if leaks:
                return retry_or_fail(
                    "REPLY_HYGIENE_VIOLATION", "customer-facing text contains internal terms: " + "; ".join(leaks),
                    "Your customer reply contains internal terminology (" + "; ".join(leaks) + "). Rewrite it in "
                    "plain customer language - no policy or rule codes, no channel or tool names, and no "
                    "mention of fraud, flags, risk, or the internal reasons a case is reviewed. Keep the "
                    "decision and the facts identical and return the full corrected final answer.",
                    candidate)
            for check in reply_checks or []:
                problems = check(getattr(result, spec.hygiene_field))
                if problems:
                    return retry_or_fail(
                        "REPLY_UNVERIFIED_CLAIM", "customer-facing text states facts the evidence does not support: "
                        + "; ".join(problems),
                        "Your customer reply states things no tool result supports (" + "; ".join(problems) + "). "
                        "Use only the order id, amounts and refund id from the reports; do not state processing "
                        "times or timelines; never say a refund was approved or issued unless the decision is "
                        "AUTO_REFUND_APPROVED (an earlier refund may be cited by its id). Return the full corrected "
                        "final answer.",
                        candidate)
        return finish(result, None, None)

    for step in range(1, spec.max_steps + 1):
        steps = step
        resp = client.messages.create(model=model, max_tokens=MAX_TOKENS, system=spec.system_prompt,
                                      tools=tools, messages=messages)
        content = list(resp.content)
        tool_uses = [b for b in content if getattr(b, "type", None) == "tool_use"]
        if not content:
            # Incident 11 (Haiku 4.5): stop_reason "tool_use" arrived with no content at all. The API rejects
            # empty messages in either role, so the transcript gets an explicit placeholder and the run a note.
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "(empty turn)"}]})
        else:
            messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in content]})
        if resp.stop_reason == "tool_use" and not tool_uses:
            snippet = "".join(getattr(b, "text", "") for b in content)[:160].replace("\n", " ")
            anomalies.append(f"turn {step}: stop_reason=tool_use but blocks were "
                             f"{[getattr(b, 'type', '?') for b in content] or 'empty'}; text={snippet!r}")

        if tool_uses:
            finals = [b for b in tool_uses if b.name == spec.finish_tool]
            if spec.output_mode == "tool" and finals and len(tool_uses) == 1:
                outcome = validate_candidate(dict(finals[0].input))
                if outcome is not None:
                    return outcome
                # validate_candidate queued a plain retry message; a tool call must be answered with a
                # tool_result, so fold the guidance into that result and keep role alternation intact.
                guidance = messages.pop()["content"]
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": finals[0].id,
                                 "content": json.dumps({"error": "INVALID_OUTPUT", "message": guidance})}]})
                continue
            results = []
            for block in tool_uses:
                args = dict(block.input)
                if block.name == spec.finish_tool:
                    out = {"error": "FINISH_NOT_ALONE",
                           "message": "Call the finish tool alone, after all other tool results are in."}
                else:
                    out = dispatcher.call(block.name, args, step)
                if verbose:
                    print(f"[{spec.name} step {step}] {block.name}({json.dumps(args, ensure_ascii=False)})"
                          f" -> {json.dumps(out, ensure_ascii=False)[:160]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
            continue

        text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
        if not text.strip():
            empty_turns += 1
            if empty_turns <= 1:   # one nudge, no retry budget spent; a second empty turn counts as invalid output
                messages.append({"role": "user", "content": "Your last turn contained no tool call and no final "
                                 "answer. Either call one of your tools now or give your complete final answer."})
                continue
        if spec.output_mode == "tool":
            outcome = retry_or_fail(
                "INVALID_OUTPUT", "final answer was given as text instead of the finish tool",
                f"Do not answer in text. Call the `{spec.finish_tool}` tool with your complete final answer.", text)
        else:
            outcome = validate_candidate(text)
        if outcome is not None:
            return outcome

    return finish(None, "MAX_STEPS_EXCEEDED", f"No final answer after {spec.max_steps} model turns.")
