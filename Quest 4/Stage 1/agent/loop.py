"""The agent loop: model turn -> tool dispatch -> repeat, until a final answer."""

import json
import mock_services as gc
import re
from anthropic import Anthropic
from agent.config import MAX_FORMAT_RETRIES, MAX_STEPS, MAX_TOKENS, MODEL  # runs the sys.path bootstrap
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import AgentResult
from pydantic import ValidationError
from collections import Counter


client = Anthropic()

# Reply hygiene gate: mechanically checkable, therefore enforced in code.
BANNED_IN_REPLY = re.compile(r"POL-[A-Z]+-\d+|\bfraud\w*\b|\bflag(?:ged|ging|s)?\b", re.IGNORECASE)

def _execute_tool(name: str, args: dict) -> dict:
    """Run a tool, converting every possible failure into the kit's error-dict shape."""
    fn = gc.TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": "UNKNOWN_TOOL", "message": f"No tool named '{name}'."}
    try:
        return fn(**args)
    except Exception as exc:  # e.g. TypeError from a malformed argument
        return {"error": "TOOL_EXECUTION_ERROR", "message": f"{type(exc).__name__}: {exc}"}


def _compact_errors(exc: ValidationError) -> str:
    """Flatten a pydantic error into one short, model-readable line."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
    )


def _try_validate(text: str):
    """Return (AgentResult, None) on success, (None, problem) on any failure."""
    raw = text.strip()
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"
    try:
        return AgentResult.model_validate(data), None
    except ValidationError as exc:
        return None, _compact_errors(exc)


def run_agent(ticket: str, verbose: bool = True) -> dict:
    """Resolve one support ticket. Returns output + the loop's own audit trail."""
    messages = [{"role": "user", "content": ticket}]
    tools_called: list[str] = []
    format_retries = 0
    call_counts: Counter = Counter()

    for step in range(1, MAX_STEPS + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=gc.TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "tool_use":
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                key = (block.name, json.dumps(dict(block.input), sort_keys=True))
                call_counts[key] += 1
                if call_counts[key] > 2:
                    out = {"error": "REPEATED_CALL",
                           "message": ("This exact call was already made twice. "
                                       "Reuse the earlier result instead of calling again.")}
                else:
                    out = _execute_tool(block.name, dict(block.input))
                tools_called.append(block.name)
                if verbose:
                    print(f"[step {step}] {block.name}({json.dumps(block.input)})")
                    print(f"        -> {json.dumps(out, ensure_ascii=False)[:200]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out),
                })
            messages.append({"role": "user", "content": results})
            continue

        text = "".join(b.text for b in resp.content if b.type == "text")
        result, problem = _try_validate(text)
        if result is not None:
            leaks = sorted({m.group(0) for m in BANNED_IN_REPLY.finditer(result.customer_response)})
            if not leaks:
                return {"output": result.model_dump(), "tools_called": tools_called,
                        "transcript": messages}
            problem = "customer_response contains internal terms: " + ", ".join(leaks)
            retry_msg = ("Your customer_response contains internal terminology ("
                         + ", ".join(leaks) + "). Rewrite the reply in plain customer "
                         "language - no policy codes and no mention of fraud, flags, or "
                         "the internal reasons a case is reviewed. Keep the decision and "
                         "facts identical and return the full corrected JSON object.")
            err_code = "REPLY_HYGIENE_VIOLATION"
        else:
            retry_msg = ("Your final answer failed validation: " + problem +
                         ". Reply again with ONLY the corrected JSON object - same "
                         "content, fixed structure. No prose, no markdown fences.")
            err_code = "INVALID_OUTPUT"
        if format_retries >= MAX_FORMAT_RETRIES:
            return {"output": {"error": err_code, "message": problem, "raw": text},
                    "tools_called": tools_called, "transcript": messages}
        format_retries += 1
        if verbose:
            print(f"[format retry {format_retries}] {problem}")
        messages.append({"role": "user", "content": retry_msg})

    return {"output": {"error": "MAX_STEPS_EXCEEDED",
                       "message": f"No final answer after {MAX_STEPS} model turns."},
            "tools_called": tools_called, "transcript": messages}