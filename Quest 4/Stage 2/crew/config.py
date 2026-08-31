"""Configuration and environment bootstrap.

Differences from Part A's config.py, both driven by decision D8 (model-agnostic):

* Importing this module never fails. The API key is checked only when a live
  client is actually built (`require_api_key()`), so the offline test suite and
  the scripted fake model run with no key at all.
* Nothing here assumes a model family. `MODEL` is a default, not a pin; the
  README will publish a per-model compatibility matrix instead.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from crew import STAGE_ROOT

load_dotenv(STAGE_ROOT / ".env")

# --- model & loop tunables ---------------------------------------------------
MODEL = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))          # model turns per agent
MAX_FORMAT_RETRIES = int(os.getenv("MAX_FORMAT_RETRIES", "1"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "10"))  # LangGraph node executions per run (tripwire)
REPEAT_CALL_LIMIT = 2                                    # identical (tool, args) calls tolerated per agent
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "text_json")      # "text_json" (Part A's way) or "tool" (finish_<agent> pseudo-tool)
if OUTPUT_MODE not in ("text_json", "tool"):
    raise SystemExit(f"OUTPUT_MODE must be 'text_json' or 'tool', not {OUTPUT_MODE!r}")

# --- paths ---------------------------------------------------------------------
LEDGER_PATH = Path(os.getenv("CREW_LEDGER_PATH", STAGE_ROOT / "memory" / "ledger.jsonl"))


def require_api_key() -> str:
    """Fail loudly, but only at the moment a real model call is about to happen."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key
