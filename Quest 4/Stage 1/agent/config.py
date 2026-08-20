"""Configuration and environment bootstrap. Import this before mock_services."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

MODEL = os.getenv("MODEL", "claude-sonnet-5")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_FORMAT_RETRIES = int(os.getenv("MAX_FORMAT_RETRIES", "1"))
MAX_TOKENS = 2048

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")