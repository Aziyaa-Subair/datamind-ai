"""Central configuration. Everything environment-specific lives here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "sales.csv"

# Comma-separated list in .env; tried in order if one is rate-limited or down.
GEMINI_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-3.7-flash,gemini-3.5-flash,gemini-3.1-flash-lite").split(",")
    if m.strip()
]


def get_api_key() -> str:
    """Return the Gemini API key or raise a clear error."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Put GEMINI_API_KEY=... in your .env file "
            "(see .env.example)."
        )
    return key
