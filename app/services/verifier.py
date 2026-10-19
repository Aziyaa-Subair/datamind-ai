"""Deterministic verifier: audits an agent answer against real tool outputs.

No LLM is involved. The check is plain Python, so it cannot be fooled by a
fluent but wrong explanation. Example: the tool returned 37,950 but the agent
wrote 39,750 -> this module reports an error.

Two levels:
  * errors   - a number the agent declared in `values_used` was never
               produced by any tool (hard failure).
  * warnings - a number in the answer text that no tool produced
               (soft: could be a year, a count, or a hallucination).
"""

import math
import re
from typing import Any

from app.schemas import AnalysisResult, ToolCall, VerificationReport

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def collect_numbers(obj: Any) -> list[float]:
    """Recursively gather every numeric value inside a tool result."""
    found: list[float] = []
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            found.extend(collect_numbers(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(collect_numbers(value))
    return found


def _matches_exact(value: float, tool_numbers: list[float]) -> bool:
    return any(math.isclose(value, t, rel_tol=1e-6, abs_tol=0.01) for t in tool_numbers)


def _matches_as_written(text_token: str, tool_numbers: list[float]) -> bool:
    """True if some tool number equals the written number at its written precision."""
    cleaned = text_token.replace(",", "")
    decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
    written = float(cleaned)
    return any(
        round(t, decimals) == written or math.isclose(t, written, rel_tol=1e-6)
        for t in tool_numbers
    )


def verify(result: AnalysisResult, tool_calls: list[ToolCall]) -> VerificationReport:
    errors: list[str] = []
    warnings: list[str] = []

    tool_numbers: list[float] = []
    for call in tool_calls:
        tool_numbers.extend(collect_numbers(call.result))

    # 1. Declared values must come from tools.
    if result.values_used and not tool_calls:
        errors.append("The answer declares numbers but no tool was called.")
    for used in result.values_used:
        if tool_calls and not _matches_exact(used.value, tool_numbers):
            errors.append(
                f"'{used.label}' = {used.value:g} does not match any tool output."
            )

    # 2. Numbers written in the answer text should come from tools too.
    for token in _NUMBER_RE.findall(result.answer):
        value = float(token.replace(",", ""))
        looks_like_year = value.is_integer() and 1900 <= value <= 2100
        is_small_count = value.is_integer() and value < 10
        if looks_like_year or is_small_count:
            continue
        if not _matches_as_written(token, tool_numbers):
            warnings.append(f"Number '{token}' in the answer text is not in any tool output.")

    if not tool_calls:
        warnings.append("No tools were called; the answer is not backed by computed data.")

    return VerificationReport(passed=not errors, errors=errors, warnings=warnings)
