"""Pure scoring logic for eval cases. No AI or API dependencies, so it is unit-tested.

A case may use these fields (all optional except question/expected):
  expected             numbers that must appear in the agent's values_used
  expected_tool        a tool name or list of acceptable tool names
  forbidden_values     numbers the agent must NOT report (it would have had to
                       do arithmetic itself - the "LLM never does math" rule)
  unanswerable_mention a word the answer must mention while saying the data
                       does not contain it (heuristic text check)
"""

import math
import re

# "not", "no", "cannot", "unable", "missing", ... or any "...n't" contraction.
_NEGATION = re.compile(r"\b(not|no|cannot|unable|missing|unavailable|without)\b|n't")


def _has(values: list[float], target: float) -> bool:
    return any(math.isclose(v, target, abs_tol=0.01) for v in values)


def score_case(
    case: dict,
    answer_text: str,
    values: list[float],
    tools: list[str],
    verifier_passed: bool,
) -> dict:
    values_ok = all(_has(values, e) for e in case.get("expected", []))

    expected_tool = case.get("expected_tool")
    if expected_tool:
        allowed = expected_tool if isinstance(expected_tool, list) else [expected_tool]
        tool_ok = bool(set(tools) & set(allowed))
    else:
        tool_ok = True

    forbidden_ok = not any(_has(values, f) for f in case.get("forbidden_values", []))

    mention = case.get("unanswerable_mention")
    if mention:
        text = answer_text.lower()
        mention_ok = mention.lower() in text and bool(_NEGATION.search(text))
    else:
        mention_ok = True

    passed = values_ok and tool_ok and forbidden_ok and mention_ok and verifier_passed
    return {
        "passed": passed,
        "values_ok": values_ok,
        "tool_ok": tool_ok,
        "forbidden_ok": forbidden_ok,
        "mention_ok": mention_ok,
        "verifier_ok": verifier_passed,
    }
