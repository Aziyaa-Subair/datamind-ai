"""Tests that the agent, tools, retries and verifier are wired together.

Uses a scripted fake model (no API key, no network) so a *known* sequence of
tool calls is replayed. The pipeline seeds the dataset profile as tool call 0,
so tests on the model's own calls skip it via agent_calls().
"""

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from app import config  # noqa: E402
from app.agents.analysis_agent import build_agent  # noqa: E402
from app.pipeline import answer_question  # noqa: E402
from app.services.dataset_service import load_dataset  # noqa: E402

FINAL_ARGS = {
    "answer": "Total revenue is 37,950.",
    "columns_used": ["Revenue"],
    "values_used": [{"label": "total revenue", "value": 37950.0}],
}


def _kinds(messages) -> list[str]:
    return [getattr(p, "part_kind", "") for m in messages for p in m.parts]


def good_script(messages, info: AgentInfo) -> ModelResponse:
    if "tool-return" not in _kinds(messages):
        return ModelResponse(parts=[ToolCallPart("get_column_total", {"column": "Revenue"})])
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, FINAL_ARGS)])


def typo_then_fix_script(messages, info: AgentInfo) -> ModelResponse:
    kinds = _kinds(messages)
    if "tool-return" in kinds:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, FINAL_ARGS)])
    column = "Revenue" if "retry-prompt" in kinds else "Revenu"  # first call has a typo
    return ModelResponse(parts=[ToolCallPart("get_column_total", {"column": column})])


def run(script):
    df = load_dataset(config.DEFAULT_DATASET)
    agent = build_agent(model=FunctionModel(script))
    return answer_question(df, "What is the total revenue?", agent=agent)


def agent_calls(verified):
    """Tool calls made by the model. The pipeline seeds the dataset profile
    as call 0, so skip it."""
    return verified.tool_calls[1:]


def test_profile_is_seeded_as_first_call():
    verified = run(good_script)
    assert verified.tool_calls[0].name == "get_dataset_profile"


def test_agent_calls_tool_and_passes_verification():
    verified = run(good_script)
    calls = agent_calls(verified)
    assert [c.name for c in calls] == ["get_column_total"]
    assert calls[0].result == 37950.0
    assert verified.report.passed


def test_bad_column_triggers_retry_and_recovers():
    verified = run(typo_then_fix_script)
    # Only the successful call is recorded; the typo was bounced back to the model.
    assert [c.args for c in agent_calls(verified)] == [{"column": "Revenue"}]
    assert verified.report.passed