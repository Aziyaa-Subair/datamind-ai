"""Pydantic models shared by the agent, verifier, UI and evals."""

from typing import Any

from pydantic import BaseModel, Field


class UsedValue(BaseModel):
    """A number the agent relied on, copied from a tool result."""

    label: str = Field(description="What the number is, e.g. 'total revenue'")
    value: float = Field(description="The number, copied exactly from a tool output")


class AnalysisResult(BaseModel):
    """What the analysis agent must return. This is the agent's structured output."""

    answer: str = Field(description="Plain-language answer for the user")
    columns_used: list[str] = Field(
        default_factory=list, description="Dataset columns the analysis used"
    )
    values_used: list[UsedValue] = Field(
        default_factory=list,
        description="Every number quoted in the answer, copied exactly from tool outputs",
    )


class ToolCall(BaseModel):
    """One recorded tool invocation (kept so the verifier can audit the agent)."""

    name: str
    args: dict[str, Any]
    result: Any


class VerificationReport(BaseModel):
    passed: bool
    errors: list[str] = Field(default_factory=list)    # claims that contradict tools
    warnings: list[str] = Field(default_factory=list)  # things worth a human look


class VerifiedAnswer(BaseModel):
    """Final object returned to the UI/API: answer + proof."""

    result: AnalysisResult
    report: VerificationReport
    tool_calls: list[ToolCall]
