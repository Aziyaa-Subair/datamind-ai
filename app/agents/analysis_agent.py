"""The DataMind analysis agent: an LLM that chooses tools; Python does the math."""

from dataclasses import dataclass, field
from typing import Any, Callable
import json
import pandas as pd
from pydantic_ai import Agent, ModelRetry, RunContext

from app import config
from app.schemas import AnalysisResult, ToolCall
from app.services import analysis_tools as tools
from app.services.dataset_service import calculate_numeric_summary, profile_dataset

INSTRUCTIONS = """\
You are DataMind AI, a data analysis assistant.

Rules:
1. Never calculate, estimate or guess numbers yourself. Always call a tool.
2. 2. The dataset profile is provided below. Use it to pick the right columns.
3. Every number in your answer, including simple counts such as the number of
   rows, must also be listed in `values_used` with a short label, copied
   exactly from a tool result or from the dataset profile. Never write a
   number in `answer` that is not in `values_used`. Also list the columns
   you used.
4. If a tool returns an error, read it and retry with a correct column name.
5. If the question asks about something the dataset does not contain (a
   column, a measure, a category such as profit or region), say so in the
   first sentence, for example "The dataset has no profit column." Then stop.
   Do not offer any other figure unless you first call a tool to get it, and
   never describe one column's value as another measure.
6. Never write a number that did not come from a tool result or the dataset
   profile in this conversation. If you did not call a tool, give no figures.
"""


@dataclass
class Deps:
    """Per-request dependencies. The DataFrame is injected, not global, so the
    same agent works for any uploaded file. `trace` records every tool call
    so the verifier can audit the answer afterwards."""

    df: pd.DataFrame
    trace: list[ToolCall] = field(default_factory=list)


def _run_tool(
    ctx: RunContext[Deps], name: str, args: dict[str, Any], fn: Callable[[], Any]
) -> Any:
    """Run a deterministic function, record it, and turn errors into retries."""
    try:
        result = fn()
    except tools.AnalysisError as exc:
        # ModelRetry sends the message back to the LLM so it can fix its call.
        raise ModelRetry(str(exc)) from exc
    ctx.deps.trace.append(ToolCall(name=name, args=args, result=result))
    return result


def build_agent(model: Any = None) -> Agent[Deps, AnalysisResult]:
    """Create the agent. `model` can be injected (used by tests); by default
    a Gemini model is built from .env settings. Built lazily so importing this
    module never needs an API key."""
    if model is None:
        from pydantic_ai.models.fallback import FallbackModel
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key=config.get_api_key())
        models = [GoogleModel(name, provider=provider) for name in config.GEMINI_MODELS]
        model = models[0] if len(models) == 1 else FallbackModel(*models)

    agent: Agent[Deps, AnalysisResult] = Agent(
        model,
        deps_type=Deps,
        output_type=AnalysisResult,
        instructions=INSTRUCTIONS,
        retries=3,
    )
    @agent.instructions
    def dataset_context(ctx: RunContext[Deps]) -> str:
        return (
            "The dataset is already loaded. Its profile (computed by Python):\n"
            + json.dumps(profile_dataset(ctx.deps.df))
            + "\nYou can answer structure questions from this profile and use "
            "the other tools for calculations."
        )


    @agent.tool
    def get_dataset_profile(ctx: RunContext[Deps]) -> dict:
        """Rows, columns, dtypes, missing values, duplicates and which columns
        are numeric, categorical or date."""
        return _run_tool(ctx, "get_dataset_profile", {}, lambda: profile_dataset(ctx.deps.df))

    @agent.tool
    def get_numeric_summary(ctx: RunContext[Deps]) -> dict:
        """count/mean/std/min/quartiles/max for every numeric column."""
        return _run_tool(
            ctx, "get_numeric_summary", {}, lambda: calculate_numeric_summary(ctx.deps.df)
        )

    @agent.tool
    def get_column_total(ctx: RunContext[Deps], column: str) -> float:
        """Sum of a numeric column, e.g. total Revenue."""
        return _run_tool(
            ctx, "get_column_total", {"column": column},
            lambda: tools.calculate_total(ctx.deps.df, column),
        )

    @agent.tool
    def get_column_stat(ctx: RunContext[Deps], column: str, stat: str) -> float:
        """One statistic of a numeric column. stat is one of:
        sum, mean, median, min, max, count."""
        return _run_tool(
            ctx, "get_column_stat", {"column": column, "stat": stat},
            lambda: tools.calculate_stat(ctx.deps.df, column, stat),
        )

    @agent.tool
    def group_by_aggregate(
        ctx: RunContext[Deps], group_by: str, value_column: str, agg: str = "sum"
    ) -> list[dict]:
        """Aggregate value_column for each group_by category (sum, mean, median,
        min, max, count), sorted highest first. Use for 'revenue by product'."""
        return _run_tool(
            ctx, "group_by_aggregate",
            {"group_by": group_by, "value_column": value_column, "agg": agg},
            lambda: tools.group_aggregate(ctx.deps.df, group_by, value_column, agg),
        )

    @agent.tool
    def get_top_n(
        ctx: RunContext[Deps], group_by: str, value_column: str, n: int = 5, agg: str = "sum"
    ) -> list[dict]:
        """The top n groups by aggregated value. Use for 'top 3 products'."""
        return _run_tool(
            ctx, "get_top_n",
            {"group_by": group_by, "value_column": value_column, "n": n, "agg": agg},
            lambda: tools.top_n(ctx.deps.df, group_by, value_column, n, agg),
        )

    @agent.tool
    def get_total_excluding_duplicates(ctx: RunContext[Deps], column: str) -> dict:
        """Sum of a numeric column after removing exact duplicate rows. Use for
        'total after removing duplicates'. Also returns the total with
        duplicates and how many duplicate rows were removed."""
        return _run_tool(
            ctx, "get_total_excluding_duplicates", {"column": column},
            lambda: tools.total_excluding_duplicates(ctx.deps.df, column),
        )
    
    @agent.tool
    def get_monthly_trend(
        ctx: RunContext[Deps], date_column: str, value_column: str
    ) -> list[dict]:
        """Sum of value_column per calendar month, in date order."""
        return _run_tool(
            ctx, "get_monthly_trend",
            {"date_column": date_column, "value_column": value_column},
            lambda: tools.monthly_trend(ctx.deps.df, date_column, value_column),
        )

    @agent.tool
    def check_revenue_consistency(ctx: RunContext[Deps]) -> dict:
        """Data-quality check: does Revenue equal Quantity * UnitPrice on every row?"""
        return _run_tool(
            ctx, "check_revenue_consistency", {},
            lambda: tools.check_revenue_consistency(ctx.deps.df),
        )

    return agent
