"""Chart specifications built deterministically from the data.

A "spec" is a plain dict (labels, numbers, notes) - no plotting library is
involved, so everything here is unit-tested without Plotly. The numbers are
computed by pandas/numpy; the AI agent only chooses WHICH chart to ask for.
app/services/chart_render.py turns a spec into a Plotly figure.

Spec shape:
    {"chart_type": "bar" | "line" | "pie" | "histogram" | "scatter",
     "title": str, "x_label": str, "y_label": str,
     "x": [...], "y": [...], "notes": [str, ...]}

`notes` say what the chart leaves out (e.g. rows with a missing date), so a
chart never silently drops data.
"""

from typing import Any

import numpy as np
import pandas as pd

from app.services.analysis_tools import (
    AnalysisError,
    _require_numeric,
    group_aggregate,
    monthly_trend,
)

CHART_TYPES = ("bar", "line", "pie", "histogram", "scatter")
MAX_PIE_SLICES = 6        # more than this is unreadable; use a bar chart
MAX_SCATTER_POINTS = 500
HISTOGRAM_BINS = 10

_AGG_LABEL = {
    "sum": "Total", "mean": "Average", "median": "Median",
    "min": "Minimum", "max": "Maximum", "count": "Count of",
}


def _spec(chart_type: str, title: str, x_label: str, y_label: str,
          x: list, y: list, notes: list[str]) -> dict:
    return {"chart_type": chart_type, "title": title, "x_label": x_label,
            "y_label": y_label, "x": x, "y": y, "notes": notes}


def _missing_note(df: pd.DataFrame, column: str, verb: str = "shown") -> list[str]:
    n = int(df[column].isna().sum())
    return [f"{n} row(s) with a missing {column} are not {verb}."] if n else []


def bar_spec(df: pd.DataFrame, group_by: str, value_column: str,
             agg: str = "sum", limit: int = 15) -> dict:
    """One bar per group, highest first."""
    if limit < 1:
        raise AnalysisError("limit must be at least 1.")
    rows = group_aggregate(df, group_by, value_column, agg)  # validates inputs
    notes = _missing_note(df, group_by)
    if len(rows) > limit:
        notes.append(f"Showing the top {limit} of {len(rows)} groups.")
        rows = rows[:limit]
    return _spec(
        "bar", f"{_AGG_LABEL[agg]} {value_column} by {group_by}",
        group_by, value_column,
        [r["group"] for r in rows], [r["value"] for r in rows], notes,
    )


def line_spec(df: pd.DataFrame, date_column: str, value_column: str) -> dict:
    """Monthly total of `value_column`, in date order."""
    rows = monthly_trend(df, date_column, value_column)  # validates inputs
    return _spec(
        "line", f"Monthly {value_column}", "Month", value_column,
        [r["period"] for r in rows], [r["value"] for r in rows],
        _missing_note(df, date_column),
    )


def pie_spec(df: pd.DataFrame, group_by: str, value_column: str,
             agg: str = "sum") -> dict:
    """Share of the total per group. Only for a few categories."""
    if agg not in ("sum", "count"):
        raise AnalysisError("A pie chart shows shares, so agg must be 'sum' or 'count'.")
    rows = group_aggregate(df, group_by, value_column, agg)
    if any(r["value"] < 0 for r in rows):
        raise AnalysisError("A pie chart cannot show negative values. Use a bar chart.")
    if len(rows) > MAX_PIE_SLICES:
        raise AnalysisError(
            f"A pie chart needs {MAX_PIE_SLICES} or fewer categories; "
            f"'{group_by}' has {len(rows)}. Use a bar chart instead."
        )
    return _spec(
        "pie", f"Share of {value_column} by {group_by}", group_by, value_column,
        [r["group"] for r in rows], [r["value"] for r in rows],
        _missing_note(df, group_by),
    )


def histogram_spec(df: pd.DataFrame, column: str, bins: int = HISTOGRAM_BINS) -> dict:
    """Row counts per value range. Bins are computed here, not by the chart."""
    _require_numeric(df, column)
    if not 1 <= bins <= 50:
        raise AnalysisError("bins must be between 1 and 50.")
    values = df[column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise AnalysisError(f"Column '{column}' has no values to chart.")
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:g} to {edges[i + 1]:g}" for i in range(len(counts))]
    return _spec(
        "histogram", f"Distribution of {column}", column, "Rows",
        labels, [float(c) for c in counts],
        _missing_note(df, column, verb="included"),
    )


def scatter_spec(df: pd.DataFrame, x_column: str, y_column: str) -> dict:
    """One point per row for two numeric columns."""
    _require_numeric(df, x_column)
    _require_numeric(df, y_column)
    pair = df[[x_column, y_column]].dropna()
    notes = []
    dropped = len(df) - len(pair)
    if dropped:
        notes.append(f"{dropped} row(s) with a missing value are not shown.")
    if len(pair) == 0:
        raise AnalysisError("No rows have both values, so there is nothing to plot.")
    if len(pair) > MAX_SCATTER_POINTS:
        notes.append(f"Showing the first {MAX_SCATTER_POINTS} of {len(pair)} points.")
        pair = pair.head(MAX_SCATTER_POINTS)
    return _spec(
        "scatter", f"{y_column} vs {x_column}", x_column, y_column,
        [float(v) for v in pair[x_column]], [float(v) for v in pair[y_column]], notes,
    )


def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
                agg: str = "sum", limit: int = 15) -> dict:
    """Single entry point used by the agent's create_chart tool.

    bar / pie : x = category column,        y = numeric column
    line      : x = date column,            y = numeric column (monthly total)
    histogram : x = numeric column          (y not needed)
    scatter   : x = numeric column,         y = numeric column
    """
    if chart_type not in CHART_TYPES:
        raise AnalysisError(f"Unsupported chart_type '{chart_type}'. Use one of {CHART_TYPES}.")
    if chart_type != "histogram" and not y:
        raise AnalysisError(f"A {chart_type} chart needs both x and y columns.")
    if chart_type == "bar":
        return bar_spec(df, x, y, agg, limit)
    if chart_type == "line":
        return line_spec(df, x, y)
    if chart_type == "pie":
        return pie_spec(df, x, y, agg)
    if chart_type == "histogram":
        return histogram_spec(df, x)
    return scatter_spec(df, x, y)


def extract_chart_specs(tool_calls: list[Any]) -> list[dict]:
    """Pull chart specs out of recorded tool calls (objects or plain dicts)."""
    specs = []
    for call in tool_calls:
        name = call["name"] if isinstance(call, dict) else call.name
        result = call["result"] if isinstance(call, dict) else call.result
        if name == "create_chart" and isinstance(result, dict):
            specs.append(result)
    return specs
