"""Deterministic analysis functions the AI agent can call as tools.

Every function validates its inputs and raises AnalysisError with a
helpful message (including the valid column names) so the agent can
correct itself. The LLM never does arithmetic - these functions do.
"""

import pandas as pd

AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")


class AnalysisError(ValueError):
    """Raised for bad columns, wrong types, or unsupported operations."""


# ---------- validation helpers ----------

def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise AnalysisError(
            f"Column '{column}' does not exist. "
            f"Available columns: {df.columns.tolist()}"
        )


def _require_numeric(df: pd.DataFrame, column: str) -> None:
    _require_column(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise AnalysisError(
            f"Column '{column}' is not numeric (type: {df[column].dtype})."
        )


def _require_datetime(df: pd.DataFrame, column: str) -> None:
    _require_column(df, column)
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise AnalysisError(f"Column '{column}' is not a date column.")


def _require_agg(agg: str) -> None:
    if agg not in AGGREGATIONS:
        raise AnalysisError(f"Unsupported aggregation '{agg}'. Use one of {AGGREGATIONS}.")


# ---------- analysis functions ----------

def calculate_total(df: pd.DataFrame, column: str) -> float:
    """Sum of a numeric column."""
    _require_numeric(df, column)
    return float(df[column].sum())


def calculate_stat(df: pd.DataFrame, column: str, stat: str) -> float:
    """One statistic (sum, mean, median, min, max, count) of a numeric column."""
    _require_agg(stat)
    _require_numeric(df, column)
    return float(getattr(df[column], stat)())


def group_aggregate(
    df: pd.DataFrame, group_by: str, value_column: str, agg: str = "sum"
) -> list[dict]:
    """Aggregate `value_column` per group, sorted from highest to lowest."""
    _require_agg(agg)
    _require_column(df, group_by)
    _require_numeric(df, value_column)
    grouped = df.groupby(group_by)[value_column].agg(agg).sort_values(ascending=False)
    return [{"group": str(k), "value": float(v)} for k, v in grouped.items()]


def top_n(
    df: pd.DataFrame, group_by: str, value_column: str, n: int = 5, agg: str = "sum"
) -> list[dict]:
    """The `n` highest groups by aggregated value."""
    if n < 1:
        raise AnalysisError("n must be at least 1.")
    return group_aggregate(df, group_by, value_column, agg)[:n]


def total_excluding_duplicates(df: pd.DataFrame, column: str) -> dict:
    """Sum of a numeric column after dropping exact duplicate rows (the first
    copy of each row is kept). Also reports how many rows were removed, so the
    caller can see what the dedupe did."""
    _require_numeric(df, column)
    deduped = df.drop_duplicates()
    return {
        "total_excluding_duplicates": float(deduped[column].sum()),
        "total_with_duplicates": float(df[column].sum()),
        "duplicate_rows_removed": int(len(df) - len(deduped)),
    }


def monthly_trend(df: pd.DataFrame, date_column: str, value_column: str) -> list[dict]:
    """Sum of `value_column` per calendar month, in chronological order."""
    _require_datetime(df, date_column)
    _require_numeric(df, value_column)
    valid = df.dropna(subset=[date_column])
    monthly = valid.groupby(valid[date_column].dt.to_period("M"))[value_column].sum()
    return [{"period": str(p), "value": float(v)} for p, v in monthly.items()]


def check_revenue_consistency(
    df: pd.DataFrame,
    quantity_column: str = "Quantity",
    price_column: str = "UnitPrice",
    revenue_column: str = "Revenue",
    tolerance: float = 0.01,
) -> dict:
    """Data-quality check: does Revenue equal Quantity * UnitPrice on every row?"""
    for col in (quantity_column, price_column, revenue_column):
        _require_numeric(df, col)
    expected = df[quantity_column] * df[price_column]
    bad = (df[revenue_column] - expected).abs() > tolerance
    mismatches = [
        {
            "row": int(i),
            "recorded": float(df.at[i, revenue_column]),
            "expected": float(expected[i]),
        }
        for i in df.index[bad]
    ]
    return {
        "rows_checked": int(len(df)),
        "mismatched_rows": len(mismatches),
        "mismatches": mismatches,
    }
