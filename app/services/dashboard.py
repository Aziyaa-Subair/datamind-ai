"""Dashboard content computed from any dataset. No AI and no API calls, so the
dashboard works even when your Gemini quota is used up.

Column choices are simple heuristics (see pick_*). They are suggestions for a
first look at the data, not a claim about what the user cares about.
"""

import pandas as pd

from app.services import analysis_tools as tools
from app.services import charts
from app.services.analysis_tools import AnalysisError

_VALUE_HINTS = ("revenue", "sales", "amount", "total", "profit", "income", "price")
MAX_CATEGORIES = 15


def pick_value_column(df: pd.DataFrame) -> str | None:
    """The numeric column to headline: a revenue-like name, else the last numeric one."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return None
    for hint in _VALUE_HINTS:
        for col in numeric:
            if hint in col.lower():
                return col
    return numeric[-1]


def pick_category_columns(df: pd.DataFrame, limit: int = 2) -> list[str]:
    """Text columns that group the rows into 2 to 15 categories."""
    picked = []
    for col in df.columns:
        is_text = not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])
        if is_text and 2 <= df[col].nunique() <= MAX_CATEGORIES:
            picked.append(col)
    return picked[:limit]


def pick_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().any():
            return col
    return None


def kpis(df: pd.DataFrame) -> list[dict]:
    """Headline numbers. format is 'int' or 'number'."""
    out = [{"label": "Rows", "value": float(len(df)), "format": "int"}]
    value = pick_value_column(df)
    if value:
        out.append({"label": f"Total {value}", "value": float(df[value].sum()), "format": "number"})
        out.append({"label": f"Average {value}", "value": float(df[value].mean()), "format": "number"})
    return out


def quality_summary(df: pd.DataFrame) -> dict:
    """Duplicates, missing values and (if the columns exist) revenue consistency."""
    missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    summary = {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": missing,
        "missing_total": int(sum(missing.values())),
        "revenue_check": None,
    }
    try:
        summary["revenue_check"] = tools.check_revenue_consistency(df)
    except AnalysisError:
        pass  # the dataset has no Quantity / UnitPrice / Revenue columns
    return summary


def auto_charts(df: pd.DataFrame) -> list[dict]:
    """A sensible first set of chart specs: trend, breakdowns, distribution."""
    value = pick_value_column(df)
    if value is None:
        return []
    builders = []
    date = pick_date_column(df)
    if date:
        builders.append(lambda: charts.line_spec(df, date, value))
    for category in pick_category_columns(df):
        builders.append(lambda c=category: charts.bar_spec(df, c, value))
    builders.append(lambda: charts.histogram_spec(df, value))

    specs = []
    for build in builders:
        try:
            specs.append(build())
        except AnalysisError:
            continue  # skip a chart that does not fit this dataset
    return specs
