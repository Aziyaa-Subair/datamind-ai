"""Loading and profiling datasets. No AI in this file - pure pandas."""

from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


class DatasetError(ValueError):
    """Raised when a dataset cannot be loaded or is unusable."""


def load_dataset(
    source: str | Path | BinaryIO, filename: str | None = None
) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    `source` is a path, or a file-like object (e.g. a Streamlit upload).
    For file-like objects pass `filename` so the format can be detected.
    Columns whose name contains "date" are parsed to datetime.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    ext = Path(name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if isinstance(source, (str, Path)) and not Path(source).exists():
        raise DatasetError(f"File not found: {source}")

    try:
        df = pd.read_csv(source) if ext == ".csv" else pd.read_excel(source)
    except Exception as exc:  # pandas raises many different error types
        raise DatasetError(f"Could not read '{name}': {exc}") from exc

    if df.empty:
        raise DatasetError("The dataset is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    return _parse_date_columns(df)


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns named like '*date*' to datetime (bad values become NaT)."""
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower() and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Structural profile of the dataset, used to orient the AI agent."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    dates = df.select_dtypes(include="datetime").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "date_columns": dates,
    }


def calculate_numeric_summary(df: pd.DataFrame) -> dict:
    """count/mean/std/min/quartiles/max for every numeric column."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}
    summary = numeric.describe().astype(float)
    return {col: summary[col].to_dict() for col in summary.columns}
