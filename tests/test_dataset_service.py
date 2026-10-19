import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app import config
from app.services.dataset_service import (
    DatasetError,
    calculate_numeric_summary,
    load_dataset,
    profile_dataset,
)


def test_load_csv_parses_dates():
    df = load_dataset(config.DEFAULT_DATASET)
    assert df.shape == (9, 6)
    assert pd.api.types.is_datetime64_any_dtype(df["OrderDate"])


def test_load_excel():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.xlsx"
        pd.DataFrame({"A": [1, 2], "B": ["x", "y"]}).to_excel(path, index=False)
        df = load_dataset(path)
    assert df.shape == (2, 2)


def test_unsupported_extension():
    with pytest.raises(DatasetError):
        load_dataset("data.json")


def test_missing_file():
    with pytest.raises(DatasetError):
        load_dataset("definitely_missing.csv")


def test_empty_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.csv"
        path.write_text("A,B\n")
        with pytest.raises(DatasetError):
            load_dataset(path)


def test_profile_sales():
    p = profile_dataset(load_dataset(config.DEFAULT_DATASET))
    assert p["rows"] == 9 and p["columns"] == 6
    assert p["duplicate_rows"] == 0
    assert p["numeric_columns"] == ["Quantity", "UnitPrice", "Revenue"]
    assert p["date_columns"] == ["OrderDate"]
    assert sum(p["missing_values"].values()) == 0


def test_profile_detects_missing_and_duplicates():
    df = pd.DataFrame({"A": [1, 1, None], "B": ["x", "x", "y"]})
    p = profile_dataset(df)
    assert p["missing_values"]["A"] == 1
    assert p["duplicate_rows"] == 1


def test_numeric_summary():
    s = calculate_numeric_summary(load_dataset(config.DEFAULT_DATASET))
    assert s["Revenue"]["max"] == 12000
    assert s["Revenue"]["count"] == 9
