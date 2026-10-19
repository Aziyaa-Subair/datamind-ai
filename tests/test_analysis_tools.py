import pandas as pd
import pytest

from app import config
from app.services import analysis_tools as t
from app.services.dataset_service import load_dataset


def sales() -> pd.DataFrame:
    return load_dataset(config.DEFAULT_DATASET)


def test_total_revenue_is_37950():
    assert t.calculate_total(sales(), "Revenue") == 37950.0


def test_unknown_column_lists_valid_columns():
    with pytest.raises(t.AnalysisError) as exc:
        t.calculate_total(sales(), "Revenu")
    assert "Available columns" in str(exc.value)


def test_non_numeric_column_rejected():
    with pytest.raises(t.AnalysisError):
        t.calculate_total(sales(), "Product")


def test_stats():
    df = sales()
    assert t.calculate_stat(df, "Revenue", "max") == 12000.0
    assert t.calculate_stat(df, "Revenue", "median") == 3600.0
    assert t.calculate_stat(df, "Quantity", "mean") == pytest.approx(55 / 9)
    assert t.calculate_stat(df, "Revenue", "count") == 9.0


def test_bad_aggregation_rejected():
    with pytest.raises(t.AnalysisError):
        t.calculate_stat(sales(), "Revenue", "variance")


def test_group_by_product():
    rows = t.group_aggregate(sales(), "Product", "Revenue")
    assert rows[0] == {"group": "Laptop", "value": 20400.0}
    assert sum(r["value"] for r in rows) == 37950.0
    assert [r["value"] for r in rows] == sorted((r["value"] for r in rows), reverse=True)


def test_group_by_category():
    rows = t.group_aggregate(sales(), "Category", "Revenue")
    assert rows == [
        {"group": "Electronics", "value": 32900.0},
        {"group": "Furniture", "value": 5050.0},
    ]


def test_top_n():
    rows = t.top_n(sales(), "Product", "Revenue", n=3)
    assert [r["group"] for r in rows] == ["Laptop", "Phone", "Desk"]
    with pytest.raises(t.AnalysisError):
        t.top_n(sales(), "Product", "Revenue", n=0)


def test_monthly_trend():
    rows = t.monthly_trend(sales(), "OrderDate", "Revenue")
    assert rows == [
        {"period": "2025-01", "value": 9400.0},
        {"period": "2025-02", "value": 7900.0},
        {"period": "2025-03", "value": 20650.0},
    ]


def test_monthly_trend_needs_date_column():
    with pytest.raises(t.AnalysisError):
        t.monthly_trend(sales(), "Product", "Revenue")


def test_revenue_consistency_clean_data():
    result = t.check_revenue_consistency(sales())
    assert result["mismatched_rows"] == 0 and result["rows_checked"] == 9


def test_revenue_consistency_finds_bad_row():
    df = sales()
    df.loc[3, "Revenue"] = 1500  # should be 10 * 100 = 1000
    result = t.check_revenue_consistency(df)
    assert result["mismatched_rows"] == 1
    assert result["mismatches"][0] == {"row": 3, "recorded": 1500.0, "expected": 1000.0}


def test_total_excluding_duplicates_on_dirty_data():
    df = load_dataset(config.PROJECT_ROOT / "data" / "sales_dirty.csv")
    result = t.total_excluding_duplicates(df, "Revenue")
    assert result == {
        "total_excluding_duplicates": 19550.0,
        "total_with_duplicates": 20950.0,
        "duplicate_rows_removed": 1,
    }


def test_total_excluding_duplicates_on_clean_data_changes_nothing():
    result = t.total_excluding_duplicates(sales(), "Revenue")
    assert result["total_excluding_duplicates"] == 37950.0
    assert result["duplicate_rows_removed"] == 0


def test_total_excluding_duplicates_validates_column():
    with pytest.raises(t.AnalysisError):
        t.total_excluding_duplicates(sales(), "Revenu")
    with pytest.raises(t.AnalysisError):
        t.total_excluding_duplicates(sales(), "Product")
