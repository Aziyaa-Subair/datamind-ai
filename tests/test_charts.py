import pandas as pd
import pytest

from app import config
from app.schemas import ToolCall
from app.services import charts
from app.services.analysis_tools import AnalysisError
from app.services.dataset_service import load_dataset


def sales() -> pd.DataFrame:
    return load_dataset(config.DEFAULT_DATASET)


def dirty() -> pd.DataFrame:
    return load_dataset(config.PROJECT_ROOT / "data" / "sales_dirty.csv")


def test_bar_spec_matches_group_totals():
    spec = charts.bar_spec(sales(), "Product", "Revenue")
    assert spec["chart_type"] == "bar"
    assert spec["x"][0] == "Laptop" and spec["y"][0] == 20400.0
    assert sum(spec["y"]) == 37950.0
    assert spec["title"] == "Total Revenue by Product"
    assert spec["notes"] == []


def test_bar_spec_limit_adds_a_note():
    spec = charts.bar_spec(sales(), "Product", "Revenue", limit=2)
    assert spec["x"] == ["Laptop", "Phone"]
    assert "top 2 of 5" in spec["notes"][0]


def test_bar_spec_reports_rows_it_leaves_out():
    spec = charts.bar_spec(dirty(), "Category", "Revenue")
    assert sum(spec["y"]) == 19600.0           # 20,950 minus the 1,350 row with no category
    assert any("1 row(s) with a missing Category" in n for n in spec["notes"])


def test_line_spec_is_monthly_and_reports_bad_dates():
    spec = charts.line_spec(dirty(), "OrderDate", "Revenue")
    assert spec["x"] == ["2025-01", "2025-02", "2025-03"]
    assert spec["y"] == [5700.0, 4600.0, 10150.0]
    assert any("missing OrderDate" in n for n in spec["notes"])


def test_pie_spec():
    spec = charts.pie_spec(sales(), "Category", "Revenue")
    assert spec["x"] == ["Electronics", "Furniture"]
    assert sum(spec["y"]) == 37950.0


def test_pie_rejects_too_many_slices_negative_values_and_means():
    df = pd.DataFrame({"k": list("abcdefgh"), "v": range(1, 9)})
    with pytest.raises(AnalysisError):
        charts.pie_spec(df, "k", "v")
    with pytest.raises(AnalysisError):
        charts.pie_spec(pd.DataFrame({"k": ["a", "b"], "v": [5, -1]}), "k", "v")
    with pytest.raises(AnalysisError):
        charts.pie_spec(sales(), "Category", "Revenue", agg="mean")


def test_histogram_counts_every_row_once():
    spec = charts.histogram_spec(sales(), "Revenue")
    assert spec["chart_type"] == "histogram"
    assert len(spec["x"]) == len(spec["y"]) == 10
    assert sum(spec["y"]) == 9.0
    assert spec["y_label"] == "Rows"


def test_histogram_of_a_constant_column_does_not_crash():
    spec = charts.histogram_spec(pd.DataFrame({"a": [5, 5, 5]}), "a", bins=3)
    assert sum(spec["y"]) == 3.0


def test_scatter_spec_and_missing_values():
    spec = charts.scatter_spec(sales(), "Quantity", "Revenue")
    assert len(spec["x"]) == 9
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
    spec = charts.scatter_spec(df, "a", "b")
    assert spec["x"] == [1.0, 3.0]
    assert "1 row(s)" in spec["notes"][0]


def test_scatter_is_capped():
    df = pd.DataFrame({"a": range(800), "b": range(800)})
    spec = charts.scatter_spec(df, "a", "b")
    assert len(spec["x"]) == charts.MAX_SCATTER_POINTS
    assert "first 500 of 800" in spec["notes"][0]


def test_bad_columns_are_rejected_with_helpful_errors():
    with pytest.raises(AnalysisError) as exc:
        charts.bar_spec(sales(), "Prodct", "Revenue")
    assert "Available columns" in str(exc.value)
    with pytest.raises(AnalysisError):
        charts.bar_spec(sales(), "Product", "Category")     # y must be numeric
    with pytest.raises(AnalysisError):
        charts.line_spec(sales(), "Product", "Revenue")     # x must be a date
    with pytest.raises(AnalysisError):
        charts.histogram_spec(sales(), "Product")


def test_build_chart_dispatch_and_validation():
    assert charts.build_chart(sales(), "bar", "Product", "Revenue")["chart_type"] == "bar"
    assert charts.build_chart(sales(), "line", "OrderDate", "Revenue")["chart_type"] == "line"
    assert charts.build_chart(sales(), "histogram", "Revenue")["chart_type"] == "histogram"
    with pytest.raises(AnalysisError):
        charts.build_chart(sales(), "donut", "Product", "Revenue")
    with pytest.raises(AnalysisError):
        charts.build_chart(sales(), "bar", "Product")        # y missing


def test_extract_chart_specs_from_objects_and_dicts():
    spec = charts.bar_spec(sales(), "Product", "Revenue")
    calls = [
        ToolCall(name="get_column_total", args={}, result=37950.0),
        ToolCall(name="create_chart", args={}, result=spec),
    ]
    assert charts.extract_chart_specs(calls) == [spec]
    dumped = [c.model_dump() for c in calls]
    assert charts.extract_chart_specs(dumped) == [spec]
