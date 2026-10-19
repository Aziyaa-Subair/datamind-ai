import pandas as pd

from app import config
from app.services import dashboard
from app.services.dataset_service import load_dataset


def sales() -> pd.DataFrame:
    return load_dataset(config.DEFAULT_DATASET)


def dirty() -> pd.DataFrame:
    return load_dataset(config.PROJECT_ROOT / "data" / "sales_dirty.csv")


def test_column_picking_on_sales():
    df = sales()
    assert dashboard.pick_value_column(df) == "Revenue"
    assert dashboard.pick_date_column(df) == "OrderDate"
    assert dashboard.pick_category_columns(df) == ["Product", "Category"]


def test_value_column_falls_back_to_last_numeric():
    df = pd.DataFrame({"name": ["a", "b"], "x": [1, 2], "y": [3, 4]})
    assert dashboard.pick_value_column(df) == "y"
    assert dashboard.pick_value_column(pd.DataFrame({"name": ["a", "b"]})) is None


def test_high_cardinality_text_columns_are_not_categories():
    df = pd.DataFrame({"id": [f"id{i}" for i in range(40)], "v": range(40)})
    assert dashboard.pick_category_columns(df) == []


def test_kpis_on_sales():
    values = {k["label"]: k["value"] for k in dashboard.kpis(sales())}
    assert values["Rows"] == 9.0
    assert values["Total Revenue"] == 37950.0
    assert abs(values["Average Revenue"] - 37950.0 / 9) < 1e-9


def test_quality_summary_clean_data():
    q = dashboard.quality_summary(sales())
    assert q["duplicate_rows"] == 0 and q["missing_total"] == 0
    assert q["revenue_check"]["mismatched_rows"] == 0


def test_quality_summary_finds_every_planted_defect():
    q = dashboard.quality_summary(dirty())
    assert q["duplicate_rows"] == 1
    assert q["missing_by_column"] == {"OrderDate": 1, "Category": 1}
    assert q["revenue_check"]["mismatched_rows"] == 1


def test_quality_summary_without_revenue_columns():
    q = dashboard.quality_summary(pd.DataFrame({"a": [1, 2]}))
    assert q["revenue_check"] is None


def test_auto_charts_on_sales():
    specs = dashboard.auto_charts(sales())
    assert [s["chart_type"] for s in specs] == ["line", "bar", "bar", "histogram"]
    assert sum(specs[1]["y"]) == 37950.0


def test_auto_charts_on_dirty_data_say_what_they_drop():
    specs = dashboard.auto_charts(dirty())
    notes = [n for s in specs for n in s["notes"]]
    assert any("missing OrderDate" in n for n in notes)
    assert any("missing Category" in n for n in notes)


def test_auto_charts_on_a_dataset_with_no_numbers():
    assert dashboard.auto_charts(pd.DataFrame({"a": ["x", "y"]})) == []
