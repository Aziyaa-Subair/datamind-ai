"""Offline check that the eval ground truth in evals/cases.json is correct.

If this fails, either the dataset or cases.json changed without the other.
Runs without any LLM or API key.
"""

import json
import math
from pathlib import Path

from app import config
from app.services import analysis_tools as t
from app.services.dataset_service import load_dataset

CASES = {c["id"]: c for c in json.loads(
    (Path(__file__).parent.parent / "evals" / "cases.json").read_text())}


def close(actual: list[float], expected: list[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(a, e, abs_tol=0.01) for a, e in zip(actual, expected)
    )


def test_ground_truth_matches_tools():
    df = load_dataset(config.DEFAULT_DATASET)
    truth = {
        "total_revenue": [t.calculate_total(df, "Revenue")],
        "top_product": [t.top_n(df, "Product", "Revenue", 1)[0]["value"]],
        "top_3_products": [r["value"] for r in t.top_n(df, "Product", "Revenue", 3)],
        "best_category": [t.top_n(df, "Category", "Revenue", 1)[0]["value"]],
        "avg_quantity": [t.calculate_stat(df, "Quantity", "mean")],
        "max_unit_price": [t.calculate_stat(df, "UnitPrice", "max")],
        "march_revenue": [t.monthly_trend(df, "OrderDate", "Revenue")[2]["value"]],
        "monthly_trend": [r["value"] for r in t.monthly_trend(df, "OrderDate", "Revenue")],
        "row_count": [float(len(df))],
        "consistency": [float(t.check_revenue_consistency(df)["mismatched_rows"])],
        "median_revenue": [t.calculate_stat(df, "Revenue", "median")],
    }
    assert set(truth) == set(CASES)
    for case_id, expected in truth.items():
        assert close(CASES[case_id]["expected"], expected), case_id


def test_headline_number():
    assert CASES["total_revenue"]["expected"] == [37950.0]
