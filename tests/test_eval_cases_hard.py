"""Offline checks for the hard eval set: the messy dataset really contains the
defects the cases rely on, and cases_hard.json matches what the tools compute.
No LLM or API key needed."""

import json
import math
from pathlib import Path

from app import config
from app.services import analysis_tools as t
from app.services.dataset_service import load_dataset, profile_dataset

CASES = {c["id"]: c for c in json.loads(
    (Path(__file__).parent.parent / "evals" / "cases_hard.json").read_text())}


def dirty():
    return load_dataset(config.PROJECT_ROOT / "data" / "sales_dirty.csv")


def close(actual, expected) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(a, e, abs_tol=0.01) for a, e in zip(actual, expected))


def test_dataset_contains_the_planted_defects():
    df = dirty()
    p = profile_dataset(df)
    assert p["rows"] == 12
    assert p["duplicate_rows"] == 1
    assert p["missing_values"]["Category"] == 1
    assert p["missing_values"]["OrderDate"] == 1          # "unknown" became NaT
    assert t.check_revenue_consistency(df)["mismatched_rows"] == 1


def test_defects_change_the_totals_as_the_cases_assume():
    df = dirty()
    total = t.calculate_total(df, "Revenue")
    by_category = sum(r["value"] for r in t.group_aggregate(df, "Category", "Revenue"))
    by_month = sum(r["value"] for r in t.monthly_trend(df, "OrderDate", "Revenue"))
    assert total == 20950.0
    assert by_category == 19600.0   # missing-category row (1,350) silently excluded
    assert by_month == 20450.0      # bad-date row (500) silently excluded


def test_ground_truth_matches_tools():
    df = dirty()
    p = profile_dataset(df)
    cons = t.check_revenue_consistency(df)["mismatches"][0]
    truth = {
        "dirty_total": [t.calculate_total(df, "Revenue")],
        "dirty_duplicates": [float(p["duplicate_rows"])],
        "dirty_missing_category": [float(p["missing_values"]["Category"])],
        "dirty_mismatch_count": [1.0],
        "dirty_mismatch_values": [cons["recorded"], cons["expected"]],
        "dirty_best_category": [t.top_n(df, "Category", "Revenue", 1)[0]["value"]],
        "dirty_top_product": [t.top_n(df, "Product", "Revenue", 1)[0]["value"]],
        "dirty_march": [t.monthly_trend(df, "OrderDate", "Revenue")[2]["value"]],
        "dirty_mean_revenue": [t.calculate_stat(df, "Revenue", "mean")],
        "unanswerable_profit": [],
        "unanswerable_region": [],
        "no_llm_math_dedupe": [t.total_excluding_duplicates(df, "Revenue")["total_excluding_duplicates"]],
    }
    assert set(truth) == set(CASES)
    for case_id, expected in truth.items():
        assert close(CASES[case_id]["expected"], expected), case_id


def test_dedupe_case_is_answerable_by_a_tool():
    # The question used to have no matching tool (the model had to subtract
    # the duplicate itself). Now a deterministic tool produces the number.
    case = CASES["no_llm_math_dedupe"]
    assert case["expected_tool"] == "get_total_excluding_duplicates"
    assert "forbidden_values" not in case
    assert case["expected"] == [19550.0]


def test_unanswerable_columns_really_do_not_exist():
    columns = [c.lower() for c in dirty().columns]
    assert "profit" not in columns and "region" not in columns
