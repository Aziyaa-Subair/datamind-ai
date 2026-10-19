from app.schemas import AnalysisResult, ToolCall, UsedValue
from app.services.verifier import collect_numbers, verify

TOTAL_CALL = ToolCall(name="get_column_total", args={"column": "Revenue"}, result=37950.0)


def result(answer: str, *values: float) -> AnalysisResult:
    return AnalysisResult(
        answer=answer,
        columns_used=["Revenue"],
        values_used=[UsedValue(label="v", value=v) for v in values],
    )


def test_correct_answer_passes():
    report = verify(result("Total revenue is 37,950.", 37950.0), [TOTAL_CALL])
    assert report.passed and not report.errors and not report.warnings


def test_wrong_declared_value_is_caught():
    # The example from the project spec: tool says 37,950, model says 39,750.
    report = verify(result("Total revenue is 39,750.", 39750.0), [TOTAL_CALL])
    assert not report.passed
    assert "39750" in report.errors[0]


def test_wrong_number_in_text_is_warned():
    report = verify(result("Total revenue is 39,750.", 37950.0), [TOTAL_CALL])
    assert report.passed  # declared value is right...
    assert any("39,750" in w for w in report.warnings)  # ...but the text is not


def test_declared_numbers_without_tools_fail():
    report = verify(result("Revenue is 37,950.", 37950.0), [])
    assert not report.passed


def test_no_tools_and_no_numbers_is_only_a_warning():
    report = verify(result("I can help with sales data."), [])
    assert report.passed and report.warnings


def test_years_and_small_counts_are_ignored():
    report = verify(result("In 2025 there were 3 products.", 37950.0), [TOTAL_CALL])
    assert not report.warnings


def test_rounded_text_matches_tool_value():
    call = ToolCall(name="get_column_stat", args={}, result=6.111111111111111)
    report = verify(result("The average quantity is 6.11.", 6.111111111111111), [call])
    assert report.passed and not report.warnings


def test_numbers_inside_nested_results_are_found():
    call = ToolCall(
        name="group_by_aggregate",
        args={},
        result=[{"group": "Laptop", "value": 20400.0}, {"group": "Phone", "value": 10400.0}],
    )
    assert collect_numbers(call.result) == [20400.0, 10400.0]
    report = verify(result("Laptop leads with 20,400.", 20400.0), [call])
    assert report.passed and not report.warnings
