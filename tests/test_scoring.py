from evals.scoring import score_case


def passes(case, answer="", values=(), tools=(), verifier=True) -> bool:
    return score_case(case, answer, list(values), list(tools), verifier)["passed"]


def test_correct_value_and_tool_passes():
    case = {"expected": [37950.0], "expected_tool": "get_column_total"}
    assert passes(case, values=[37950.0], tools=["get_column_total"])


def test_wrong_value_fails():
    case = {"expected": [37950.0], "expected_tool": "get_column_total"}
    assert not passes(case, values=[39750.0], tools=["get_column_total"])


def test_any_listed_tool_is_accepted():
    case = {"expected": [20400.0], "expected_tool": ["group_by_aggregate", "get_top_n"]}
    assert passes(case, values=[20400.0], tools=["get_top_n"])
    assert not passes(case, values=[20400.0], tools=["get_column_stat"])


def test_failed_verifier_fails_even_if_values_match():
    case = {"expected": [1.0]}
    assert not passes(case, values=[1.0], verifier=False)


def test_rounded_expected_value_is_accepted():
    case = {"expected": [1745.83]}
    assert passes(case, values=[1745.8333333])


def test_forbidden_value_fails():
    # The agent reported a number only arithmetic could produce.
    case = {"expected": [], "forbidden_values": [19550.0]}
    assert not passes(case, values=[19550.0])
    assert passes(case, values=[20950.0])


def test_unanswerable_needs_mention_and_negation():
    case = {"expected": [], "unanswerable_mention": "profit"}
    assert passes(case, answer="The dataset does not contain a Profit column.")
    assert passes(case, answer="I can't calculate profit from this data.")
    assert not passes(case, answer="Total profit is 12,000.")      # invented answer
    assert not passes(case, answer="The total revenue is 37,950.")  # ignored the question


def test_no_expected_tool_means_tool_is_not_checked():
    assert passes({"expected": [5.0]}, values=[5.0], tools=[])
