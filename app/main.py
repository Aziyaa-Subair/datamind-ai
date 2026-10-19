"""Command-line entry point.

    python -m app.main                          # deterministic demo, no API needed
    python -m app.main "What is the total revenue?"   # asks the AI agent
"""

import sys

from app import config
from app.services.analysis_tools import calculate_total
from app.services.dataset_service import (
    calculate_numeric_summary,
    load_dataset,
    profile_dataset,
)


def main() -> None:
    df = load_dataset(config.DEFAULT_DATASET)
    print(df.head(), "\n")
    print("Profile:", profile_dataset(df), "\n")
    print("Total revenue:", calculate_total(df, "Revenue"), "\n")
    print("Numeric summary:", calculate_numeric_summary(df), "\n")

    if len(sys.argv) > 1:
        from app.pipeline import answer_question  # imported here: needs an API key

        verified = answer_question(df, " ".join(sys.argv[1:]))
        print("AI answer:", verified.result.answer)
        print("Verification passed:", verified.report.passed)
        for msg in verified.report.errors + verified.report.warnings:
            print(" -", msg)
        print("Tools called:", [c.name for c in verified.tool_calls])


if __name__ == "__main__":
    main()
