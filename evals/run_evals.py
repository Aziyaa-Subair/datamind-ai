"""Score the agent against questions with known answers.

    python -m evals.run_evals            # all sets
    python -m evals.run_evals basic      # clean sales.csv, simple questions
    python -m evals.run_evals hard       # messy sales_dirty.csv, traps

Calls the real model, so it needs your API key. Each case ends as one of:
  PASS   - see evals/scoring.py: expected numbers reported, accepted tool used,
           no forbidden numbers, verifier passed.
  FAIL   - the agent answered, but the answer did not meet those conditions.
  ERROR  - the model call itself failed (quota, overload, network). The agent
           never answered, so this is NOT counted against accuracy.

Accuracy = PASS / (PASS + FAIL). Re-run later to retry ERROR cases.
Ground truth is computed with pandas; tests/ re-checks it offline.
"""

import json
import sys
import time
from pathlib import Path

from app import config
from app.agents.analysis_agent import build_agent
from app.pipeline import answer_question
from app.services.dataset_service import load_dataset
from evals.scoring import score_case

HERE = Path(__file__).parent
CASE_SETS = {"basic": HERE / "cases.json", "hard": HERE / "cases_hard.json"}
RESULTS_FILE = HERE / "results.json"
DEFAULT_DATASET = "sales.csv"
PAUSE_SECONDS = 2  # be gentle with free-tier rate limits
# Only these mean "the service failed". Anything else (e.g. the agent exhausting
# its tool retries) is a real agent failure and counts as FAIL.
INFRA_ERRORS = {
    "ModelHTTPError", "ModelAPIError", "FallbackExceptionGroup",
    "ConnectError", "ReadTimeout", "TimeoutError", "ConnectionError",
}

def run(selected: list[str], only: set[str] | None = None) -> None:
    agent = build_agent()
    datasets = {}  # load each dataset once
    results = []

    for set_name in selected:
        cases = json.loads(CASE_SETS[set_name].read_text())
        if only:
            cases = [c for c in cases if c["id"] in only]
        print(f"\n== {set_name} ({len(cases)} cases) ==")
        for case in cases:
            dataset = case.get("dataset", DEFAULT_DATASET)
            if dataset not in datasets:
                datasets[dataset] = load_dataset(config.PROJECT_ROOT / "data" / dataset)

            record = {"set": set_name, "id": case["id"], "dataset": dataset}
            try:
                out = answer_question(datasets[dataset], case["question"], agent=agent)
            except Exception as exc:
                name = type(exc).__name__
                record.update(
                    status="ERROR" if name in INFRA_ERRORS else "FAIL",
                    error=f"{name}: {exc}",
                )
            else:
                values = [u.value for u in out.result.values_used]
                tools = [c.name for c in out.tool_calls]
                checks = score_case(case, out.result.answer, values, tools, out.report.passed)
                record.update(
                    status="PASS" if checks.pop("passed") else "FAIL",
                    checks=checks,
                    answer=out.result.answer,
                    values=values,
                    tools=tools,
                    verifier_errors=out.report.errors,
                )

            results.append(record)
            print(f"{record['status']:<5}  {case['id']}")
            time.sleep(PAUSE_SECONDS)

    for name in selected:
        subset = [r for r in results if r["set"] == name]
        passed = sum(r["status"] == "PASS" for r in subset)
        answered = sum(r["status"] != "ERROR" for r in subset)
        errored = len(subset) - answered
        line = f"{name}: {passed}/{answered} = {passed / answered:.0%}" if answered else f"{name}: no answered cases"
        print(line + (f"  ({errored} API error(s), not scored)" if errored else ""))

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Details written to {RESULTS_FILE}")


if __name__ == "__main__":
    args = sys.argv[1:]
    sets = [a for a in args if a in CASE_SETS] or list(CASE_SETS)
    ids = {a for a in args if a not in CASE_SETS}
    known = {c["id"] for s in sets for c in json.loads(CASE_SETS[s].read_text())}
    unknown = ids - known
    if unknown:
        raise SystemExit(f"Unknown case id(s) {sorted(unknown)}. Known: {sorted(known)}")
    run(sets, ids or None)