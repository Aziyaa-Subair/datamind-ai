# DataMind AI

Ask questions about a CSV/Excel file in plain English. An AI agent (Pydantic AI + Gemini)
chooses which analysis tool to call; **pandas does all the math**; a **deterministic
verifier** then checks every number in the answer against the tool outputs.

## Why it is built this way

| Principle | How it is enforced |
|---|---|
| The LLM never calculates | It can only call fixed, validated functions in `app/services/analysis_tools.py` |
| No arbitrary code execution | Tools are a closed set; no `eval`/generated code |
| Answers are auditable | Every tool call is recorded and shown in the UI |
| Hallucinated numbers are caught | `app/services/verifier.py` compares claims with tool results (plain Python, no second LLM) |
| Quality is measured | `evals/` scores the agent on questions with pandas-computed answers |


```

