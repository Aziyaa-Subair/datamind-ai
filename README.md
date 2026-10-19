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

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then put your key in .env
python scripts\list_models.py # see which models your key supports
pytest                        # run the tests (no API key needed)
python -m app.main            # deterministic demo
python -m app.main "What is the total revenue?"   # ask the agent
streamlit run app/streamlit_app.py
python -m evals.run_evals     # score the agent (calls the real model)
```

## Eval results

_Run `python -m evals.run_evals` and paste your score here, e.g. "9/11 (82%) on 11 questions."_

## Roadmap
- [x] Data layer, tools, structured output, verifier, tests, evals, Streamlit UI
- [ ] Plotly chart tool (bar / line) and chart display in the UI
- [ ] More eval cases (ambiguous questions, missing columns, dirty data)
- [ ] FastAPI endpoints, then MCP server exposing the same tools
- [ ] Dockerfile, deployment, optional SQLite history
