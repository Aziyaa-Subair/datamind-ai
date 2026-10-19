"""DataMind AI dashboard.   Run from the project root:

    streamlit run app/streamlit_app.py

Tabs:
  Dashboard - KPIs, charts and data-quality checks. No AI, no API quota used.
  Ask AI    - questions in plain English; answers are verified, charts supported.
  Data      - preview and profile.
"""

import sys
from pathlib import Path

# Make `import app...` work when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.services import dashboard  # noqa: E402
from app.services.chart_render import to_figure  # noqa: E402
from app.services.charts import extract_chart_specs  # noqa: E402
from app.services.dataset_service import (  # noqa: E402
    DatasetError,
    load_dataset,
    profile_dataset,
)

st.set_page_config(page_title="DataMind AI", layout="wide")

DATA_DIR = config.PROJECT_ROOT / "data"
EXAMPLES = [
    "What is the total revenue?",
    "Which product generated the most revenue?",
    "Show a bar chart of revenue by category",
    "Show the monthly revenue trend as a line chart",
]


# ---------- helpers ----------

def fmt(value: float, kind: str) -> str:
    if kind == "int" or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def show_chart(spec: dict, key: str) -> None:
    st.plotly_chart(to_figure(spec), key=key)
    for note in spec.get("notes", []):
        st.caption(f"Note: {note}")
    with st.expander("View as table"):
        table = pd.DataFrame(list(zip(spec["x"], spec["y"])),
                             columns=[spec["x_label"], spec["y_label"]])
        st.dataframe(table)


@st.cache_resource
def get_agent():
    from app.agents.analysis_agent import build_agent  # needs an API key

    return build_agent()


def ask_agent(df: pd.DataFrame, question: str) -> dict:
    """Run the agent and return a plain dict we can keep in session state."""
    try:
        from app.pipeline import answer_question

        verified = answer_question(df, question, agent=get_agent())
    except Exception as exc:  # missing key, quota, network...
        return {"role": "assistant", "error": f"{type(exc).__name__}: {exc}"}
    dumped = verified.model_dump()
    return {
        "role": "assistant",
        "answer": dumped["result"]["answer"],
        "passed": dumped["report"]["passed"],
        "errors": dumped["report"]["errors"],
        "warnings": dumped["report"]["warnings"],
        "tool_calls": dumped["tool_calls"],
    }


def render_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
            return
        if "error" in message:
            st.error(f"The AI request failed: {message['error']}")
            return
        st.write(message["answer"])
        if message["passed"]:
            st.success("Verified: every number the AI declared matches a tool result.")
        else:
            st.error("Verification FAILED - do not trust the numbers in this answer.")
        for text in message["errors"]:
            st.error(text)
        for text in message["warnings"]:
            st.warning(text)
        for n, spec in enumerate(extract_chart_specs(message["tool_calls"])):
            show_chart(spec, key=f"chat-{index}-{n}")
        with st.expander("How this was calculated"):
            for call in message["tool_calls"]:
                if call["name"] == "create_chart":
                    st.code(f"create_chart({call['args']})")
                else:
                    st.code(f"{call['name']}({call['args']}) -> {str(call['result'])[:300]}")


# ---------- sidebar: choose data ----------

samples = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx"))

with st.sidebar:
    st.header("Data")
    options = (["Sample dataset"] if samples else []) + ["Upload a file"]
    source = st.radio("Source", options)
    chosen, uploaded = None, None
    if source == "Sample dataset":
        chosen = st.selectbox("Dataset", samples)
    else:
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])

if chosen:
    dataset_name = chosen
    loader = lambda: load_dataset(DATA_DIR / chosen)  # noqa: E731
elif uploaded is not None:
    dataset_name = uploaded.name
    loader = lambda: load_dataset(uploaded, filename=uploaded.name)  # noqa: E731
else:
    st.title("DataMind AI")
    st.info("Pick a sample dataset or upload a CSV/Excel file in the sidebar.")
    st.stop()

try:
    df = loader()
except DatasetError as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.get("messages_for") != dataset_name:  # new dataset, fresh chat
    st.session_state.messages = []
    st.session_state.messages_for = dataset_name

st.title("DataMind AI")
st.caption(f"{dataset_name}: {len(df):,} rows, {df.shape[1]} columns. "
           "Numbers are calculated by Python, then verified.")

tab_dashboard, tab_ask, tab_data = st.tabs(["Dashboard", "Ask AI", "Data"])


# ---------- Dashboard tab ----------

with tab_dashboard:
    headline = dashboard.kpis(df)
    for col, kpi in zip(st.columns(len(headline)), headline):
        col.metric(kpi["label"], fmt(kpi["value"], kpi["format"]))

    specs = dashboard.auto_charts(df)
    if not specs:
        st.info("No numeric column found, so there is nothing to chart.")
    for start in range(0, len(specs), 2):
        for col, spec in zip(st.columns(2), specs[start:start + 2]):
            with col:
                show_chart(spec, key=f"dash-{start}-{spec['title']}")

    st.subheader("Data quality")
    quality = dashboard.quality_summary(df)
    problems = []
    if quality["duplicate_rows"]:
        problems.append(f"{quality['duplicate_rows']} duplicate row(s).")
    for column, count in quality["missing_by_column"].items():
        problems.append(f"{count} missing value(s) in {column}.")
    check = quality["revenue_check"]
    if check and check["mismatched_rows"]:
        problems.append(
            f"{check['mismatched_rows']} row(s) where Revenue does not equal Quantity x UnitPrice."
        )
    if problems:
        for text in problems:
            st.warning(text)
        if check and check["mismatches"]:
            st.dataframe(pd.DataFrame(check["mismatches"]))
    else:
        st.success("No duplicates, missing values or revenue mismatches found.")


# ---------- Ask AI tab ----------

with tab_ask:
    st.caption("Examples: " + "  |  ".join(f'"{e}"' for e in EXAMPLES))
    for i, message in enumerate(st.session_state.messages):
        render_message(message, i)

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Ask a question about this dataset")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Analysing..."):
            st.session_state.messages.append(ask_agent(df, question))
        st.rerun()


# ---------- Data tab ----------

with tab_data:
    st.subheader("Preview")
    st.dataframe(df.head(100))
    st.subheader("Profile")
    st.json(profile_dataset(df), expanded=False)
