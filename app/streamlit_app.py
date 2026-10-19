"""Streamlit UI.   Run from the project root:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Make `import app...` work when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.pipeline import answer_question  # noqa: E402
from app.agents.analysis_agent import build_agent  # noqa: E402
from app.services.dataset_service import DatasetError, load_dataset, profile_dataset  # noqa: E402

st.set_page_config(page_title="DataMind AI", layout="wide")
st.title("DataMind AI")
st.caption("Ask questions about your data. Numbers are calculated by Python, then verified.")


@st.cache_resource
def get_agent():
    return build_agent()


uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Upload a file to begin. Try data/sales.csv from the repo.")
    st.stop()

try:
    df = load_dataset(uploaded, filename=uploaded.name)
except DatasetError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
with right:
    st.subheader("Profile")
    st.json(profile_dataset(df), expanded=False)

question = st.text_input("Your question", placeholder="What is the total revenue?")
if question:
    with st.spinner("Analysing..."):
        try:
            verified = answer_question(df, question, agent=get_agent())
        except Exception as exc:
            st.error(f"The AI request failed: {exc}")
            st.stop()

    st.markdown(f"### Answer\n{verified.result.answer}")
    if verified.report.passed:
        st.success("Verified: every declared number matches a tool output.")
    else:
        st.error("Verification FAILED - do not trust this answer.")
    for msg in verified.report.errors:
        st.error(msg)
    for msg in verified.report.warnings:
        st.warning(msg)

    with st.expander("How this was calculated"):
        for call in verified.tool_calls:
            st.code(f"{call.name}({call.args}) -> {call.result}")
