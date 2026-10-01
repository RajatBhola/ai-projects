"""Streamlit front end:  streamlit run app.py"""

from __future__ import annotations

import sys
from numbers import Number
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nl2sql.config import Settings  # noqa: E402
from nl2sql.database import Database  # noqa: E402
from nl2sql.llm import OpenAIClient  # noqa: E402
from nl2sql.pipeline import NL2SQL  # noqa: E402
from nl2sql.semantic import SemanticLayer  # noqa: E402

EXAMPLES = [
    "What was our total revenue in 2025?",
    "Show revenue by product category, highest first.",
    "How many orders were placed each month in 2025?",
    "Who are our top 5 customers by total revenue?",
    "What is the return rate per sales channel?",
]

st.set_page_config(page_title="Ask your data", page_icon=":bar_chart:", layout="wide")
settings = Settings.from_env()


@st.cache_resource
def get_pipeline() -> NL2SQL:
    db = Database(settings.db_path)
    semantic = SemanticLayer.load(settings.semantic_layer_path)
    return NL2SQL(db, OpenAIClient(model=settings.openai_model), semantic,
                  max_rows=settings.max_rows, max_attempts=settings.max_attempts)


st.title("Ask your data")
st.caption(f"Plain-English questions over a sample e-commerce warehouse · model: {settings.openai_model}")

if not settings.db_path.exists():
    st.error(f"No database at {settings.db_path}. Run `nl2sql seed` first.")
    st.stop()

with st.sidebar:
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

question = st.text_input("Your question", key="question", placeholder="e.g. Which country had the most revenue last year?")
if not question:
    st.stop()

with st.spinner("Writing and checking SQL..."):
    try:
        answer = get_pipeline().ask(question)
    except Exception as e:  # noqa: BLE001 - surface API/config errors to the user
        st.error(f"Something went wrong: {e}")
        st.stop()

if not answer.ok:
    st.warning(answer.error)
else:
    st.write(answer.explanation)
    for a in answer.assumptions:
        st.info(f"Assumption: {a}")
    df = pd.DataFrame(answer.rows, columns=answer.columns)
    st.dataframe(df, use_container_width=True)
    numeric = len(answer.columns) == 2 and answer.rows and isinstance(answer.rows[0][1], Number)
    if numeric and 1 < len(answer.rows) <= 50:
        st.bar_chart(df.astype({answer.columns[1]: float}).set_index(answer.columns[0]))
    if answer.limit_applied:
        st.caption(f"Results limited to {settings.max_rows} rows.")

with st.expander("SQL and validation details"):
    if answer.sql:
        st.code(answer.sql, language="sql")
    st.write(f"Tables given to the model: {', '.join(answer.tables_in_context)}")
    if answer.matched_terms:
        st.write(f"Business definitions applied: {', '.join(answer.matched_terms)}")
    for i, att in enumerate(answer.attempts, 1):
        if att.error:
            st.write(f"Attempt {i} rejected at **{att.stage}**: {att.error}")
            st.code(att.sql, language="sql")
    st.write(f"{len(answer.attempts)} attempt(s) · {answer.seconds}s · "
             f"{answer.input_tokens + answer.output_tokens} tokens")
