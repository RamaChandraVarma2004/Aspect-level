from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining.approaches import APPROACHES
from src.aspect_mining.pipeline import AspectOpinionMiner, analyze_with_all_versions

st.set_page_config(page_title="Aspect Opinion Mining - 4 Versions", page_icon="🧠", layout="wide")

st.title("🧠 Aspect-Level Opinion Mining (4 Recruiter-Friendly Versions)")
st.caption("Intern-level, modular, explainable pipeline with 4 different approaches.")

st.markdown("### Choose one version or compare all")
selected = st.selectbox("Approach", list(APPROACHES.keys()), index=3)
st.info(APPROACHES[selected]().description)

sample = (
    "Battery life is excellent, but camera quality is not good. "
    "The screen is amazing although the speakers are very weak."
)
text = st.text_area("Paste review", value=sample, height=180)

col1, col2 = st.columns(2)
run_single = col1.button("Run selected version", type="primary")
run_all = col2.button("Run all 4 versions")


def render_rows(rows: list[dict], title: str):
    st.subheader(title)
    if not rows:
        st.warning("No aspects found.")
        return
    df = pd.DataFrame(
        {
            "Aspect": [r["aspect"] for r in rows],
            "Sentiment": [r["sentiment"] for r in rows],
            "Score": [r["score"] for r in rows],
            "Approach": [r["approach"] for r in rows],
            "Evidence": [", ".join(e["word"] for e in r["evidences"]) for r in rows],
            "Sentence": [r["sentence"] for r in rows],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.code(json.dumps(rows, indent=2), language="json")


if run_single:
    rows = AspectOpinionMiner(approach=selected).analyze(text)
    render_rows(rows, f"Output: {selected}")

if run_all:
    all_outputs = analyze_with_all_versions(text)
    for k, rows in all_outputs.items():
        render_rows(rows, f"Output: {k}")
