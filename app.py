from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining import AspectOpinionMiner, AVAILABLE_VERSIONS

st.set_page_config(page_title="Aspect-Level Opinion Mining - 4 Versions", page_icon="🧠", layout="wide")
st.title("🧠 Aspect-Level Opinion Mining (4 Recruiter-Ready Versions)")
st.caption("Pick one approach or compare all four. Each output includes evidence for explainability.")

sample = (
    "The battery life is amazing, but the camera quality is not good. "
    "I love the screen, though the speakers are very weak and charging speed is decent."
)

col1, col2 = st.columns([2, 1])
with col1:
    text = st.text_area("Paste a review", value=sample, height=180)
with col2:
    mode = st.selectbox("Choose analysis mode", ["single", "compare-all"], index=1)
    version = st.selectbox("Single version", list(AVAILABLE_VERSIONS.keys()), index=3)

st.markdown("### Approach Summary")
for k, v in AVAILABLE_VERSIONS.items():
    st.markdown(f"- **{k.upper()}**: {v}")


def render_rows(rows: list[dict], title: str):
    st.subheader(title)
    if not rows:
        st.warning("No aspects found.")
        return
    df = pd.DataFrame(
        {
            "Aspect": [r["aspect"] for r in rows],
            "Sentiment": [r["sentiment"].title() for r in rows],
            "Score": [r["score"] for r in rows],
            "Approach": [r.get("approach", "-") for r in rows],
            "Evidence": [", ".join(e["word"] for e in r.get("evidences", [])) for r in rows],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.code(json.dumps(rows, indent=2), language="json")


if st.button("Analyze", type="primary"):
    if mode == "single":
        try:
            rows = AspectOpinionMiner(version=version).analyze(text)
            render_rows(rows, f"Results ({version.upper()})")
        except Exception as e:
            st.error(f"{version.upper()} unavailable in this environment: {e}")
    else:
        tabs = st.tabs([x.upper() for x in AVAILABLE_VERSIONS.keys()])
        for i, ver in enumerate(AVAILABLE_VERSIONS):
            with tabs[i]:
                try:
                    rows = AspectOpinionMiner(version=ver).analyze(text)
                    render_rows(rows, f"Results ({ver.upper()})")
                except Exception as e:
                    st.error(f"{ver.upper()} unavailable: {e}")
