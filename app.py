from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining import AspectOpinionMiner, analyze_all_versions

st.set_page_config(page_title="Aspect-Level Opinion Mining", page_icon="🧠", layout="wide")

st.title("🧠 Aspect-Level Opinion Mining — 4 Recruiter-Friendly Versions")
st.caption("Intern-level, explainable ABSA with four progressively stronger approaches.")

version_docs = {
    "v1": "**V1 Proximity Rules:** nearby opinion words per aspect.",
    "v2": "**V2 Dependency-Aware:** prefers syntactic links before distance fallback.",
    "v3": "**V3 Contrast-Aware:** handles *but/however/though* style clause shifts.",
    "v4": "**V4 Ensemble (Recommended):** averages V1+V2+V3 for stable output.",
}

with st.expander("Approach overview"):
    for line in version_docs.values():
        st.markdown(f"- {line}")

sample = (
    "The battery life is amazing, but the camera quality is not good. "
    "I love the display, though the speakers are very weak."
)
text = st.text_area("Paste a review", value=sample, height=180)

mode = st.radio(
    "Analysis mode",
    options=["Run recommended version (V4)", "Compare all 4 versions"],
    horizontal=True,
)


def render_rows(rows: list[dict], title: str):
    st.subheader(title)
    if not rows:
        st.warning("No clear aspects found. Try a more descriptive review.")
        return

    df = pd.DataFrame(
        {
            "Aspect": [r["aspect"] for r in rows],
            "Sentiment": [r["sentiment"].title() for r in rows],
            "Score": [r["score"] for r in rows],
            "Sentence": [r["sentence"] for r in rows],
            "Evidence Words": [", ".join(ev["word"] for ev in r["evidences"]) for r in rows],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.code(json.dumps(rows, indent=2), language="json")


if st.button("Analyze review", type="primary"):
    if mode == "Run recommended version (V4)":
        rows = AspectOpinionMiner(version="v4").analyze(text)
        render_rows(rows, "V4 Ensemble Output")
    else:
        all_outputs = analyze_all_versions(text)
        cols = st.columns(2)
        versions = ["v1", "v2", "v3", "v4"]
        for idx, version in enumerate(versions):
            with cols[idx % 2]:
                st.markdown(version_docs[version])
                render_rows(all_outputs[version], f"{version.upper()} Output")
