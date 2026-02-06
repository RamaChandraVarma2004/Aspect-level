from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining import AspectOpinionMiner, analyze_with_all_versions

st.set_page_config(page_title="Aspect-Level Opinion Mining", page_icon="🧠", layout="wide")

st.title("🧠 Aspect-Level Opinion Mining")
st.caption("4 internship-friendly, explainable versions for recruiter demos.")

version_meta = {
    "v1": "Proximity heuristic baseline",
    "v2": "Dependency-aware association (recommended)",
    "v3": "Hybrid sentence+local scoring",
    "v4": "Clause-aware contrastive scoring",
}

with st.expander("What are the 4 versions?"):
    st.markdown(
        """
        - **V1**: Simple proximity rules (fast baseline).
        - **V2**: Dependency-based linking between aspects and opinion words (**best default**).
        - **V3**: Hybrid of sentence-level tone + local aspect evidence.
        - **V4**: Contrast-aware approach for "but/however/though" style sentences.
        """
    )

sample = (
    "The battery life is amazing, but the camera quality is not good. "
    "I love the screen, though the speakers are very weak."
)

text = st.text_area("Paste a review", value=sample, height=180)
col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox(
        "Choose version",
        ["v1", "v2", "v3", "v4"],
        index=1,
        format_func=lambda x: f"{x.upper()} — {version_meta[x]}",
    )
with col2:
    compare_all = st.checkbox("Run all 4 versions", value=False)

if st.button("Analyze review", type="primary"):
    if compare_all:
        outputs = analyze_with_all_versions(text)
        tabs = st.tabs([f"{v.upper()}" for v in outputs])
        for tab, (version, rows) in zip(tabs, outputs.items()):
            with tab:
                if not rows:
                    st.warning("No clear aspects found.")
                    continue
                df = pd.DataFrame(
                    {
                        "Aspect": [r["aspect"] for r in rows],
                        "Sentiment": [r["sentiment"].title() for r in rows],
                        "Score": [r["score"] for r in rows],
                        "Confidence": [r["confidence"] for r in rows],
                        "Method": [r["method"] for r in rows],
                    }
                )
                st.dataframe(df, use_container_width=True)
                st.code(json.dumps(rows, indent=2), language="json")
    else:
        miner = AspectOpinionMiner(version=selected)
        rows = miner.analyze(text)

        if not rows:
            st.warning("No clear aspects found. Try a more descriptive review.")
        else:
            df = pd.DataFrame(
                {
                    "Aspect": [r["aspect"] for r in rows],
                    "Sentiment": [r["sentiment"].title() for r in rows],
                    "Score": [r["score"] for r in rows],
                    "Confidence": [r["confidence"] for r in rows],
                    "Sentence": [r["sentence"] for r in rows],
                    "Evidence Words": [", ".join(ev["word"] for ev in r["evidences"]) for r in rows],
                }
            )
            st.subheader("Results Table")
            st.dataframe(df, use_container_width=True)

            st.subheader("Per-aspect Explanation")
            for r in rows:
                with st.container(border=True):
                    st.markdown(f"**Aspect:** `{r['aspect']}`")
                    st.markdown(f"**Sentiment:** `{r['sentiment']}` | **Score:** `{r['score']}` | **Confidence:** `{r['confidence']}`")
                    if r["evidences"]:
                        for ev in r["evidences"]:
                            extra = []
                            if ev["negated"]:
                                extra.append("negated")
                            if ev["intensifier"]:
                                extra.append(f"intensifier={ev['intensifier']}")
                            details = f" ({', '.join(extra)})" if extra else ""
                            st.markdown(
                                f"- `{ev['word']}` -> base `{ev['base_score']}`, adjusted `{ev['adjusted_score']}`{details}"
                            )
                    else:
                        st.markdown("- No nearby opinion evidence; labeled neutral.")

            st.subheader("Structured JSON")
            st.code(json.dumps(rows, indent=2), language="json")
