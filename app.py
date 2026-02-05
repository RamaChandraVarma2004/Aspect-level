"""Streamlit UI for aspect-level opinion mining."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.aspect_mining import AspectMiningPipeline

st.set_page_config(page_title="Aspect-Level Opinion Mining", page_icon="💬", layout="wide")

st.title("💬 Aspect-Level Opinion Mining")
st.caption("Explainable NLP pipeline for extracting product aspects and their sentiment.")

pipeline = AspectMiningPipeline()

example_text = (
    "The battery life is amazing, but the screen quality is not good in sunlight. "
    "I love the camera, although the charging speed is very slow."
)

review_text = st.text_area(
    "Paste a user review",
    value=example_text,
    height=180,
    help="Tip: Add sentences with multiple aspects to see per-aspect sentiment.",
)

if st.button("Analyze Review", type="primary"):
    results = pipeline.analyze(review_text)

    if not results:
        st.warning("No clear aspect-opinion pairs found. Try a more descriptive review.")
    else:
        rows = [
            {
                "Aspect": r.aspect,
                "Sentiment": r.sentiment,
                "Score": r.score,
                "Opinion Words": ", ".join(r.opinion_words) if r.opinion_words else "—",
                "Sentence": r.sentence,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)
        st.subheader("Per-Aspect Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Structured JSON Output")
        st.code(json.dumps([r.to_dict() for r in results], indent=2), language="json")

with st.expander("Design Notes (for internship review)"):
    st.markdown(
        """
- **Explainable pipeline**: sentence split → POS tagging → noun-phrase aspect extraction → opinion scoring.
- **Interpretability first**: nearest-aspect association and explicit rules for negation/intensifiers.
- **Practical tradeoff**: lightweight rules are fast and clear, but may miss implicit sentiments.
        """
    )
