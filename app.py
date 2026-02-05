"""Streamlit UI for aspect-level opinion mining demo."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.aspect_miner import AspectOpinionPipeline

st.set_page_config(page_title="Aspect-Level Opinion Mining", page_icon="🧠", layout="wide")

st.title("🧠 Aspect-Level Opinion Mining")
st.caption("Extract product aspects and sentiment in a transparent, recruiter-friendly way.")

with st.expander("What this app does", expanded=False):
    st.markdown(
        """
        - Splits text into sentences and extracts noun-based aspects
        - Detects opinion words with lexicon + negation/intensifier rules
        - Associates opinions to aspects by interpretable proximity scoring
        - Returns table + JSON for easy consumption
        """
    )

example = (
    "The battery life is amazing, but the screen quality is not good. "
    "Camera is very sharp, though the speaker is slightly weak."
)

text = st.text_area("Paste a review", value=example, height=170)

if st.button("Analyze Review", type="primary"):
    if not text.strip():
        st.warning("Please enter review text before analyzing.")
    else:
        with st.spinner("Running NLP pipeline..."):
            pipeline = AspectOpinionPipeline()
            output = pipeline.analyze(text)

        st.subheader("Aspect Sentiments")
        if output["aspects"]:
            df = pd.DataFrame(output["aspects"])
            st.dataframe(df[["aspect", "sentiment", "score", "evidence"]], use_container_width=True)
        else:
            st.info("No clear aspects were detected.")

        st.subheader("Structured JSON Output")
        st.code(json.dumps(output, indent=2), language="json")
