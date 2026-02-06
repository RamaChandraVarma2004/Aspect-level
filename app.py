from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining import AspectOpinionMiner

st.set_page_config(page_title="Aspect-Level Opinion Mining", page_icon="🧠", layout="wide")

st.title("🧠 Aspect-Level Opinion Mining")
st.caption("Rule-based, interpretable NLP pipeline for internship-ready demos.")

with st.expander("How this works"):
    st.markdown(
        """
        1. **Preprocessing:** sentence split, tokenization, lowercasing, stopwords, lemmatization via spaCy.
        2. **Aspect extraction:** noun chunks and useful nouns (supports multi-word aspects).
        3. **Opinion detection:** adjective/verb sentiment words.
        4. **Scoring:** lexicon polarity with negation and intensifier rules.
        5. **Association:** aspects matched to nearby opinion words in each sentence.
        """
    )

sample = (
    "The battery life is amazing, but the camera quality is not good. "
    "I love the screen, though the speakers are very weak."
)

text = st.text_area("Paste a review", value=sample, height=180)

if st.button("Analyze review", type="primary"):
    miner = AspectOpinionMiner()
    rows = miner.analyze(text)

    if not rows:
        st.warning("No clear aspects found. Try a more descriptive review.")
    else:
        df = pd.DataFrame(
            {
                "Aspect": [r["aspect"] for r in rows],
                "Sentiment": [r["sentiment"].title() for r in rows],
                "Score": [r["score"] for r in rows],
                "Sentence": [r["sentence"] for r in rows],
                "Evidence Words": [", ".join(ev["word"] for ev in r["evidences"]) for r in rows],
            }
        )

        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("Per-aspect Explanation")
        for r in rows:
            with st.container(border=True):
                st.markdown(f"**Aspect:** `{r['aspect']}`  ")
                st.markdown(f"**Sentiment:** `{r['sentiment']}` (score: `{r['score']}`)")
                if r["evidences"]:
                    st.markdown("**Evidence:**")
                    for ev in r["evidences"]:
                        extra = []
                        if ev["negated"]:
                            extra.append("negated")
                        if ev["intensifier"]:
                            extra.append(f"intensifier={ev['intensifier']}")
                        hint = f" ({', '.join(extra)})" if extra else ""
                        st.markdown(
                            f"- `{ev['word']}` → base `{ev['base_score']}`, adjusted `{ev['adjusted_score']}`{hint}"
                        )
                else:
                    st.markdown("No opinion words near this aspect; labeled neutral by rule.")

        st.subheader("Structured JSON")
        st.code(json.dumps(rows, indent=2), language="json")
