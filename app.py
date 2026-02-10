from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.aspect_mining import AspectOpinionMiner
from src.aspect_mining.variants import run_variant

st.set_page_config(page_title="Aspect-Level Opinion Mining Lab", page_icon="🧪", layout="wide")

SAMPLE_REVIEWS = [
    "Battery life is excellent and charging speed is fast, but the camera is disappointing in low light.",
    "The screen quality is amazing. Speakers are weak and the phone feels heavy.",
    "I love the design and display, however the software experience is not smooth.",
    "Keyboard is decent, trackpad is great, but build quality is not reliable.",
]

st.title("🧪 Aspect-Level Opinion Mining Lab")
st.caption("Four explainable pipeline versions with intentionally different outputs and presentation styles.")

with st.sidebar:
    st.header("Input")
    if st.button("Load sample reviews"):
        st.session_state["review_input"] = "\n".join(SAMPLE_REVIEWS)

    review_blob = st.text_area(
        "Enter one review per line",
        key="review_input",
        height=240,
        placeholder="Review 1...\nReview 2...\nReview 3...",
    )

    st.markdown("---")
    selected_versions = st.multiselect(
        "Choose versions to run simultaneously",
        ["v1", "v2", "v3", "v4"],
        default=["v1", "v2", "v3", "v4"],
        help="Each version applies a different explainable rule profile.",
    )

analyze = st.button("Analyze Reviews", type="primary", use_container_width=True)

if analyze:
    reviews = [line.strip() for line in review_blob.splitlines() if line.strip()]
    if not reviews:
        st.warning("Please provide at least one review line.")
        st.stop()

    miner = AspectOpinionMiner()

    version_results = {v: run_variant(miner, reviews, v) for v in selected_versions}
    if not version_results:
        st.warning("Select at least one version.")
        st.stop()

    tabs = st.tabs([payload["name"] for payload in version_results.values()])

    for tab, (version_key, payload) in zip(tabs, version_results.items()):
        with tab:
            st.subheader(payload["name"])

            # Shared per-review flat table.
            per_review_rows = []
            for review in payload["reviews"]:
                for row in review.aspects:
                    per_review_rows.append(
                        {
                            "review_id": review.review_id,
                            "aspect": row["aspect"],
                            "sentiment": row["sentiment"],
                            "score": row["score"],
                            "evidence": ", ".join(ev["word"] for ev in row["evidences"]),
                        }
                    )

            if per_review_rows:
                df = pd.DataFrame(per_review_rows)
            else:
                df = pd.DataFrame(columns=["review_id", "aspect", "sentiment", "score", "evidence"])

            agg_df = pd.DataFrame(payload["aggregated"])

            if version_key == "v1":
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("#### Per-review extraction table")
                    st.dataframe(df, use_container_width=True)
                with c2:
                    st.markdown("#### Most discussed aspects")
                    if not agg_df.empty:
                        st.bar_chart(agg_df.set_index("aspect")["frequency"])
                st.markdown("#### Aggregated aspect sentiment")
                st.dataframe(agg_df, use_container_width=True)

            elif version_key == "v2":
                st.info("Precision-first output: only multi-word aspects and the nearest opinion evidence are kept.")
                st.markdown("#### Analyst cards")
                for _, row in agg_df.iterrows():
                    with st.container(border=True):
                        st.write(
                            f"**{row['aspect']}** → dominant: `{row['dominant_sentiment']}` | "
                            f"mentions: `{int(row['frequency'])}` | avg score: `{row['avg_score']}`"
                        )
                        st.progress(min(1.0, float(row["positive"]) / max(1.0, float(row["frequency"]))), text="Positive ratio")
                with st.expander("Raw extraction rows"):
                    st.dataframe(df, use_container_width=True)

            elif version_key == "v3":
                st.success("Recall-focused output: repeated aspects amplify sentiment strength.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total aspect mentions", int(df.shape[0]))
                m2.metric("Unique aspects", int(agg_df.shape[0]))
                m3.metric("Top aspect", agg_df.iloc[0]["aspect"] if not agg_df.empty else "N/A")

                if not agg_df.empty:
                    melt = agg_df[["aspect", "positive", "negative", "neutral"]].set_index("aspect")
                    st.area_chart(melt)
                st.markdown("#### Weighted sentiment table")
                st.dataframe(df, use_container_width=True)

            else:
                st.warning("Contrast-aware briefing: highlights trade-off-heavy reviews.")
                for review in payload["reviews"]:
                    with st.container(border=True):
                        st.markdown(f"**Review {review.review_id}**: {review.review_text}")
                        r_df = pd.DataFrame(review.aspects)
                        if r_df.empty:
                            st.write("No aspects found")
                            continue
                        pos = int((r_df["sentiment"] == "positive").sum())
                        neg = int((r_df["sentiment"] == "negative").sum())
                        neu = int((r_df["sentiment"] == "neutral").sum())
                        st.write(f"Sentiment mix → ✅ {pos} | ❌ {neg} | ⚪ {neu}")
                        st.dataframe(r_df[["aspect", "sentiment", "score", "sentence"]], use_container_width=True)

                st.markdown("#### JSON briefing output")
                st.code(json.dumps(payload["aggregated"], indent=2), language="json")
