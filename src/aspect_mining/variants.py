from __future__ import annotations

from collections import defaultdict

from .pipeline import AspectOpinionMiner, ReviewAnalysis


def _dominant(counts: dict[str, int]) -> str:
    return max(counts, key=counts.get)


def run_variant(miner: AspectOpinionMiner, reviews: list[str], variant: str) -> dict:
    """Execute one of four explainable rule profiles.

    Each profile intentionally emphasizes different analysis behavior so students
    can compare precision/recall trade-offs in interviews and demos.
    """

    analyses = miner.analyze_reviews(reviews)

    if variant == "v1":
        aggregated = miner.aggregate_aspects(analyses)
        return {"name": "Version 1 - Balanced Rule Pipeline", "reviews": analyses, "aggregated": aggregated}

    if variant == "v2":
        transformed = _conservative_nearest_opinion(analyses)
        aggregated = _aggregate_generic(transformed)
        return {"name": "Version 2 - Conservative Precision Mode", "reviews": transformed, "aggregated": aggregated}

    if variant == "v3":
        transformed = _recall_boost_with_frequency_weight(analyses)
        aggregated = _aggregate_generic(transformed)
        return {"name": "Version 3 - Recall + Strength Emphasis", "reviews": transformed, "aggregated": aggregated}

    transformed = _contrast_mode(analyses)
    aggregated = _aggregate_generic(transformed)
    return {"name": "Version 4 - Contrast-Aware Review Briefing", "reviews": transformed, "aggregated": aggregated}


def _conservative_nearest_opinion(analyses: list[ReviewAnalysis]) -> list[ReviewAnalysis]:
    out: list[ReviewAnalysis] = []
    for item in analyses:
        rows = []
        for row in item.aspects:
            if len(row["aspect"].split()) == 1:
                continue
            if row["evidences"]:
                best = sorted(row["evidences"], key=lambda e: e["distance"])[0]
                score = best["adjusted_score"]
                sentiment = "positive" if score > 0.4 else "negative" if score < -0.4 else "neutral"
                rows.append({**row, "score": round(score, 3), "sentiment": sentiment, "evidences": [best]})
            else:
                rows.append(row)
        out.append(ReviewAnalysis(review_id=item.review_id, review_text=item.review_text, aspects=rows))
    return out


def _recall_boost_with_frequency_weight(analyses: list[ReviewAnalysis]) -> list[ReviewAnalysis]:
    out: list[ReviewAnalysis] = []
    for item in analyses:
        counts = defaultdict(int)
        for row in item.aspects:
            counts[row["aspect"].lower()] += 1

        rows = []
        for row in item.aspects:
            repeat_boost = min(1.3, 1.0 + (counts[row["aspect"].lower()] - 1) * 0.15)
            score = round(row["score"] * repeat_boost, 3)
            sentiment = "positive" if score > 0.35 else "negative" if score < -0.35 else "neutral"
            rows.append({**row, "score": score, "sentiment": sentiment, "repeat_boost": round(repeat_boost, 2)})
        out.append(ReviewAnalysis(review_id=item.review_id, review_text=item.review_text, aspects=rows))
    return out


def _contrast_mode(analyses: list[ReviewAnalysis]) -> list[ReviewAnalysis]:
    markers = {"but", "however", "though", "although", "yet"}
    out: list[ReviewAnalysis] = []
    for item in analyses:
        rows = []
        lower = item.review_text.lower()
        is_contrastive = any(f" {m} " in lower for m in markers)
        for row in item.aspects:
            score = row["score"]
            if is_contrastive and row["sentence"].lower().strip().startswith(("however", "though", "but")):
                score = round(score * 1.2, 3)
            sentiment = "positive" if score > 0.45 else "negative" if score < -0.45 else "neutral"
            rows.append({**row, "score": score, "sentiment": sentiment, "contrastive_review": is_contrastive})
        out.append(ReviewAnalysis(review_id=item.review_id, review_text=item.review_text, aspects=rows))
    return out


def _aggregate_generic(analyses: list[ReviewAnalysis]) -> list[dict]:
    bucket = defaultdict(lambda: {"aspect": "", "frequency": 0, "positive": 0, "negative": 0, "neutral": 0, "_scores": []})
    for review in analyses:
        for row in review.aspects:
            key = row["aspect"].lower()
            rec = bucket[key]
            rec["aspect"] = row["aspect"]
            rec["frequency"] += 1
            rec[row["sentiment"]] += 1
            rec["_scores"].append(row["score"])

    result = []
    for rec in bucket.values():
        counts = {"positive": rec["positive"], "negative": rec["negative"], "neutral": rec["neutral"]}
        result.append(
            {
                "aspect": rec["aspect"],
                "frequency": rec["frequency"],
                "positive": rec["positive"],
                "negative": rec["negative"],
                "neutral": rec["neutral"],
                "avg_score": round(sum(rec["_scores"]) / len(rec["_scores"]), 3),
                "dominant_sentiment": _dominant(counts),
            }
        )
    return sorted(result, key=lambda x: (-x["frequency"], x["aspect"].lower()))
