from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .aspect_extractor import AspectExtractor
from .association import AspectOpinionAssociator
from .preprocess import TextPreprocessor


@dataclass
class ReviewAnalysis:
    """Container for one review's extracted aspect-level sentiment rows."""

    review_id: int
    review_text: str
    aspects: list[dict]


class AspectOpinionMiner:
    """Explainable orchestrator for aspect-level opinion mining.

    This class is intentionally lightweight and modular: each method maps to a
    clear step in a rule-first NLP pipeline so interns can explain the flow.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.associator = AspectOpinionAssociator()

    def analyze(self, text: str) -> list[dict]:
        """Analyze a single review and return aspect-level results."""
        doc = self.preprocessor.process(text)
        aspects = self.aspect_extractor.extract(doc)
        aspect_sentiments = self.associator.associate(aspects)
        return [item.to_dict() for item in aspect_sentiments]

    def analyze_reviews(self, reviews: list[str]) -> list[ReviewAnalysis]:
        """Analyze many reviews while preserving per-review traceability."""
        clean_reviews = [r.strip() for r in reviews if r and r.strip()]
        output: list[ReviewAnalysis] = []
        for idx, review in enumerate(clean_reviews, start=1):
            output.append(ReviewAnalysis(review_id=idx, review_text=review, aspects=self.analyze(review)))
        return output

    def aggregate_aspects(self, analyses: list[ReviewAnalysis]) -> list[dict]:
        """Aggregate aspect sentiment counts and compute dominant sentiment."""
        bucket: dict[str, dict] = defaultdict(
            lambda: {
                "aspect": "",
                "frequency": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "avg_score": 0.0,
                "_scores": [],
            }
        )

        for review in analyses:
            for row in review.aspects:
                key = row["aspect"].lower()
                rec = bucket[key]
                rec["aspect"] = row["aspect"]
                rec["frequency"] += 1
                rec[row["sentiment"]] += 1
                rec["_scores"].append(row["score"])

        results: list[dict] = []
        for rec in bucket.values():
            rec["avg_score"] = round(sum(rec["_scores"]) / len(rec["_scores"]), 3)
            sentiment_counts = {"positive": rec["positive"], "negative": rec["negative"], "neutral": rec["neutral"]}
            rec["dominant_sentiment"] = max(sentiment_counts, key=sentiment_counts.get)
            rec.pop("_scores", None)
            results.append(rec)

        return sorted(results, key=lambda x: (-x["frequency"], x["aspect"].lower()))
