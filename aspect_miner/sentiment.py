"""Interpretable lexicon + rule-based sentiment scoring."""

from __future__ import annotations

from typing import Dict, Tuple

# Compact but useful sentiment lexicon for demo-quality explainability.
BASE_SENTIMENT: Dict[str, float] = {
    "good": 1.0,
    "great": 1.6,
    "excellent": 2.0,
    "amazing": 2.1,
    "smooth": 1.0,
    "fast": 1.0,
    "love": 1.8,
    "like": 1.0,
    "nice": 1.0,
    "clear": 0.8,
    "bad": -1.2,
    "poor": -1.5,
    "slow": -1.0,
    "laggy": -1.6,
    "terrible": -2.2,
    "awful": -2.1,
    "hate": -1.8,
    "disappointing": -1.6,
    "dim": -0.9,
    "expensive": -1.0,
    "cheap": 0.5,
    "decent": 0.5,
    "okay": 0.3,
    "average": 0.0,
    "long": 0.6,
    "short": -0.4,
}

NEGATIONS = {"not", "never", "no", "n't", "hardly"}
INTENSIFIERS = {
    "very": 1.4,
    "extremely": 1.8,
    "too": 1.5,
    "really": 1.3,
    "slightly": 0.7,
    "somewhat": 0.8,
    "quite": 1.2,
}


class SentimentScorer:
    def score_token(self, token) -> Tuple[float, str]:
        base = BASE_SENTIMENT.get(token.lemma_.lower())
        if base is None:
            return 0.0, "token not in explainable lexicon"

        score = base
        reasons = [f"base={base:+.2f}"]

        # Check a small context window on the left for negations/intensifiers.
        for left in token.doc[max(0, token.i - 3) : token.i]:
            low = left.lower_
            if low in NEGATIONS:
                score *= -1
                reasons.append(f"negation({low})")
            elif low in INTENSIFIERS:
                score *= INTENSIFIERS[low]
                reasons.append(f"intensifier({low}*{INTENSIFIERS[low]:.1f})")

        return score, ", ".join(reasons)

    @staticmethod
    def polarity_label(score: float) -> str:
        if score > 0.2:
            return "positive"
        if score < -0.2:
            return "negative"
        return "neutral"
