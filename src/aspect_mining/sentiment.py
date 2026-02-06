from __future__ import annotations

from .lexicon import INTENSIFIERS, NEGATIONS, SENTIMENT_LEXICON
from .schemas import OpinionEvidence


class SentimentScorer:
    """Rule-based sentiment scoring with negation and intensity handling."""

    def score_opinion(self, sentence, opinion_tok, aspect_center: int) -> OpinionEvidence:
        lemma = opinion_tok.lemma_.lower()
        base = SENTIMENT_LEXICON.get(lemma, 0.0)
        adjusted = base

        window = sentence[max(sentence.start, opinion_tok.i - 3) : opinion_tok.i]
        window_words = [tok.text.lower() for tok in window]

        negated = any(w in NEGATIONS for w in window_words)
        intensifier = next((w for w in reversed(window_words) if w in INTENSIFIERS), None)

        if intensifier:
            adjusted *= INTENSIFIERS[intensifier]
        if negated:
            adjusted *= -1

        distance = abs(opinion_tok.i - aspect_center)
        distance_penalty = max(0.35, 1 - distance * 0.08)
        adjusted *= distance_penalty

        return OpinionEvidence(
            word=opinion_tok.text,
            base_score=base,
            adjusted_score=round(adjusted, 3),
            negated=negated,
            intensifier=intensifier,
            distance=distance,
        )

    @staticmethod
    def label(score: float) -> str:
        if score > 0.4:
            return "positive"
        if score < -0.4:
            return "negative"
        return "neutral"
