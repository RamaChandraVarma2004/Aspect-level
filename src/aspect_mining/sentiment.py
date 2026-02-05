"""Opinion word scoring with explainable negation/intensifier handling."""

from __future__ import annotations

from nltk.sentiment import SentimentIntensityAnalyzer

from .config import PipelineConfig
from .preprocessing import TokenFeatures
from .schemas import OpinionEvidence


class OpinionScorer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.vader = SentimentIntensityAnalyzer()

    def extract_opinions(self, tokens: list[TokenFeatures]) -> list[OpinionEvidence]:
        evidence: list[OpinionEvidence] = []

        for i, token in enumerate(tokens):
            if not (token.pos.startswith("JJ") or token.pos.startswith("RB") or token.pos.startswith("VB")):
                continue

            base = self.vader.lexicon.get(token.lemma, 0.0)
            if base == 0.0:
                base = self.vader.lexicon.get(token.text.lower(), 0.0)
            if base == 0.0:
                continue

            negated = self._is_negated(tokens, i)
            intensifier = self._get_intensifier(tokens, i)
            adjusted = base

            if negated:
                adjusted *= -1
            if intensifier:
                adjusted *= self.config.intensifiers[intensifier]

            evidence.append(
                OpinionEvidence(
                    word=token.text,
                    lemma=token.lemma,
                    raw_score=base,
                    adjusted_score=adjusted,
                    negated=negated,
                    intensifier=intensifier,
                    token_index=i,
                )
            )
        return evidence

    def classify(self, score: float) -> str:
        if score > self.config.neutral_margin:
            return "positive"
        if score < -self.config.neutral_margin:
            return "negative"
        return "neutral"

    def _is_negated(self, tokens: list[TokenFeatures], idx: int) -> bool:
        window = tokens[max(0, idx - 3) : idx]
        return any(t.lemma in self.config.negations for t in window)

    def _get_intensifier(self, tokens: list[TokenFeatures], idx: int) -> str | None:
        window = tokens[max(0, idx - 2) : idx]
        for t in reversed(window):
            if t.lemma in self.config.intensifiers:
                return t.lemma
        return None
