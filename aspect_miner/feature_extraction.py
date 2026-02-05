"""Linguistic feature extraction for explainable ABSA-like rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SentenceFeatures:
    sentence_text: str
    noun_chunks: List[str]
    candidate_opinion_tokens: List[str]


class FeatureExtractor:
    """Extract POS-informed features from spaCy parsed sentences."""

    OPINION_POS = {"ADJ", "ADV", "VERB"}

    def extract(self, sent) -> SentenceFeatures:
        noun_chunks = [chunk.text.strip().lower() for chunk in sent.noun_chunks]
        candidate_opinion_tokens = [
            token.text.lower()
            for token in sent
            if token.pos_ in self.OPINION_POS and not token.is_stop and token.is_alpha
        ]

        return SentenceFeatures(
            sentence_text=sent.text.strip(),
            noun_chunks=noun_chunks,
            candidate_opinion_tokens=candidate_opinion_tokens,
        )
