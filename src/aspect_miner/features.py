"""Linguistic feature extraction for aspects and opinions."""

from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Span, Token


@dataclass
class OpinionToken:
    token: Token
    base_score: float
    final_score: float
    reason: str


class LinguisticFeatureExtractor:
    """Extract candidate aspect chunks and opinion-bearing tokens."""

    NEGATIONS = {"not", "no", "never", "n't", "hardly", "barely"}
    INTENSIFIERS = {
        "very": 1.5,
        "extremely": 1.8,
        "super": 1.6,
        "too": 1.3,
        "quite": 1.2,
        "really": 1.4,
        "slightly": 0.8,
        "somewhat": 0.8,
    }

    def __init__(self, positive_lexicon: set[str], negative_lexicon: set[str]):
        self.positive_lexicon = positive_lexicon
        self.negative_lexicon = negative_lexicon

    def extract_noun_chunks(self, sentence: Span) -> list[Span]:
        """Return noun chunks and compound-aware noun spans."""
        return [chunk for chunk in sentence.noun_chunks if chunk.root.pos_ in {"NOUN", "PROPN"}]

    def detect_opinion_tokens(self, sentence: Span) -> list[OpinionToken]:
        opinions: list[OpinionToken] = []
        for tok in sentence:
            lemma = tok.lemma_.lower()
            if tok.is_punct or tok.is_space:
                continue

            base_score = 0.0
            if lemma in self.positive_lexicon:
                base_score = 1.0
            elif lemma in self.negative_lexicon:
                base_score = -1.0

            if base_score == 0 and tok.pos_ not in {"ADJ", "ADV", "VERB"}:
                continue
            if base_score == 0:
                continue

            modifier = 1.0
            reason_parts = [f"lexicon:{lemma}={base_score:+.1f}"]

            prev = tok.nbor(-1) if tok.i > sentence.start else None
            if prev is not None and prev.lemma_.lower() in self.INTENSIFIERS:
                factor = self.INTENSIFIERS[prev.lemma_.lower()]
                modifier *= factor
                reason_parts.append(f"intensifier:{prev.text}x{factor}")

            negated = False
            if prev is not None and prev.lemma_.lower() in self.NEGATIONS:
                negated = True
            if any(child.dep_ == "neg" for child in tok.children):
                negated = True
            if negated:
                modifier *= -1
                reason_parts.append("negation_flip")

            final_score = base_score * modifier
            opinions.append(
                OpinionToken(
                    token=tok,
                    base_score=base_score,
                    final_score=final_score,
                    reason="; ".join(reason_parts),
                )
            )

        return opinions
