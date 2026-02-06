from __future__ import annotations

from typing import TYPE_CHECKING

from .lexicon import SENTIMENT_LEXICON

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span, Token


class LinguisticFeatureExtractor:
    """Extracts explainable linguistic signals used by later components."""

    def opinion_tokens(self, sentence: "Span") -> list["Token"]:
        opinions: list["Token"] = []
        for tok in sentence:
            lemma = tok.lemma_.lower()
            if tok.pos_ in {"ADJ", "VERB"} and lemma in SENTIMENT_LEXICON:
                opinions.append(tok)
        return opinions

    def noun_chunks(self, doc: "Doc") -> list["Span"]:
        return list(doc.noun_chunks)
