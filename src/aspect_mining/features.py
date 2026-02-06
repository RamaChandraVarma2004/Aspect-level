from __future__ import annotations

from spacy.tokens import Span, Token

from .lexicon import SENTIMENT_LEXICON


class LinguisticFeatureExtractor:
    """Extract explainable linguistic signals."""

    def opinion_tokens(self, sentence: Span) -> list[Token]:
        return [
            tok
            for tok in sentence
            if tok.pos_ in {"ADJ", "VERB"} and tok.lemma_.lower() in SENTIMENT_LEXICON
        ]

    def dependency_related(self, aspect_root: Token, sentence: Span) -> list[Token]:
        related: list[Token] = []
        for tok in sentence:
            if tok.lemma_.lower() not in SENTIMENT_LEXICON:
                continue
            if tok.head == aspect_root or aspect_root.head == tok:
                related.append(tok)
                continue
            if tok.dep_ in {"acomp", "amod", "advcl", "ROOT"} and tok.head == aspect_root.head:
                related.append(tok)
        return related
