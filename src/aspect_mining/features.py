from __future__ import annotations

from spacy.tokens import Span, Token, Doc

from .lexicon import SENTIMENT_LEXICON


class LinguisticFeatureExtractor:
    """Extracts explainable linguistic signals used by later components."""

    def opinion_tokens(self, sentence: Span) -> list[Token]:
        opinions: list[Token] = []
        for tok in sentence:
            lemma = (tok.lemma_ or tok.text).lower()
            pos_ok = tok.pos_ in {"ADJ", "VERB"} or (not tok.pos_ and lemma in SENTIMENT_LEXICON)
            if pos_ok and lemma in SENTIMENT_LEXICON:
                opinions.append(tok)
        return opinions

    def noun_chunks(self, doc: Doc) -> list[Span]:
        try:
            return list(doc.noun_chunks)
        except (ValueError, NotImplementedError):
            return []
