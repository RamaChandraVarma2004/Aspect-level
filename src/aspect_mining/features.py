from __future__ import annotations

from spacy.tokens import Span, Token, Doc

from .lexicon import SENTIMENT_LEXICON


class LinguisticFeatureExtractor:
    """Extracts explainable linguistic signals used by later components."""

    def opinion_tokens(self, sentence: Span) -> list[Token]:
        opinions: list[Token] = []
        for tok in sentence:
            lemma = tok.lemma_.lower()
            # POS catches unknown words, lexicon ensures interpretable polarity.
            if tok.pos_ in {"ADJ", "VERB"} and lemma in SENTIMENT_LEXICON:
                opinions.append(tok)
        return opinions

    def noun_chunks(self, doc: Doc) -> list[Span]:
        return list(doc.noun_chunks)
