from __future__ import annotations

from collections import OrderedDict
from spacy.tokens import Doc, Span, Token

from .lexicon import GENERIC_ASPECTS, SENTIMENT_LEXICON


class AspectExtractor:
    """Extract noun-based aspects with multi-word support."""

    def extract(self, doc: Doc) -> list[Span]:
        aspects: "OrderedDict[tuple[int, int], Span]" = OrderedDict()

        try:
            noun_chunks = list(doc.noun_chunks)
        except (ValueError, NotImplementedError):
            noun_chunks = []

        for chunk in noun_chunks:
            cleaned = self._clean_chunk(chunk)
            if cleaned:
                aspects[(cleaned.start, cleaned.end)] = cleaned

        for tok in doc:
            if self._is_candidate_noun(tok):
                span = doc[tok.i : tok.i + 1]
                cleaned = self._clean_chunk(span)
                if cleaned:
                    aspects[(cleaned.start, cleaned.end)] = cleaned

            if tok.i < len(doc) - 1:
                span = doc[tok.i : tok.i + 2]
                cleaned = self._clean_chunk(span)
                if cleaned and len(cleaned) > 1:
                    aspects[(cleaned.start, cleaned.end)] = cleaned

        return list(aspects.values())

    def _is_candidate_noun(self, tok: Token) -> bool:
        if tok.is_stop or tok.is_punct or not tok.is_alpha:
            return False
        if tok.pos_:
            return tok.pos_ == "NOUN" and tok.dep_ in {"nsubj", "dobj", "pobj", "attr", ""}
        # Fallback heuristic when POS/dependency tags are unavailable.
        lower = tok.text.lower()
        return lower not in SENTIMENT_LEXICON and len(lower) > 2

    def _clean_chunk(self, span: Span) -> Span | None:
        doc = span.doc
        content_tokens: list[Token] = [
            tok
            for tok in span
            if not tok.is_stop
            and not tok.is_punct
            and tok.is_alpha
            and (tok.pos_ in {"NOUN", "PROPN", "ADJ", ""})
            and tok.text.lower() not in SENTIMENT_LEXICON
        ]
        if not content_tokens:
            return None

        start = content_tokens[0].i
        end = content_tokens[-1].i + 1
        cleaned = doc[start:end]

        head_lemma = (cleaned.root.lemma_ or cleaned.root.text).lower()
        if head_lemma in GENERIC_ASPECTS:
            return None

        if len(cleaned.text) <= 2:
            return None

        return cleaned
