from __future__ import annotations

from collections import OrderedDict
from spacy.tokens import Doc, Span, Token

from .lexicon import GENERIC_ASPECTS


class AspectExtractor:
    """Extract noun-based aspects with multi-word support."""

    def extract(self, doc: Doc) -> list[Span]:
        aspects: "OrderedDict[tuple[int, int], Span]" = OrderedDict()

        for chunk in doc.noun_chunks:
            cleaned = self._clean_chunk(chunk)
            if cleaned:
                aspects[(cleaned.start, cleaned.end)] = cleaned

        for tok in doc:
            if tok.pos_ == "NOUN" and tok.dep_ in {"nsubj", "dobj", "pobj", "attr"}:
                span = doc[tok.i : tok.i + 1]
                cleaned = self._clean_chunk(span)
                if cleaned:
                    aspects[(cleaned.start, cleaned.end)] = cleaned

        return list(aspects.values())

    def _clean_chunk(self, span: Span) -> Span | None:
        doc = span.doc
        content_tokens: list[Token] = [
            tok
            for tok in span
            if not tok.is_stop and not tok.is_punct and tok.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        if not content_tokens:
            return None

        start = content_tokens[0].i
        end = content_tokens[-1].i + 1
        cleaned = doc[start:end]

        head_lemma = cleaned.root.lemma_.lower()
        if head_lemma in GENERIC_ASPECTS:
            return None

        if len(cleaned.text) <= 2:
            return None

        return cleaned
