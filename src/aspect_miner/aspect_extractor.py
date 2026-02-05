"""Aspect candidate extraction and filtering."""

from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Span, Token


@dataclass
class AspectCandidate:
    text: str
    root: Token
    source: str


class AspectExtractor:
    """Extract noun-based aspects while filtering generic terms."""

    GENERIC_TERMS = {
        "thing",
        "stuff",
        "product",
        "item",
        "one",
        "anything",
        "everything",
    }

    def extract(self, sentence: Span, chunks: list[Span]) -> list[AspectCandidate]:
        candidates: list[AspectCandidate] = []
        seen: set[tuple[int, str]] = set()

        for chunk in chunks:
            cleaned_tokens = [
                tok for tok in chunk if tok.pos_ in {"NOUN", "PROPN", "ADJ"} and not tok.is_stop
            ]
            if not cleaned_tokens:
                continue
            phrase = " ".join(tok.text.lower() for tok in cleaned_tokens)
            root_lemma = chunk.root.lemma_.lower()
            if root_lemma in self.GENERIC_TERMS or phrase in self.GENERIC_TERMS:
                continue
            key = (chunk.root.i, phrase)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(AspectCandidate(text=phrase, root=chunk.root, source="noun_chunk"))

        for tok in sentence:
            if tok.pos_ not in {"NOUN", "PROPN"}:
                continue
            if tok.dep_ in {"compound", "amod"}:
                continue
            if tok.lemma_.lower() in self.GENERIC_TERMS:
                continue
            compounds = [c for c in tok.lefts if c.dep_ == "compound"]
            phrase_tokens = compounds + [tok]
            phrase = " ".join(t.text.lower() for t in phrase_tokens)
            key = (tok.i, phrase)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(AspectCandidate(text=phrase, root=tok, source="compound_rule"))

        return candidates
