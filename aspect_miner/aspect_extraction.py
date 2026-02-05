"""Extract noun-based aspects, including multi-word compounds."""

from __future__ import annotations

from typing import List, Set


class AspectExtractor:
    GENERIC_TERMS = {
        "thing",
        "stuff",
        "product",
        "item",
        "something",
        "everything",
        "anything",
    }

    def extract(self, sent) -> List[str]:
        aspects: Set[str] = set()

        # Noun chunks are strong high-precision aspect candidates.
        for chunk in sent.noun_chunks:
            normalized = " ".join(tok.lemma_.lower() for tok in chunk if tok.is_alpha and not tok.is_stop)
            if normalized and normalized not in self.GENERIC_TERMS:
                aspects.add(normalized)

        # Extra backup rule for compounds not captured well by chunks in edge cases.
        for token in sent:
            if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha:
                compounds = [child.lemma_.lower() for child in token.children if child.dep_ == "compound"]
                phrase = " ".join(compounds + [token.lemma_.lower()]).strip()
                if phrase and phrase not in self.GENERIC_TERMS:
                    aspects.add(phrase)

        return sorted(aspects)
