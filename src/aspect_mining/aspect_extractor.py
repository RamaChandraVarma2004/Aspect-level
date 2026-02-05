"""Aspect extraction rules based on noun phrases and compounds."""

from __future__ import annotations

from dataclasses import dataclass

from nltk import RegexpParser

from .config import PipelineConfig
from .preprocessing import TokenFeatures


@dataclass
class AspectCandidate:
    aspect: str
    token_indexes: list[int]


class AspectExtractor:
    """Extracts noun-driven aspects and filters generic terms."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # Adjectives + nouns captures phrases like "battery life" and "screen quality".
        self.chunker = RegexpParser(r"NP: {<JJ.*>*<NN.*>+}")

    def extract(self, tokens: list[TokenFeatures]) -> list[AspectCandidate]:
        if not tokens:
            return []

        tagged = [(t.text, t.pos) for t in tokens]
        tree = self.chunker.parse(tagged)

        candidates: list[AspectCandidate] = []
        consumed = 0
        for node in tree:
            if isinstance(node, tuple):
                consumed += 1
                continue

            leaves = node.leaves()
            length = len(leaves)
            idxs = list(range(consumed, consumed + length))
            consumed += length

            phrase_tokens = [tokens[i] for i in idxs]
            noun_lemmas = [t.lemma for t in phrase_tokens if t.pos.startswith("NN")]
            if not noun_lemmas:
                continue

            aspect_text = " ".join(t.lemma for t in phrase_tokens if t.text.isalpha())
            aspect_text = " ".join(aspect_text.split())
            if not self._is_informative(aspect_text, noun_lemmas):
                continue

            candidates.append(AspectCandidate(aspect=aspect_text, token_indexes=idxs))

        return self._deduplicate(candidates)

    def _is_informative(self, aspect_text: str, noun_lemmas: list[str]) -> bool:
        if not aspect_text or len(aspect_text) < self.config.min_aspect_len:
            return False
        words = aspect_text.split()
        if len(words) > self.config.max_aspect_words:
            return False
        return not all(noun in self.config.generic_nouns for noun in noun_lemmas)

    def _deduplicate(self, candidates: list[AspectCandidate]) -> list[AspectCandidate]:
        seen: set[str] = set()
        unique: list[AspectCandidate] = []
        for candidate in candidates:
            if candidate.aspect in seen:
                continue
            seen.add(candidate.aspect)
            unique.append(candidate)
        return unique
