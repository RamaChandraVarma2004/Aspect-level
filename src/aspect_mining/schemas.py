"""Typed structures for cleaner interfaces between pipeline modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OpinionEvidence:
    word: str
    lemma: str
    raw_score: float
    adjusted_score: float
    negated: bool
    intensifier: str | None
    token_index: int


@dataclass
class AspectSentimentResult:
    aspect: str
    sentence: str
    sentiment: str
    score: float
    opinion_words: list[str] = field(default_factory=list)
    evidence: list[OpinionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["evidence"] = [asdict(e) for e in self.evidence]
        return payload
