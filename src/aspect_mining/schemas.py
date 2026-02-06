from dataclasses import dataclass, asdict
from typing import List


@dataclass
class OpinionEvidence:
    word: str
    base_score: float
    adjusted_score: float
    negated: bool
    intensifier: str | None
    distance: int
    source: str = "rule"


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    score: float
    sentence: str
    evidences: List[OpinionEvidence]
    approach: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["evidences"] = [asdict(ev) for ev in self.evidences]
        return payload
