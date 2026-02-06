from dataclasses import asdict, dataclass
from typing import List


@dataclass
class OpinionEvidence:
    word: str
    score: float
    reason: str


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
        payload["evidences"] = [asdict(x) for x in self.evidences]
        return payload
