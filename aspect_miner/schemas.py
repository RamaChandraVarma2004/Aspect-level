from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class OpinionEvidence:
    token: str
    lemma: str
    score: float
    reason: str


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    score: float
    sentence: str
    opinions: List[OpinionEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "score": round(self.score, 3),
            "sentence": self.sentence,
            "opinions": [
                {
                    "token": evidence.token,
                    "lemma": evidence.lemma,
                    "score": round(evidence.score, 3),
                    "reason": evidence.reason,
                }
                for evidence in self.opinions
            ],
        }
