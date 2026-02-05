"""Aspect-opinion association logic with interpretable distance rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from .aspect_extractor import AspectCandidate
from .features import OpinionToken


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    score: float
    evidence: list[str] = field(default_factory=list)


class AspectOpinionAssociator:
    """Associate aspects to nearby opinions, handling multi-aspect sentences."""

    def associate(
        self,
        aspects: list[AspectCandidate],
        opinions: list[OpinionToken],
    ) -> list[AspectSentiment]:
        results: list[AspectSentiment] = []
        for aspect in aspects:
            weighted_scores: list[float] = []
            evidence: list[str] = []

            for opinion in opinions:
                distance = abs(opinion.token.i - aspect.root.i)
                if distance > 7:
                    continue
                weight = 1 / (distance + 1)
                contribution = opinion.final_score * weight
                weighted_scores.append(contribution)
                evidence.append(
                    f"{opinion.token.text} ({opinion.final_score:+.2f}, d={distance}, w={weight:.2f})"
                )

            score = sum(weighted_scores)
            sentiment = self._label(score)
            results.append(
                AspectSentiment(
                    aspect=aspect.text,
                    sentiment=sentiment,
                    score=round(score, 3),
                    evidence=evidence,
                )
            )
        return results

    @staticmethod
    def _label(score: float) -> str:
        if score > 0.18:
            return "positive"
        if score < -0.18:
            return "negative"
        return "neutral"
