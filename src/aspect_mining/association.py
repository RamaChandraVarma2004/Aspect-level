"""Associates aspects to nearest opinion words for multi-aspect sentences."""

from __future__ import annotations

from collections import defaultdict

from .aspect_extractor import AspectCandidate
from .config import PipelineConfig
from .schemas import OpinionEvidence


class AspectOpinionAssociator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def associate(
        self,
        aspects: list[AspectCandidate],
        opinions: list[OpinionEvidence],
    ) -> dict[str, list[OpinionEvidence]]:
        grouped: dict[str, list[OpinionEvidence]] = defaultdict(list)
        for opinion in opinions:
            nearest = self._nearest_aspect(opinion.token_index, aspects)
            if nearest is None:
                continue
            grouped[nearest.aspect].append(opinion)
        return grouped

    def _nearest_aspect(self, opinion_idx: int, aspects: list[AspectCandidate]) -> AspectCandidate | None:
        nearest: AspectCandidate | None = None
        best_distance = float("inf")

        for candidate in aspects:
            center = sum(candidate.token_indexes) / len(candidate.token_indexes)
            distance = abs(opinion_idx - center)
            if distance <= self.config.association_window and distance < best_distance:
                best_distance = distance
                nearest = candidate

        return nearest
