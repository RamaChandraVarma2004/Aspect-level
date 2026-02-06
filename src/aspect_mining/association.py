from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Span, Token

from .features import LinguisticFeatureExtractor
from .schemas import AspectSentiment, OpinionEvidence
from .sentiment import SentimentScorer


@dataclass
class AssociationConfig:
    max_distance: int = 7


class BaseAssociator:
    """Base class with reusable helpers for interpretable aspect-opinion association."""

    def __init__(self, config: AssociationConfig | None = None):
        self.config = config or AssociationConfig()
        self.features = LinguisticFeatureExtractor()
        self.sentiment = SentimentScorer()

    @staticmethod
    def _group_by_sentence(aspects: list["Span"]) -> dict[int, list["Span"]]:
        by_sentence: dict[int, list["Span"]] = defaultdict(list)
        for asp in aspects:
            by_sentence[asp.sent.start].append(asp)
        return by_sentence

    @staticmethod
    def _pack_result(aspect: "Span", evidences: list[OpinionEvidence], scorer: SentimentScorer) -> AspectSentiment:
        score = 0.0
        if evidences:
            score = sum(ev.adjusted_score for ev in evidences) / len(evidences)
        return AspectSentiment(
            aspect=aspect.text,
            sentiment=scorer.label(score),
            score=round(score, 3),
            sentence=aspect.sent.text,
            evidences=sorted(evidences, key=lambda x: x.distance),
        )

    def associate(self, aspects: list["Span"]) -> list[AspectSentiment]:
        raise NotImplementedError


class ProximityAssociator(BaseAssociator):
    """Version 1: nearest-window opinion assignment in each sentence."""

    def associate(self, aspects: list["Span"]) -> list[AspectSentiment]:
        results: list[AspectSentiment] = []
        for _, sentence_aspects in self._group_by_sentence(aspects).items():
            sentence = sentence_aspects[0].sent
            opinion_tokens = self.features.opinion_tokens(sentence)
            for aspect in sentence_aspects:
                center = aspect.root.i
                evidences = [
                    self.sentiment.score_opinion(sentence, op_tok, center)
                    for op_tok in opinion_tokens
                    if abs(op_tok.i - center) <= self.config.max_distance
                ]
                results.append(self._pack_result(aspect, evidences, self.sentiment))
        return results


class DependencyAwareAssociator(BaseAssociator):
    """Version 2: prefer syntactic links, fallback to proximity.

    Why: recruiters like seeing dependency-based explainability beyond pure distance.
    """

    def _dependency_candidates(self, aspect: "Span", opinion_tokens: list["Token"]) -> Iterable["Token"]:
        head = aspect.root
        for tok in opinion_tokens:
            if tok.head == head or head.head == tok or tok.head == head.head:
                yield tok

    def associate(self, aspects: list["Span"]) -> list[AspectSentiment]:
        results: list[AspectSentiment] = []
        for _, sentence_aspects in self._group_by_sentence(aspects).items():
            sentence = sentence_aspects[0].sent
            opinion_tokens = self.features.opinion_tokens(sentence)

            for aspect in sentence_aspects:
                center = aspect.root.i
                dep_candidates = list(self._dependency_candidates(aspect, opinion_tokens))
                chosen = dep_candidates if dep_candidates else opinion_tokens
                evidences = [
                    self.sentiment.score_opinion(sentence, op_tok, center)
                    for op_tok in chosen
                    if abs(op_tok.i - center) <= self.config.max_distance
                ]
                results.append(self._pack_result(aspect, evidences, self.sentiment))
        return results


class ContrastAwareAssociator(BaseAssociator):
    """Version 3: clause-aware assignment around contrast markers (but/however)."""

    CONTRAST_MARKERS = {"but", "however", "although", "though", "while"}

    @staticmethod
    def _token_side(sentence: "Span", token_index: int) -> int:
        marker_positions = [t.i for t in sentence if t.text.lower() in ContrastAwareAssociator.CONTRAST_MARKERS]
        if not marker_positions:
            return 0
        split = marker_positions[0]
        return -1 if token_index < split else 1

    def associate(self, aspects: list["Span"]) -> list[AspectSentiment]:
        results: list[AspectSentiment] = []
        for _, sentence_aspects in self._group_by_sentence(aspects).items():
            sentence = sentence_aspects[0].sent
            opinion_tokens = self.features.opinion_tokens(sentence)

            for aspect in sentence_aspects:
                center = aspect.root.i
                aspect_side = self._token_side(sentence, center)
                side_matched = [
                    op for op in opinion_tokens if self._token_side(sentence, op.i) == aspect_side
                ]
                chosen = side_matched if side_matched else opinion_tokens
                evidences = [
                    self.sentiment.score_opinion(sentence, op_tok, center)
                    for op_tok in chosen
                    if abs(op_tok.i - center) <= self.config.max_distance
                ]
                results.append(self._pack_result(aspect, evidences, self.sentiment))
        return results


class EnsembleAssociator(BaseAssociator):
    """Version 4: average scores from V1/V2/V3 for stable and practical output."""

    def __init__(self, config: AssociationConfig | None = None):
        super().__init__(config=config)
        self.v1 = ProximityAssociator(config=config)
        self.v2 = DependencyAwareAssociator(config=config)
        self.v3 = ContrastAwareAssociator(config=config)

    def associate(self, aspects: list["Span"]) -> list[AspectSentiment]:
        v1_results = self.v1.associate(aspects)
        v2_results = self.v2.associate(aspects)
        v3_results = self.v3.associate(aspects)

        merged: list[AspectSentiment] = []
        for r1, r2, r3 in zip(v1_results, v2_results, v3_results):
            avg_score = round((r1.score + r2.score + r3.score) / 3, 3)
            merged_evidences = sorted(
                r1.evidences + r2.evidences + r3.evidences,
                key=lambda x: (x.word.lower(), x.distance, -abs(x.adjusted_score)),
            )
            dedup: list[OpinionEvidence] = []
            seen: set[tuple[str, int]] = set()
            for ev in merged_evidences:
                key = (ev.word.lower(), ev.distance)
                if key not in seen:
                    seen.add(key)
                    dedup.append(ev)
            merged.append(
                AspectSentiment(
                    aspect=r1.aspect,
                    sentiment=self.sentiment.label(avg_score),
                    score=avg_score,
                    sentence=r1.sentence,
                    evidences=dedup,
                )
            )

        return merged
