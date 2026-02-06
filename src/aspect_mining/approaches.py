from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from .aspect_extractor import AspectExtractor
from .features import LinguisticFeatureExtractor
from .schemas import AspectSentiment, OpinionEvidence
from .sentiment import SentimentScorer


class Approach(Protocol):
    name: str

    def analyze_doc(self, doc) -> list[AspectSentiment]: ...


class V1ProximityRuleBased:
    """Version 1: nearest-opinion window association."""

    name = "V1 - Proximity Rules (baseline)"

    def __init__(self):
        self.aspect_extractor = AspectExtractor()
        self.features = LinguisticFeatureExtractor()
        self.scorer = SentimentScorer()

    def analyze_doc(self, doc) -> list[AspectSentiment]:
        aspects = self.aspect_extractor.extract(doc)
        by_sentence = defaultdict(list)
        for asp in aspects:
            by_sentence[asp.sent.start].append(asp)

        rows: list[AspectSentiment] = []
        for sentence_aspects in by_sentence.values():
            sentence = sentence_aspects[0].sent
            opinion_tokens = self.features.opinion_tokens(sentence)
            for aspect in sentence_aspects:
                center = aspect.root.i
                evidences = [
                    self.scorer.score_opinion(sentence, op_tok, center)
                    for op_tok in opinion_tokens
                    if abs(op_tok.i - center) <= 7
                ]
                score = sum(ev.adjusted_score for ev in evidences) / len(evidences) if evidences else 0.0
                rows.append(
                    AspectSentiment(
                        aspect=aspect.text,
                        sentiment=self.scorer.label(score),
                        score=round(score, 3),
                        confidence=self.scorer.confidence(score, len(evidences)),
                        sentence=sentence.text,
                        method=self.name,
                        evidences=sorted(evidences, key=lambda x: x.distance),
                    )
                )
        return rows


class V2DependencyRuleBased:
    """Version 2: dependency-first association for better multi-aspect handling."""

    name = "V2 - Dependency Rules"

    def __init__(self):
        self.aspect_extractor = AspectExtractor()
        self.features = LinguisticFeatureExtractor()
        self.scorer = SentimentScorer()

    def analyze_doc(self, doc) -> list[AspectSentiment]:
        aspects = self.aspect_extractor.extract(doc)
        rows: list[AspectSentiment] = []

        for aspect in aspects:
            sentence = aspect.sent
            related = self.features.dependency_related(aspect.root, sentence)
            # fallback to local window if dependency does not give evidence
            if not related:
                related = [tok for tok in self.features.opinion_tokens(sentence) if abs(tok.i - aspect.root.i) <= 5]

            evidences = [self.scorer.score_opinion(sentence, tok, aspect.root.i) for tok in related]
            score = sum(ev.adjusted_score for ev in evidences) / len(evidences) if evidences else 0.0

            rows.append(
                AspectSentiment(
                    aspect=aspect.text,
                    sentiment=self.scorer.label(score),
                    score=round(score, 3),
                    confidence=self.scorer.confidence(score, len(evidences)),
                    sentence=sentence.text,
                    method=self.name,
                    evidences=sorted(evidences, key=lambda x: x.distance),
                )
            )
        return rows


class V3HybridSentenceWeighted:
    """Version 3: sentence-level sentiment + local aspect correction.

    Great for intern-level explainability: combines coarse sentence tone with
    aspect-local evidence.
    """

    name = "V3 - Hybrid (sentence + local)"

    def __init__(self):
        self.aspect_extractor = AspectExtractor()
        self.features = LinguisticFeatureExtractor()
        self.scorer = SentimentScorer()

    def _sentence_score(self, sentence) -> float:
        opinions = self.features.opinion_tokens(sentence)
        if not opinions:
            return 0.0
        scores = [self.scorer.score_opinion(sentence, tok, tok.i).adjusted_score for tok in opinions]
        return sum(scores) / len(scores)

    def analyze_doc(self, doc) -> list[AspectSentiment]:
        aspects = self.aspect_extractor.extract(doc)
        rows: list[AspectSentiment] = []

        for aspect in aspects:
            sentence = aspect.sent
            local_ops = [tok for tok in self.features.opinion_tokens(sentence) if abs(tok.i - aspect.root.i) <= 6]
            evidences = [self.scorer.score_opinion(sentence, tok, aspect.root.i) for tok in local_ops]

            local_score = sum(ev.adjusted_score for ev in evidences) / len(evidences) if evidences else 0.0
            sentence_score = self._sentence_score(sentence)
            score = 0.65 * local_score + 0.35 * sentence_score

            rows.append(
                AspectSentiment(
                    aspect=aspect.text,
                    sentiment=self.scorer.label(score),
                    score=round(score, 3),
                    confidence=self.scorer.confidence(score, len(evidences)),
                    sentence=sentence.text,
                    method=self.name,
                    evidences=sorted(evidences, key=lambda x: x.distance),
                )
            )
        return rows


class V4ContrastiveClauseAware:
    """Version 4: clause-aware rules emphasizing contrast words (but/however/though)."""

    name = "V4 - Clause-aware Contrastive Rules"
    CONTRAST = {"but", "however", "though", "although", "yet"}

    def __init__(self):
        self.aspect_extractor = AspectExtractor()
        self.features = LinguisticFeatureExtractor()
        self.scorer = SentimentScorer()

    def _clause_weight(self, sentence, aspect_i: int, opinion_i: int) -> float:
        left, right = sorted((aspect_i, opinion_i))
        between = {tok.text.lower() for tok in sentence if left <= tok.i <= right}
        return 0.85 if between & self.CONTRAST else 1.0

    def analyze_doc(self, doc) -> list[AspectSentiment]:
        aspects = self.aspect_extractor.extract(doc)
        rows: list[AspectSentiment] = []

        for aspect in aspects:
            sentence = aspect.sent
            opinions = self.features.opinion_tokens(sentence)
            raw: list[OpinionEvidence] = [self.scorer.score_opinion(sentence, tok, aspect.root.i) for tok in opinions]

            adjusted_evs: list[OpinionEvidence] = []
            for ev, tok in zip(raw, opinions):
                clause_factor = self._clause_weight(sentence, aspect.root.i, tok.i)
                ev.adjusted_score = round(ev.adjusted_score * clause_factor, 3)
                adjusted_evs.append(ev)

            near = [ev for ev in adjusted_evs if ev.distance <= 7]
            score = sum(ev.adjusted_score for ev in near) / len(near) if near else 0.0

            rows.append(
                AspectSentiment(
                    aspect=aspect.text,
                    sentiment=self.scorer.label(score),
                    score=round(score, 3),
                    confidence=self.scorer.confidence(score, len(near)),
                    sentence=sentence.text,
                    method=self.name,
                    evidences=sorted(near, key=lambda x: x.distance),
                )
            )
        return rows


APPROACHES = {
    "v1": V1ProximityRuleBased,
    "v2": V2DependencyRuleBased,
    "v3": V3HybridSentenceWeighted,
    "v4": V4ContrastiveClauseAware,
}
