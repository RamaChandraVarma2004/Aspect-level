from __future__ import annotations

from collections import defaultdict
from spacy.tokens import Span

from .features import LinguisticFeatureExtractor
from .schemas import AspectSentiment
from .sentiment import SentimentScorer


class AspectOpinionAssociator:
    """Attach opinion words to aspects in the same sentence with clear heuristics."""

    def __init__(self):
        self.features = LinguisticFeatureExtractor()
        self.sentiment = SentimentScorer()

    def associate(self, aspects: list[Span]) -> list[AspectSentiment]:
        by_sentence: dict[int, list[Span]] = defaultdict(list)
        for asp in aspects:
            by_sentence[asp.sent.start].append(asp)

        results: list[AspectSentiment] = []
        for sentence_start, sentence_aspects in by_sentence.items():
            sentence = sentence_aspects[0].sent
            opinion_tokens = self.features.opinion_tokens(sentence)

            for aspect in sentence_aspects:
                center = aspect.root.i
                evidences = [
                    self.sentiment.score_opinion(sentence, op_tok, center)
                    for op_tok in opinion_tokens
                    if abs(op_tok.i - center) <= 7
                ]

                score = 0.0
                if evidences:
                    score = sum(ev.adjusted_score for ev in evidences) / len(evidences)

                results.append(
                    AspectSentiment(
                        aspect=aspect.text,
                        sentiment=self.sentiment.label(score),
                        score=round(score, 3),
                        sentence=sentence.text,
                        evidences=sorted(evidences, key=lambda x: x.distance),
                    )
                )

        return results
