"""Composable pipeline orchestrating all ABOM components."""

from __future__ import annotations

from typing import Dict, List

from .aspect_extraction import AspectExtractor
from .association import AspectOpinionAssociator
from .feature_extraction import FeatureExtractor
from .preprocessing import TextPreprocessor
from .schemas import AspectSentiment
from .sentiment import SentimentScorer


class AspectOpinionPipeline:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.aspect_extractor = AspectExtractor()
        self.sentiment_scorer = SentimentScorer()
        self.associator = AspectOpinionAssociator(self.sentiment_scorer)

    def analyze(self, text: str) -> Dict:
        doc = self.preprocessor.process(text)
        records: List[AspectSentiment] = []

        for sent in doc.sents:
            if not sent.text.strip():
                continue

            _ = self.feature_extractor.extract(sent)
            aspects = self.aspect_extractor.extract(sent)
            for aspect in aspects:
                opinions = self.associator.opinions_for_aspect(sent, aspect)
                score = sum(op.score for op in opinions)
                records.append(
                    AspectSentiment(
                        aspect=aspect,
                        sentiment=self.sentiment_scorer.polarity_label(score),
                        score=score,
                        sentence=sent.text.strip(),
                        opinions=opinions,
                    )
                )

        return {
            "input_text": text,
            "aspect_results": [record.to_dict() for record in records],
            "summary": self._summarize(records),
        }

    @staticmethod
    def _summarize(records: List[AspectSentiment]) -> Dict[str, int]:
        summary = {"positive": 0, "negative": 0, "neutral": 0}
        for record in records:
            summary[record.sentiment] += 1
        return summary
