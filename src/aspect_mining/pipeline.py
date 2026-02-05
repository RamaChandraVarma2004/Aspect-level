"""End-to-end pipeline orchestrator for aspect-level opinion mining."""

from __future__ import annotations

from .aspect_extractor import AspectExtractor
from .association import AspectOpinionAssociator
from .config import PipelineConfig
from .nlp_resources import ensure_nltk_resources
from .preprocessing import Preprocessor
from .schemas import AspectSentimentResult
from .sentiment import OpinionScorer


class AspectMiningPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        ensure_nltk_resources()
        self.config = config or PipelineConfig()
        self.preprocessor = Preprocessor()
        self.aspect_extractor = AspectExtractor(self.config)
        self.opinion_scorer = OpinionScorer(self.config)
        self.associator = AspectOpinionAssociator(self.config)

    def analyze(self, text: str) -> list[AspectSentimentResult]:
        results: list[AspectSentimentResult] = []

        for sentence in self.preprocessor.split_sentences(text):
            tokens = self.preprocessor.tokenize_with_features(sentence)
            aspects = self.aspect_extractor.extract(tokens)
            opinions = self.opinion_scorer.extract_opinions(tokens)
            mapping = self.associator.associate(aspects, opinions)

            for aspect in aspects:
                evidence = mapping.get(aspect.aspect, [])
                score = sum(item.adjusted_score for item in evidence)
                sentiment = self.opinion_scorer.classify(score)
                results.append(
                    AspectSentimentResult(
                        aspect=aspect.aspect,
                        sentence=sentence,
                        sentiment=sentiment,
                        score=round(score, 3),
                        opinion_words=[item.word for item in evidence],
                        evidence=evidence,
                    )
                )

        return results
