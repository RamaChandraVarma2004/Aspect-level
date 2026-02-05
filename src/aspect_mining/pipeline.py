from __future__ import annotations

from .aspect_extractor import AspectExtractor
from .association import AspectOpinionAssociator
from .preprocess import TextPreprocessor


class AspectOpinionMiner:
    """End-to-end orchestrator for modular aspect-level sentiment mining."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.associator = AspectOpinionAssociator()

    def analyze(self, text: str) -> list[dict]:
        doc = self.preprocessor.process(text)
        aspects = self.aspect_extractor.extract(doc)
        aspect_sentiments = self.associator.associate(aspects)
        return [item.to_dict() for item in aspect_sentiments]
