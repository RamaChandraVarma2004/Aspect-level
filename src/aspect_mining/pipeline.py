from __future__ import annotations

from .aspect_extractor import AspectExtractor
from .association import (
    ContrastAwareAssociator,
    DependencyAwareAssociator,
    EnsembleAssociator,
    ProximityAssociator,
)
from .preprocess import TextPreprocessor


class AspectOpinionMiner:
    """End-to-end orchestrator for 4 internship-friendly ABSA versions.

    Versions:
    - v1: Proximity-based
    - v2: Dependency-aware
    - v3: Contrast/clause-aware
    - v4: Ensemble (recommended default for stability)
    """

    ASSOCIATORS = {
        "v1": ProximityAssociator,
        "v2": DependencyAwareAssociator,
        "v3": ContrastAwareAssociator,
        "v4": EnsembleAssociator,
    }

    def __init__(self, version: str = "v4"):
        if version not in self.ASSOCIATORS:
            raise ValueError(f"Unsupported version '{version}'. Use one of {list(self.ASSOCIATORS)}")
        self.version = version
        self.preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.associator = self.ASSOCIATORS[version]()

    def analyze(self, text: str) -> list[dict]:
        doc = self.preprocessor.process(text)
        aspects = self.aspect_extractor.extract(doc)
        aspect_sentiments = self.associator.associate(aspects)
        return [item.to_dict() for item in aspect_sentiments]


def analyze_all_versions(text: str) -> dict[str, list[dict]]:
    """Convenience utility for side-by-side comparison in the UI."""
    return {version: AspectOpinionMiner(version=version).analyze(text) for version in ["v1", "v2", "v3", "v4"]}
