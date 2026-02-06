from __future__ import annotations

from .approaches import APPROACHES
from .preprocess import TextPreprocessor


class AspectOpinionMiner:
    """Pipeline entrypoint supporting 4 different explainable approaches."""

    def __init__(self, approach: str = "v4_ensemble_consensus"):
        if approach not in APPROACHES:
            raise ValueError(f"Unknown approach '{approach}'. Available: {', '.join(APPROACHES)}")
        self.preprocessor = TextPreprocessor()
        self.approach_name = approach
        self.approach = APPROACHES[approach]()

    def analyze(self, text: str) -> list[dict]:
        doc = self.preprocessor.process(text)
        return [r.to_dict() for r in self.approach.run(doc)]


def analyze_with_all_versions(text: str) -> dict[str, list[dict]]:
    doc = TextPreprocessor().process(text)
    results: dict[str, list[dict]] = {}
    for name, klass in APPROACHES.items():
        results[name] = [x.to_dict() for x in klass().run(doc)]
    return results
