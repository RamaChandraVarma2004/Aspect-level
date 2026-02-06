from __future__ import annotations

from .approaches import APPROACHES
from .preprocess import TextPreprocessor


class AspectOpinionMiner:
    """Versioned ABSA orchestrator.

    Supports four different explainable approaches so users can compare output
    quality and discuss trade-offs in interviews.
    """

    def __init__(self, version: str = "v2"):
        if version not in APPROACHES:
            raise ValueError(f"Unknown version: {version}. Available: {', '.join(APPROACHES)}")
        self.version = version
        self.preprocessor = TextPreprocessor()
        self.approach = APPROACHES[version]()

    def analyze(self, text: str) -> list[dict]:
        doc = self.preprocessor.process(text)
        rows = self.approach.analyze_doc(doc)
        return [row.to_dict() for row in rows]


def analyze_with_all_versions(text: str) -> dict[str, list[dict]]:
    outputs: dict[str, list[dict]] = {}
    for version in APPROACHES:
        miner = AspectOpinionMiner(version=version)
        outputs[version] = miner.analyze(text)
    return outputs
