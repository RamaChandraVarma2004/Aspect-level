from __future__ import annotations

from .versions.v1_spacy_rule import V1SpacyRuleMiner
from .versions.v2_dependency_focus import V2DependencyMiner
from .versions.v3_lexicon_window import V3LexiconWindowMiner
from .versions.v4_hybrid_ensemble import V4HybridEnsembleMiner


class AspectOpinionMiner:
    """Factory wrapper to run one of four clearly different ALOM approaches."""

    def __init__(self, version: str = "v4"):
        self.version = version.lower()
        self.engine = self._build(self.version)

    def _build(self, version: str):
        if version == "v1":
            return V1SpacyRuleMiner()
        if version == "v2":
            return V2DependencyMiner()
        if version == "v3":
            return V3LexiconWindowMiner()
        if version == "v4":
            return V4HybridEnsembleMiner()
        raise ValueError("version must be one of: v1, v2, v3, v4")

    def analyze(self, text: str) -> list[dict]:
        return self.engine.analyze(text)
