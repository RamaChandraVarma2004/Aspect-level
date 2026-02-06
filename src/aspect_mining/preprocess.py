from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import importlib


@dataclass
class PreprocessConfig:
    model_name: str = "en_core_web_sm"


class TextPreprocessor:
    """Loads and serves a spaCy pipeline used by all downstream modules."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._nlp = self._load_model(self.config.model_name)

    @staticmethod
    def _load_model(model_name: str) -> Any:
        spacy = importlib.import_module("spacy")
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is missing. Run: "
                f"python -m spacy download {model_name}"
            ) from exc

    @property
    def nlp(self):
        return self._nlp

    def process(self, text: str):
        return self._nlp(text.strip())
