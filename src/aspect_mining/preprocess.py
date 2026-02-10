from __future__ import annotations

from dataclasses import dataclass
import spacy
from spacy.language import Language


@dataclass
class PreprocessConfig:
    model_name: str = "en_core_web_sm"


class TextPreprocessor:
    """Loads and serves a spaCy pipeline used by all downstream modules."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._nlp, self.using_fallback = self._load_model(self.config.model_name)

    @staticmethod
    def _load_model(model_name: str) -> tuple[Language, bool]:
        try:
            return spacy.load(model_name), False
        except OSError:
            # Offline-safe fallback for restricted environments.
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp, True

    @property
    def nlp(self) -> Language:
        return self._nlp

    def process(self, text: str):
        return self._nlp(text.strip())
