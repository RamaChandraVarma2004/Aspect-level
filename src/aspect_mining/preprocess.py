from __future__ import annotations

from dataclasses import dataclass
import spacy
from spacy.language import Language


@dataclass
class PreprocessConfig:
    model_name: str = "en_core_web_sm"


class TextPreprocessor:
    """Centralized spaCy preprocessing.

    Why: every approach should use the same tokenizer, sentence splitter,
    POS tags, and lemmas to keep behavior comparable.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._nlp = self._load_model(self.config.model_name)

    @staticmethod
    def _load_model(model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is missing. Run: python -m spacy download {model_name}"
            ) from exc

    @property
    def nlp(self) -> Language:
        return self._nlp

    def process(self, text: str):
        return self._nlp(text.strip())
