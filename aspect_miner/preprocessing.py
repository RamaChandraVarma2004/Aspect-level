"""Preprocessing module.

Why: we keep this separate so each NLP phase is explainable and replaceable.
"""

from __future__ import annotations

import spacy
from spacy.language import Language


class TextPreprocessor:
    """Sentence splitting, tokenization, lemmatization, and stopword flags."""

    def __init__(self) -> None:
        self.nlp = self._load_model()

    @staticmethod
    def _load_model() -> Language:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Ready-to-run fallback: download model if user installed dependencies only.
            from spacy.cli import download

            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def process(self, text: str):
        return self.nlp(text)
