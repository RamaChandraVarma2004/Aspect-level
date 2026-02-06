from __future__ import annotations


class TextPreprocessor:
    """Centralized spaCy loader for sentence split, tokenization, POS and lemmas."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = self._load()

    def _load(self):
        try:
            import spacy
        except ModuleNotFoundError as exc:
            raise RuntimeError("spaCy is not installed. Run: pip install -r requirements.txt") from exc

        try:
            return spacy.load(self.model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Missing spaCy model '{self.model_name}'. Run: python -m spacy download {self.model_name}"
            ) from exc

    def process(self, text: str):
        return self._nlp(text.strip())
