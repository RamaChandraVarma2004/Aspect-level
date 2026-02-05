"""Text preprocessing component for ABSA pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from spacy.tokens import Doc, Span, Token


@dataclass
class PreprocessedSentence:
    text: str
    span: Span
    content_tokens: list[Token]


class TextPreprocessor:
    """Runs segmentation, tokenization, lemmatization and stopword filtering.

    Why: explicit preprocessing step makes pipeline transparent and debuggable.
    """

    def preprocess(self, doc: Doc) -> list[PreprocessedSentence]:
        sentences: list[PreprocessedSentence] = []
        for sent in doc.sents:
            content_tokens = [
                tok
                for tok in sent
                if not tok.is_space and not tok.is_punct and not tok.is_stop
            ]
            sentences.append(
                PreprocessedSentence(
                    text=sent.text.strip(),
                    span=sent,
                    content_tokens=content_tokens,
                )
            )
        return sentences
