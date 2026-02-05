"""Text preprocessing module.

Why this exists:
- Keeps data cleanup and linguistic normalization reusable and testable.
- Makes each later component work with a consistent token representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@dataclass
class TokenFeatures:
    text: str
    lemma: str
    pos: str
    is_stop: bool
    token_index: int


class Preprocessor:
    """Handles sentence split, tokenization, lemmatization, and stopword flags."""

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    def tokenize_with_features(self, sentence: str) -> list[TokenFeatures]:
        raw_tokens = word_tokenize(sentence)
        tags = pos_tag(raw_tokens)
        features: list[TokenFeatures] = []

        for idx, (token, pos) in enumerate(tags):
            lower = token.lower()
            lemma = self._lemmatize(token, pos)
            features.append(
                TokenFeatures(
                    text=token,
                    lemma=lemma,
                    pos=pos,
                    is_stop=lower in self.stopwords,
                    token_index=idx,
                )
            )
        return features

    def _lemmatize(self, token: str, pos_tag_value: str) -> str:
        pos_map = {
            "J": "a",  # adjective
            "V": "v",  # verb
            "N": "n",  # noun
            "R": "r",  # adverb
        }
        wn_pos = pos_map.get(pos_tag_value[0], "n")
        return self.lemmatizer.lemmatize(token.lower(), wn_pos)
