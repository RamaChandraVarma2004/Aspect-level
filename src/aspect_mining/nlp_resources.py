"""NLTK resource setup to make the project plug-and-play."""

from __future__ import annotations

import nltk

_REQUIRED = [
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
    "stopwords",
    "vader_lexicon",
]


def ensure_nltk_resources() -> None:
    """Download required corpora/models only if missing."""
    for resource in _REQUIRED:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "averaged_perceptron_tagger":
                nltk.data.find("taggers/averaged_perceptron_tagger")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "vader_lexicon":
                nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download(resource, quiet=True)
