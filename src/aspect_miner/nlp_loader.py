"""Utilities for loading NLP resources with graceful fallbacks."""

from __future__ import annotations

from functools import lru_cache

import nltk
import spacy
from spacy.language import Language


@lru_cache(maxsize=1)
def load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load spaCy model and auto-download if missing.

    Why: this keeps the demo easy to run for recruiters by handling first-run setup.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


@lru_cache(maxsize=1)
def load_opinion_lexicons() -> tuple[set[str], set[str]]:
    """Load positive and negative lexicons from NLTK opinion corpus."""
    try:
        from nltk.corpus import opinion_lexicon

        positive = set(opinion_lexicon.positive())
        negative = set(opinion_lexicon.negative())
        if positive and negative:
            return positive, negative
    except LookupError:
        nltk.download("opinion_lexicon", quiet=True)
        from nltk.corpus import opinion_lexicon

        return set(opinion_lexicon.positive()), set(opinion_lexicon.negative())

    # Fallback, should be rare.
    return {
        "good",
        "great",
        "excellent",
        "amazing",
        "fast",
        "smooth",
    }, {"bad", "poor", "slow", "terrible", "awful"}
