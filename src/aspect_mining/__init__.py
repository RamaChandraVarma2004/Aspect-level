"""Aspect-level opinion mining package with 4 interchangeable approaches."""

from .pipeline import AspectOpinionMiner

AVAILABLE_VERSIONS = {
    "v1": "spaCy rule-based noun-chunk + distance",
    "v2": "spaCy dependency-focused aspect-opinion linking",
    "v3": "lightweight pure-python lexicon + context window",
    "v4": "hybrid ensemble over v1/v2/v3 with consensus",
}

__all__ = ["AspectOpinionMiner", "AVAILABLE_VERSIONS"]
