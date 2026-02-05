"""Centralized configuration for interpretable rule-based aspect mining."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration knobs to keep behavior transparent and easy to tune."""

    min_aspect_len: int = 2
    max_aspect_words: int = 4
    association_window: int = 6
    neutral_margin: float = 0.15
    intensifiers: dict[str, float] = field(
        default_factory=lambda: {
            "very": 1.3,
            "really": 1.25,
            "extremely": 1.5,
            "super": 1.35,
            "slightly": 0.75,
            "somewhat": 0.85,
            "too": 1.2,
        }
    )
    negations: set[str] = field(
        default_factory=lambda: {
            "not",
            "never",
            "no",
            "hardly",
            "barely",
            "n't",
        }
    )
    generic_nouns: set[str] = field(
        default_factory=lambda: {
            "thing",
            "things",
            "stuff",
            "product",
            "item",
            "experience",
            "time",
        }
    )
