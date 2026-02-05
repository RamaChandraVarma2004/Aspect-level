"""Aspect-opinion association with interpretable neighborhood + dependency rules."""

from __future__ import annotations

from typing import List

from .schemas import OpinionEvidence
from .sentiment import SentimentScorer


class AspectOpinionAssociator:
    def __init__(self, scorer: SentimentScorer, window: int = 5) -> None:
        self.scorer = scorer
        self.window = window

    def opinions_for_aspect(self, sent, aspect: str) -> List[OpinionEvidence]:
        aspect_tokens = set(aspect.split())
        evidence: List[OpinionEvidence] = []

        for token in sent:
            if not token.is_alpha:
                continue

            raw_score, reason = self.scorer.score_token(token)
            if raw_score == 0:
                continue

            near_aspect = False
            # Heuristic 1: lexical neighborhood window
            for tok in sent[max(0, token.i - self.window) : token.i + self.window + 1]:
                if tok.lemma_.lower() in aspect_tokens or tok.text.lower() in aspect_tokens:
                    near_aspect = True
                    break

            # Heuristic 2: dependency graph ties (token modifies noun in aspect)
            dep_connected = (
                token.head.lemma_.lower() in aspect_tokens
                or any(child.lemma_.lower() in aspect_tokens for child in token.children)
            )

            if near_aspect or dep_connected:
                evidence.append(
                    OpinionEvidence(
                        token=token.text,
                        lemma=token.lemma_.lower(),
                        score=raw_score,
                        reason=reason,
                    )
                )

        return evidence
