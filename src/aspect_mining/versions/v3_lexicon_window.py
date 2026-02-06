from __future__ import annotations

from ..lexicon import SENTIMENT_LEXICON, NEGATIONS, INTENSIFIERS, GENERIC_ASPECTS
from ..schemas import AspectSentiment, OpinionEvidence
from ..utils import split_sentences, simple_tokens, candidate_aspects_from_tokens


class V3LexiconWindowMiner:
    name = "V3 - Lightweight Lexicon+Window Pipeline"

    def analyze(self, text: str) -> list[dict]:
        rows: list[AspectSentiment] = []
        for sent in split_sentences(text):
            clauses = [c.strip() for c in __import__("re").split(r"\bbut\b|\bthough\b|\bhowever\b", sent.lower()) if c.strip()]
            for clause in clauses:
                tokens = simple_tokens(clause)
                aspects = [a for a in candidate_aspects_from_tokens(tokens) if a.split()[-1] not in GENERIC_ASPECTS][:10]
                opinions = [(i, t) for i, t in enumerate(tokens) if t in SENTIMENT_LEXICON]
                for asp in aspects:
                    a_head = asp.split()[-1]
                    positions = [i for i, t in enumerate(tokens) if t == a_head]
                    center = positions[0] if positions else 0
                    ev = []
                    for i, op in opinions:
                        if abs(i - center) > 6:
                            continue
                        base = SENTIMENT_LEXICON[op]
                        prev = tokens[max(0, i - 3) : i]
                        neg = any(w in NEGATIONS for w in prev)
                        intens = next((w for w in reversed(prev) if w in INTENSIFIERS), None)
                        score = base * (INTENSIFIERS[intens] if intens else 1.0)
                        if neg:
                            score *= -1
                        score *= max(0.45, 1 - 0.1 * abs(i - center))
                        ev.append(OpinionEvidence(op, base, round(score, 3), neg, intens, abs(i - center), source="v3"))
                    avg = sum(e.adjusted_score for e in ev) / len(ev) if ev else 0.0
                    sent_label = "positive" if avg > 0.4 else "negative" if avg < -0.4 else "neutral"
                    rows.append(AspectSentiment(asp, sent_label, round(avg, 3), sent, ev, self.name))
        return [r.to_dict() for r in rows]
