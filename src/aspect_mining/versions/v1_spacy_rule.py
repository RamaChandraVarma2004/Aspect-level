from __future__ import annotations

from ..lexicon import SENTIMENT_LEXICON, NEGATIONS, INTENSIFIERS, GENERIC_ASPECTS
from ..schemas import AspectSentiment, OpinionEvidence


class V1SpacyRuleMiner:
    name = "V1 - spaCy Rule Pipeline"

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy

            self.nlp = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError("V1 requires spaCy model en_core_web_sm") from exc

    def analyze(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        results: list[AspectSentiment] = []
        for sent in doc.sents:
            aspects = []
            for chunk in sent.noun_chunks:
                root = chunk.root.lemma_.lower()
                if root not in GENERIC_ASPECTS:
                    aspects.append(chunk)
            opinions = [t for t in sent if t.lemma_.lower() in SENTIMENT_LEXICON and t.pos_ in {"ADJ", "VERB"}]
            for asp in aspects:
                evidences = []
                for op in opinions:
                    dist = abs(op.i - asp.root.i)
                    if dist > 7:
                        continue
                    base = SENTIMENT_LEXICON[op.lemma_.lower()]
                    prev = [t.text.lower() for t in doc[max(sent.start, op.i - 3) : op.i]]
                    neg = any(w in NEGATIONS for w in prev)
                    intens = next((w for w in reversed(prev) if w in INTENSIFIERS), None)
                    score = base * (INTENSIFIERS[intens] if intens else 1.0)
                    if neg:
                        score *= -1
                    score *= max(0.35, 1 - 0.08 * dist)
                    evidences.append(OpinionEvidence(op.text, base, round(score, 3), neg, intens, dist, source="v1"))
                avg = sum(e.adjusted_score for e in evidences) / len(evidences) if evidences else 0.0
                sentiment = "positive" if avg > 0.4 else "negative" if avg < -0.4 else "neutral"
                results.append(AspectSentiment(asp.text, sentiment, round(avg, 3), sent.text, evidences, self.name))
        return [r.to_dict() for r in results]
