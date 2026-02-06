from __future__ import annotations

from ..lexicon import SENTIMENT_LEXICON, NEGATIONS, INTENSIFIERS, GENERIC_ASPECTS
from ..schemas import AspectSentiment, OpinionEvidence


class V2DependencyMiner:
    name = "V2 - Dependency Relation Pipeline"

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy

            self.nlp = spacy.load(model_name)
        except Exception as exc:
            raise RuntimeError("V2 requires spaCy model en_core_web_sm") from exc

    def analyze(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        out: list[AspectSentiment] = []
        for sent in doc.sents:
            for tok in sent:
                if tok.pos_ != "NOUN" or tok.lemma_.lower() in GENERIC_ASPECTS:
                    continue
                evidence = []
                for child in tok.children:
                    if child.lemma_.lower() in SENTIMENT_LEXICON and child.pos_ in {"ADJ", "VERB"}:
                        evidence.append(self._e(sent, child, tok.i))
                if tok.head.lemma_.lower() in SENTIMENT_LEXICON and tok.head.pos_ in {"ADJ", "VERB"}:
                    evidence.append(self._e(sent, tok.head, tok.i))
                score = sum(e.adjusted_score for e in evidence) / len(evidence) if evidence else 0.0
                sentiment = "positive" if score > 0.4 else "negative" if score < -0.4 else "neutral"
                out.append(AspectSentiment(tok.text, sentiment, round(score, 3), sent.text, evidence, self.name))
        return [x.to_dict() for x in out]

    def _e(self, sent, op_tok, center):
        base = SENTIMENT_LEXICON[op_tok.lemma_.lower()]
        prev = [t.text.lower() for t in sent[max(sent.start, op_tok.i - 3) : op_tok.i]]
        neg = any(w in NEGATIONS for w in prev)
        intens = next((w for w in reversed(prev) if w in INTENSIFIERS), None)
        score = base * (INTENSIFIERS[intens] if intens else 1)
        if neg:
            score *= -1
        dist = abs(op_tok.i - center)
        score *= max(0.4, 1 - dist * 0.1)
        return OpinionEvidence(op_tok.text, base, round(score, 3), neg, intens, dist, source="v2")
