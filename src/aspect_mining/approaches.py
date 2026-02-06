from __future__ import annotations

from collections import defaultdict

from .lexicon import GENERIC_ASPECTS, INTENSIFIERS, NEGATIONS, SENTIMENT_LEXICON
from .schemas import AspectSentiment, OpinionEvidence


class BaseApproach:
    name = "base"
    description = "base"

    def _label(self, score: float) -> str:
        if score > 0.35:
            return "positive"
        if score < -0.35:
            return "negative"
        return "neutral"

    def _sentiment_for_token(self, sentence, tok, aspect_idx: int) -> OpinionEvidence | None:
        lemma = tok.lemma_.lower()
        if lemma not in SENTIMENT_LEXICON:
            return None
        score = SENTIMENT_LEXICON[lemma]
        left = sentence[max(sentence.start, tok.i - 3) : tok.i]
        left_words = [x.text.lower() for x in left]
        reason = ["lexicon"]
        if any(w in NEGATIONS for w in left_words):
            score *= -1
            reason.append("negation")
        found_int = next((w for w in reversed(left_words) if w in INTENSIFIERS), None)
        if found_int:
            score *= INTENSIFIERS[found_int]
            reason.append(f"intensifier:{found_int}")

        distance = abs(tok.i - aspect_idx)
        score *= max(0.35, 1 - (distance * 0.08))
        reason.append(f"distance:{distance}")
        return OpinionEvidence(word=tok.text, score=round(score, 3), reason=" + ".join(reason))

    def run(self, doc) -> list[AspectSentiment]:
        raise NotImplementedError


class Version1NearbyWindow(BaseApproach):
    name = "v1_nearby_window"
    description = "Noun chunks + nearby opinion words (simple and interview-friendly)."

    def _aspects(self, doc):
        seen, aspects = set(), []
        for ch in doc.noun_chunks:
            if ch.root.lemma_.lower() in GENERIC_ASPECTS:
                continue
            txt = ch.text.strip().lower()
            if txt not in seen:
                seen.add(txt)
                aspects.append(ch)
        return aspects

    def run(self, doc) -> list[AspectSentiment]:
        results = []
        for sentence in doc.sents:
            sentence_aspects = [a for a in self._aspects(doc) if a.start >= sentence.start and a.end <= sentence.end]
            opinion_tokens = [t for t in sentence if t.pos_ in {"ADJ", "VERB"}]
            for asp in sentence_aspects:
                evidences = []
                for tok in opinion_tokens:
                    if abs(tok.i - asp.root.i) <= 7:
                        ev = self._sentiment_for_token(sentence, tok, asp.root.i)
                        if ev:
                            evidences.append(ev)
                score = sum(e.score for e in evidences) / len(evidences) if evidences else 0.0
                results.append(
                    AspectSentiment(
                        aspect=asp.text,
                        sentiment=self._label(score),
                        score=round(score, 3),
                        sentence=sentence.text,
                        evidences=sorted(evidences, key=lambda x: abs(x.score), reverse=True),
                        approach=self.name,
                    )
                )
        return results


class Version2DependencyRules(BaseApproach):
    name = "v2_dependency_rules"
    description = "Dependency-linked aspect-opinion mapping (amod/acomp/attr)."

    def run(self, doc) -> list[AspectSentiment]:
        by_aspect = defaultdict(list)
        for tok in doc:
            if tok.pos_ != "NOUN" or tok.lemma_.lower() in GENERIC_ASPECTS:
                continue

            for child in tok.children:
                if child.dep_ in {"amod", "acomp", "attr", "advcl"} or child.pos_ in {"ADJ", "VERB"}:
                    ev = self._sentiment_for_token(tok.sent, child, tok.i)
                    if ev:
                        by_aspect[(tok.text, tok.sent.text)].append(ev)

            if tok.head and tok.head.pos_ in {"ADJ", "VERB"}:
                ev = self._sentiment_for_token(tok.sent, tok.head, tok.i)
                if ev:
                    by_aspect[(tok.text, tok.sent.text)].append(ev)

        results = []
        for (aspect, sentence), evidences in by_aspect.items():
            score = sum(e.score for e in evidences) / len(evidences)
            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=self._label(score),
                    score=round(score, 3),
                    sentence=sentence,
                    evidences=evidences,
                    approach=self.name,
                )
            )
        return results


class Version3ClauseAware(BaseApproach):
    name = "v3_clause_aware"
    description = "Clause-aware matching around conjunctions for mixed sentiment sentences."

    def run(self, doc) -> list[AspectSentiment]:
        results = []
        for sentence in doc.sents:
            boundaries = [sentence.start]
            boundaries += [t.i + 1 for t in sentence if t.dep_ == "cc" or t.text.lower() in {"but", "although", "though"}]
            boundaries.append(sentence.end)
            boundaries = sorted(set(boundaries))

            for i in range(len(boundaries) - 1):
                clause = doc[boundaries[i] : boundaries[i + 1]]
                aspects = [t for t in clause if t.pos_ == "NOUN" and t.lemma_.lower() not in GENERIC_ASPECTS]
                opinions = [t for t in clause if t.pos_ in {"ADJ", "VERB"}]
                for asp in aspects:
                    evidences = [self._sentiment_for_token(sentence, op, asp.i) for op in opinions]
                    evidences = [e for e in evidences if e]
                    score = sum(e.score for e in evidences) / len(evidences) if evidences else 0.0
                    results.append(
                        AspectSentiment(
                            aspect=asp.text,
                            sentiment=self._label(score),
                            score=round(score, 3),
                            sentence=sentence.text,
                            evidences=evidences,
                            approach=self.name,
                        )
                    )
        return results


class Version4EnsembleConsensus(BaseApproach):
    name = "v4_ensemble_consensus"
    description = "Ensemble of v1+v2+v3 with consensus score (best practical default)."

    def __init__(self):
        self.v1 = Version1NearbyWindow()
        self.v2 = Version2DependencyRules()
        self.v3 = Version3ClauseAware()

    def run(self, doc) -> list[AspectSentiment]:
        combined = self.v1.run(doc) + self.v2.run(doc) + self.v3.run(doc)
        grouped = defaultdict(list)
        for item in combined:
            grouped[(item.aspect.lower(), item.sentence)].append(item)

        results = []
        for (aspect_key, sentence), items in grouped.items():
            scores = [x.score for x in items]
            avg = sum(scores) / len(scores)
            evidences = []
            for it in items:
                evidences.extend(it.evidences[:1])
            aspect_name = items[0].aspect
            results.append(
                AspectSentiment(
                    aspect=aspect_name,
                    sentiment=self._label(avg),
                    score=round(avg, 3),
                    sentence=sentence,
                    evidences=evidences,
                    approach=self.name,
                )
            )
        return results


APPROACHES = {
    Version1NearbyWindow.name: Version1NearbyWindow,
    Version2DependencyRules.name: Version2DependencyRules,
    Version3ClauseAware.name: Version3ClauseAware,
    Version4EnsembleConsensus.name: Version4EnsembleConsensus,
}
