from __future__ import annotations

from collections import defaultdict

from .v3_lexicon_window import V3LexiconWindowMiner


class V4HybridEnsembleMiner:
    name = "V4 - Hybrid Ensemble Pipeline"

    def __init__(self):
        self.v3 = V3LexiconWindowMiner()
        self.v1 = None
        self.v2 = None
        try:
            from .v1_spacy_rule import V1SpacyRuleMiner
            from .v2_dependency_focus import V2DependencyMiner

            self.v1 = V1SpacyRuleMiner()
            self.v2 = V2DependencyMiner()
        except Exception:
            pass

    def analyze(self, text: str) -> list[dict]:
        all_rows = []
        all_rows.extend(self.v3.analyze(text))
        if self.v1:
            all_rows.extend(self.v1.analyze(text))
        if self.v2:
            all_rows.extend(self.v2.analyze(text))

        grouped = defaultdict(list)
        for r in all_rows:
            key = (r["aspect"].lower(), r["sentence"])
            grouped[key].append(r)

        final = []
        for (_, sentence), rows in grouped.items():
            aspect = rows[0]["aspect"]
            avg = sum(x["score"] for x in rows) / len(rows)
            label = "positive" if avg > 0.4 else "negative" if avg < -0.4 else "neutral"
            evidences = []
            for x in rows:
                evidences.extend(x.get("evidences", []))
            final.append(
                {
                    "aspect": aspect,
                    "sentiment": label,
                    "score": round(avg, 3),
                    "sentence": sentence,
                    "evidences": evidences[:5],
                    "approach": self.name,
                }
            )
        return final
