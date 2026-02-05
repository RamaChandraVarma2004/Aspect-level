"""End-to-end modular pipeline for aspect-level opinion mining."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .aspect_extractor import AspectExtractor
from .association import AspectOpinionAssociator
from .features import LinguisticFeatureExtractor
from .nlp_loader import load_opinion_lexicons, load_spacy_model
from .preprocessing import TextPreprocessor


@dataclass
class SentenceResult:
    sentence: str
    aspects: list[dict]


class AspectOpinionPipeline:
    """Coordinates all NLP components into an explainable workflow."""

    def __init__(self):
        nlp = load_spacy_model()
        positive, negative = load_opinion_lexicons()
        self.nlp = nlp
        self.preprocessor = TextPreprocessor()
        self.features = LinguisticFeatureExtractor(positive, negative)
        self.aspect_extractor = AspectExtractor()
        self.associator = AspectOpinionAssociator()

    def analyze(self, text: str) -> dict:
        doc = self.nlp(text)
        prep_sentences = self.preprocessor.preprocess(doc)

        sentence_results: list[SentenceResult] = []
        all_aspects: list[dict] = []

        for sent in prep_sentences:
            chunks = self.features.extract_noun_chunks(sent.span)
            aspects = self.aspect_extractor.extract(sent.span, chunks)
            opinions = self.features.detect_opinion_tokens(sent.span)
            associated = self.associator.associate(aspects, opinions)

            aspect_dicts = [asdict(item) for item in associated]
            all_aspects.extend(aspect_dicts)
            sentence_results.append(SentenceResult(sentence=sent.text, aspects=aspect_dicts))

        return {
            "input_text": text,
            "sentences": [asdict(s) for s in sentence_results],
            "aspects": all_aspects,
        }
