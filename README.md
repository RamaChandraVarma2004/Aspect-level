# Aspect-Level Opinion Mining (Python + Streamlit)

A ready-to-run, modular NLP project that extracts product **aspects** from user reviews and predicts sentiment polarity for each aspect.

## Why this project is internship-friendly
- Uses an explicit and explainable NLP pipeline (easy to discuss in interviews)
- Handles multi-aspect sentences with mixed opinions
- Includes a clean UI for non-technical demos
- Prioritizes readable, production-style structure

## Features
- Sentence segmentation, tokenization, lemmatization (spaCy)
- POS-aware aspect and opinion extraction
- Multi-word aspect support (e.g., `battery life`, `camera quality`)
- Rule-based sentiment scoring (positive/negative/neutral)
- Negation and intensifier handling (e.g., `not good`, `very weak`)
- Aspect-opinion association with distance-based heuristics
- Clear UI table + JSON output + evidence trace

## Project structure

```text
.
├── app.py
├── requirements.txt
├── src/aspect_mining/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── lexicon.py
│   ├── aspect_extractor.py
│   ├── sentiment.py
│   ├── association.py
│   ├── schemas.py
│   └── pipeline.py
└── tests/
    └── test_pipeline.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

## Example input

> The battery life is amazing, but the camera quality is not good. I love the screen, though the speakers are very weak.

## Example output (shape)

```json
[
  {
    "aspect": "battery life",
    "sentiment": "positive",
    "score": 1.7,
    "sentence": "The battery life is amazing, but the camera quality is not good.",
    "evidences": [
      {
        "word": "amazing",
        "base_score": 2.5,
        "adjusted_score": 2.1,
        "negated": false,
        "intensifier": null,
        "distance": 2
      }
    ]
  }
]
```

## Design choices
1. **Modular architecture**: each NLP step has its own module for maintainability and explainability.
2. **Rule-first approach**: transparent rules beat black-box behavior for interview demos.
3. **Distance-aware association**: simple but practical for sentences containing multiple aspects.
4. **Evidence trace**: every sentiment label links to concrete opinion words and adjustments.

## Limitations
- Lexicon is intentionally compact; domain coverage can be expanded.
- Rules may miss implicit sentiment (e.g., sarcasm, nuanced context).
- Dependency heuristics are lightweight, not full relation extraction.

## Future improvements
- Add domain-adaptive sentiment lexicons (electronics, restaurants, etc.).
- Add confidence calibration and weak-supervision evaluation set.
- Train a hybrid reranker on top of rule outputs for harder cases.
- Export results to CSV/API endpoint for product integration.
