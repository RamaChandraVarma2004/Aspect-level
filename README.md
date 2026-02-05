# Aspect-Level Opinion Mining (Python + Flask + spaCy)

A ready-to-run, internship-friendly project that extracts product aspects from reviews and assigns sentiment to each aspect with explainable rules.

## Why this project is resume-worthy
- **Modular NLP pipeline**: preprocessing, features, aspects, sentiment, and aspect-opinion association are all separated.
- **Explainable behavior**: each sentiment decision stores lexical evidence (`base`, `negation`, `intensifier`).
- **Product mindset**: clean web UI + JSON API endpoint for integration.

## Features
- Sentence segmentation, tokenization, lemmatization, stopword handling (spaCy).
- POS/dependency-informed aspect extraction with multi-word aspects (e.g., `battery life`, `screen quality`).
- Rule-based sentiment classes: **positive / negative / neutral**.
- Negation + intensifier handling (`not good`, `very bad`, etc.).
- Aspect-opinion linking in sentences with multiple aspects.
- Table output and structured JSON output.

## Project structure

```text
Aspect-level/
├── app.py
├── aspect_miner/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── aspect_extraction.py
│   ├── sentiment.py
│   ├── association.py
│   ├── pipeline.py
│   └── schemas.py
├── templates/index.html
├── static/styles.css
├── tests/test_pipeline.py
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open: `http://localhost:5000`

> Note: if `en_core_web_sm` is missing, the app auto-downloads it once.

## Example input

```text
The battery life is great but the screen quality is not good.
I love the camera, yet the speakers are very weak and disappointing.
```

## Example output (simplified)

```json
{
  "aspect_results": [
    {"aspect": "battery life", "sentiment": "positive"},
    {"aspect": "screen quality", "sentiment": "negative"},
    {"aspect": "camera", "sentiment": "positive"},
    {"aspect": "speaker", "sentiment": "negative"}
  ]
}
```

## Design choices
1. **spaCy parser + rule-based logic** instead of heavy neural ABSA models to keep behavior explainable.
2. **Noun chunks + compound nouns** for extracting meaningful aspects.
3. **Small explicit sentiment lexicon** for transparent scoring and easier debugging.
4. **Proximity + dependency association** to handle mixed sentiment in one sentence.

## Limitations
- Domain-general lexicon is intentionally compact, so some opinion words may be missed.
- Coreference and long-range sentiment links are not fully handled.
- Irony/sarcasm handling is out of scope.

## Future improvements
- Expand lexicon via domain adaptation from labeled data.
- Add confidence scores from weak supervision.
- Optional transformer fallback (e.g., ABSA model) while preserving rule explanations.
- Add export to CSV and dashboard metrics.

## API usage

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"review":"Battery life is great but charging speed is slow."}'
```
