# Aspect-Level Opinion Mining Lab (Explainable NLP + Streamlit)

A resume-ready internship project that performs **aspect-level opinion mining** on product reviews using a transparent, rule-based NLP pipeline.

## Problem Statement
Users often write one review containing mixed opinions across multiple product features (e.g., *"battery is great, camera is bad"*). Document-level sentiment misses this nuance.

This project extracts:
1. Product aspects (noun / noun phrases)
2. Aspect-specific sentiment (positive / negative / neutral)
3. Aggregated sentiment statistics across many reviews
4. Explainable evidence for every decision

---

## Core Features
- Single-review and multi-review analysis
- Full NLP preprocessing pipeline:
  - sentence segmentation
  - tokenization
  - lowercasing
  - lemmatization
  - stopword-aware aspect cleaning
- POS-based linguistic analysis
- Multi-word aspect extraction (`battery life`, `screen quality`)
- Rule-based sentiment with negation + intensifier handling
- Aspect-opinion proximity mapping for mixed-sentiment sentences
- Aggregation module with:
  - positive / negative / neutral counts
  - aspect frequency
  - dominant sentiment
  - average sentiment score
- Streamlit UI with **4 simultaneously runnable versions**

---

## Explainable Pipeline
1. **Input handling**: parse one or more reviews.
2. **Preprocessing**: spaCy sentence split + tokenization + lemmatization.
3. **Linguistic analysis**: POS tags identify noun-based aspects and adjective/verb opinions.
4. **Aspect extraction**: noun chunks + filtered nouns form candidate aspects.
5. **Sentiment scoring**: lexicon polarity adjusted by negation/intensifiers.
6. **Aspect-opinion association**: nearest-in-sentence opinions weighted by distance.
7. **Aggregation**: per-aspect mention counts and dominant sentiment across reviews.
8. **Presentation**: tables/cards/charts/JSON for technical and non-technical users.

---

## Four Distinct Versions in UI
The app supports four analysis variants with intentionally different output styles:

- **Version 1 – Balanced Rule Pipeline**
  - Default behavior
  - Dataframe + aggregation table + aspect-frequency bar chart

- **Version 2 – Conservative Precision Mode**
  - Keeps multi-word aspects and nearest evidence only
  - Card-style analyst summaries and confidence-style progress bars

- **Version 3 – Recall + Strength Emphasis**
  - Boosts repeated aspect mentions in a review
  - Metrics + area chart + weighted mention table

- **Version 4 – Contrast-Aware Briefing**
  - Highlights trade-off reviews containing contrast markers (`but`, `however`)
  - Review-by-review briefing blocks + JSON summary output

---

## Project Structure
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
│   ├── pipeline.py
│   └── variants.py
└── tests/
    └── test_pipeline.py
```

---

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

---

## Example Input (one review per line)
```text
Battery life is excellent and charging speed is fast, but the camera is disappointing in low light.
The screen quality is amazing. Speakers are weak and the phone feels heavy.
I love the design and display, however the software experience is not smooth.
```

## Example Output Shape
```json
{
  "aspect": "battery life",
  "frequency": 4,
  "positive": 3,
  "negative": 1,
  "neutral": 0,
  "avg_score": 0.91,
  "dominant_sentiment": "positive"
}
```

---

## Limitations
- Lexicon coverage is intentionally compact for explainability.
- Rule-based linking can miss implicit sentiment and sarcasm.
- Not dependency-tree perfect for long, complex sentences.

## Future Improvements
- Domain-specific aspect and sentiment lexicons (phones, laptops, cosmetics).
- Better coreference resolution (e.g., handling pronouns like *it*).
- Export pipeline results as API endpoints.
- Add lightweight confidence estimates and calibration checks.
