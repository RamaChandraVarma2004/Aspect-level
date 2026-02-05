# Aspect-Level Opinion Mining (Python + Streamlit)

A ready-to-run, resume-friendly NLP project that extracts **aspects** from user reviews and predicts **sentiment per aspect**.

## Why this project is internship-ready
- Uses a **modular pipeline** instead of one script.
- Uses practical NLP fundamentals: tokenization, POS tagging, lemmatization, noun phrase extraction.
- Handles **multi-aspect sentences** and **mixed sentiment**.
- Uses interpretable rules for **negation** and **intensifiers**.
- Ships with a clean UI and structured JSON output for product/demo usage.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── src/
│   └── aspect_mining/
│       ├── __init__.py
│       ├── config.py
│       ├── nlp_resources.py
│       ├── preprocessing.py
│       ├── aspect_extractor.py
│       ├── sentiment.py
│       ├── association.py
│       ├── pipeline.py
│       └── schemas.py
└── README.md
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open: http://localhost:8501

> On first run, NLTK resources are auto-downloaded by the pipeline.

## Example Input

```text
The battery life is amazing, but the screen quality is not good in sunlight.
I love the camera, although the charging speed is very slow.
```

## Example Output (table)

| Aspect           | Sentiment | Score | Opinion Words |
|------------------|-----------|-------|---------------|
| battery life     | positive  | 2.0   | amazing       |
| screen quality   | negative  | -1.9  | good          |
| camera           | positive  | 3.2   | love          |
| charging speed   | negative  | -2.6  | slow          |

## Design Choices
- **Aspect extraction**: regex noun-phrase chunking (`JJ* + NN+`) to capture multi-word aspects.
- **Opinion scoring**: VADER lexicon score at token level.
- **Negation handling**: flips polarity if a nearby negation exists (`not`, `never`, `n't`, etc.).
- **Intensifier handling**: scales polarity (`very`, `extremely`, `slightly`, etc.).
- **Aspect-opinion linking**: nearest-aspect rule within configurable token window.

## Limitations
- Rule-based extraction may miss implicit aspects (e.g., "gets hot quickly" -> "thermals").
- Nearest-word association can fail in complex syntax.
- Domain adaptation is limited without custom lexicons.

## Future Improvements
- Dependency parsing for stronger aspect-opinion links.
- Hybrid approach with transformer classifier for difficult cases.
- Feedback loop: let users correct results to improve rules.
- Export analysis history as CSV/JSON for product analytics.

## Resume Bullet Suggestion
Built an explainable aspect-level opinion mining app (Python, NLTK, Streamlit) that extracts noun-phrase aspects from customer reviews, handles negation/intensifiers, and produces per-aspect sentiment in both table and JSON formats.
