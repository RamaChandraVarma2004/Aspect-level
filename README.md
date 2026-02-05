# Aspect-Level Opinion Mining (Python + Streamlit)

A ready-to-run, internship-to-entry-level **Aspect-Based Sentiment Analysis (ABSA)** project.
It extracts product aspects (including multi-word aspects like `battery life`) and predicts sentiment per aspect using interpretable linguistic rules.

## Why this project is resume-worthy
- **Modular NLP pipeline** (preprocessing → features → aspects → association → output)
- **Explainable rules**, not a black-box classifier
- **Product-minded UI** for non-technical reviewers
- Handles **mixed opinions** in the same review

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL (usually `http://localhost:8501`).

> On first run, spaCy model (`en_core_web_sm`) and NLTK lexicon are auto-downloaded if missing.

---

## Project Structure

```text
.
├── app.py                          # Streamlit UI
├── requirements.txt
├── src/aspect_miner/
│   ├── __init__.py
│   ├── nlp_loader.py               # model + lexicon loading
│   ├── preprocessing.py            # segmentation/tokenization/lemmatization/stopwords
│   ├── features.py                 # POS-level opinion token detection
│   ├── aspect_extractor.py         # noun-based aspect extraction
│   ├── association.py              # aspect-opinion linkage + sentiment label
│   └── pipeline.py                 # orchestrates all modules
└── tests/
    └── test_pipeline.py
```

---

## Pipeline Design

1. **Text preprocessing**
   - sentence segmentation
   - tokenization
   - lemmatization (via spaCy token lemmas)
   - stopword handling for cleaner aspect candidates

2. **Linguistic features**
   - POS tagging from spaCy
   - noun chunks as aspect candidates
   - opinion-bearing words from lexicon + POS constraints

3. **Aspect extraction**
   - noun chunks + compound noun rules
   - supports multi-word aspects (`screen quality`, `battery life`)
   - filters generic low-information terms (`thing`, `stuff`, `product`)

4. **Opinion & sentiment rules**
   - positive/negative wordlists (NLTK opinion lexicon)
   - negation handling (`not good` → negative)
   - intensifier scaling (`very bad` stronger negative)

5. **Aspect-opinion association**
   - links each aspect to nearby opinion words by token distance
   - weighted contribution based on proximity
   - outputs sentiment class: `positive`, `negative`, `neutral`

6. **Output representation**
   - UI table for quick interpretation
   - structured JSON for downstream use

---

## Example

Input:

> "The battery life is amazing, but the screen quality is not good."

Potential output (simplified):

```json
[
  {"aspect": "battery life", "sentiment": "positive", "score": 0.50},
  {"aspect": "screen quality", "sentiment": "negative", "score": -0.50}
]
```

---

## Limitations
- Rule-based associations can miss long-range dependencies.
- Domain-specific words (e.g., slang) may require lexicon expansion.
- No coreference resolution (e.g., "It is great" after mentioning a product aspect).

## Future Improvements
- Add dependency-path based association as a stronger syntactic heuristic.
- Add domain adaptation (electronics, restaurants, apps).
- Add confidence metrics and evaluation dataset.
- Optional hybrid upgrade with a lightweight transformer baseline.

