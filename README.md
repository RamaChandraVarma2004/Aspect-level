# Aspect-Level Opinion Mining (4 Versions, Internship-Ready)

A resume-friendly NLP project for **Aspect-Level Opinion Mining** with **4 different explainable approaches** and a clean Streamlit UI.

## What makes this recruiter-friendly?
- Modular architecture with separable NLP stages.
- Four approaches to compare trade-offs (great for mentor discussions).
- Explainable per-aspect evidence (opinion words, negation, intensifiers, distance).
- Professional UI with single-version and all-version comparison modes.

## 4 Versions (Different Approaches)
1. **V1 – Proximity Rules**
   - Baseline: associate aspect with nearby opinion words in a token window.
2. **V2 – Dependency Rules (Recommended)**
   - Uses syntactic links first; fallback to local window.
   - Better for mixed opinions in same sentence.
3. **V3 – Hybrid (Sentence + Local)**
   - Combines overall sentence sentiment with aspect-local sentiment.
4. **V4 – Clause-Aware Contrastive**
   - Adds handling for contrast words like *but/however/though*.

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
│   ├── approaches.py
│   ├── schemas.py
│   └── pipeline.py
└── tests/
    ├── conftest.py
    └── test_pipeline.py
```

## Download & Run

### Option A: Download ZIP
1. Click **Code → Download ZIP** in GitHub.
2. Extract and open terminal in that folder.

### Option B: Clone
```bash
git clone <repo-url>
cd Aspect-level
```

### Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Launch UI
```bash
streamlit run app.py
```

## Example Input
> The battery life is amazing, but the camera quality is not good. I love the screen, though the speakers are very weak.

## Output Format
Each aspect entry includes:
- `aspect`
- `sentiment` (positive/negative/neutral)
- `score`
- `confidence`
- `method`
- `sentence`
- `evidences[]` (word-level explainability)

## Design Notes
- **Interpretability first**: every decision is rule-traceable.
- **Modular pipeline**: easy to refactor or extend with ML model later.
- **Intern-level clarity**: concise modules, simple heuristics, practical UX.

## Limitations
- Rule-based methods can miss sarcasm and deep context.
- Domain lexicon is small by design (easy to extend).
- Dependency parse quality affects V2/V4 in noisy text.

## Future Improvements
- Add domain packs (electronics/hotel/food lexicons).
- Introduce confidence calibration and benchmark dataset evaluation.
- Add optional transformer reranker for edge cases.
