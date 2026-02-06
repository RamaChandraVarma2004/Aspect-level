# Aspect-Level Opinion Mining — 4 Different Approaches (Internship/Resume Ready)

This project provides **4 versions** of Aspect-Level Opinion Mining (ALOM), each with a different design philosophy so you can show mentors/recruiters breadth and explainability.

## What makes this recruiter-friendly
- Modular, readable architecture
- Evidence-backed predictions (word-level sentiment rationale)
- Multiple approaches with tradeoff discussion
- Streamlit UI for quick demo

## The 4 versions

1. **V1 — spaCy Rule Pipeline**
   - Noun chunks for aspects
   - Lexicon-based opinion words
   - Distance + negation + intensifier scoring

2. **V2 — Dependency Relation Pipeline**
   - Dependency links between nouns and adjective/verb opinions
   - Better precision on multi-aspect sentences

3. **V3 — Lightweight Lexicon+Window Pipeline**
   - Pure Python fallback (no spaCy runtime)
   - Fast and simple, great for explaining baseline logic

4. **V4 — Hybrid Ensemble Pipeline (Recommended)**
   - Combines outputs of V1/V2/V3
   - Uses consensus average for robust sentiment
   - **Best default for demos**

## Download / Run

### Option A: Download ZIP
1. GitHub → `Code` → `Download ZIP`
2. Extract and open terminal inside project folder

### Option B: Clone
```bash
git clone <repo-url>
cd Aspect-level
```

### Setup and launch
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Usage
- Choose `single` mode (one version) or `compare-all` mode (all 4 tabs).
- Paste review text and click **Analyze**.
- Inspect table + JSON for explainable per-aspect output.

## Example input
> The battery life is amazing, but the camera quality is not good. I love the screen, though the speakers are very weak.

## Design tradeoffs
- V1: easy to understand, strong baseline.
- V2: syntax-aware, often more precise.
- V3: dependency-free fallback, easiest to maintain.
- V4: most stable for real demos and mixed reviews.

## Limitations
- Lexicon size is limited; domain adaptation improves quality.
- No sarcasm/implicit sentiment understanding.
- English-focused heuristics.

## Future upgrades
- Domain-specific lexicon packs (phones, hotels, restaurants)
- Confidence scoring + calibration
- Small labeled benchmark and evaluation dashboard
