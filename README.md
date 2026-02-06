# Aspect-Level Opinion Mining: 4 Versions (Internship + Resume Ready)

This project gives you **4 different approaches** for aspect-level opinion mining in one clean app.

## What mentors/recruiters will like
- Clear modular architecture (easy to explain in interviews)
- Multiple approaches and trade-off thinking
- Interpretable outputs with evidence words and rules
- Product-style UI for demos

## The 4 versions
1. **v1_nearby_window**
   - Noun chunks for aspects + nearby opinion words.
   - Best for fast baseline and easy understanding.

2. **v2_dependency_rules**
   - Uses dependency relations (`amod`, `acomp`, `attr`) for aspect-opinion links.
   - Better linguistic precision.

3. **v3_clause_aware**
   - Splits mixed-opinion sentences into clauses around conjunctions.
   - Better for "A is good but B is bad" patterns.

4. **v4_ensemble_consensus** (**recommended default**)
   - Combines v1 + v2 + v3 and averages sentiment scores.
   - Most balanced for practical demo quality.

## Quick start (download + run)

### Option A: Download ZIP
1. GitHub -> **Code** -> **Download ZIP**
2. Extract
3. Run:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

### Option B: Clone
```bash
git clone <repo-url>
cd Aspect-level
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Example review
"Battery life is excellent, but camera quality is not good. The screen is amazing although the speakers are very weak."

## Project structure

```text
src/aspect_mining/
  preprocess.py
  lexicon.py
  schemas.py
  approaches.py
  pipeline.py
app.py
tests/
```

## Design notes
- Rule-based design is intentional for explainability.
- Negation and intensifiers are explicit and auditable.
- Ensemble version is practical while staying interpretable.

## Limitations
- Lexicon is compact (extend by domain).
- Not designed for sarcasm or deeply implicit sentiment.

## Future improvements
- Domain-specific lexicons (electronics/hospitality).
- Optional transformer reranker on top of rules.
- CSV/API export and confidence scoring.
