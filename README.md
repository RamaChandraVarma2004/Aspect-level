# Aspect-Level Opinion Mining (Python + Streamlit)

A ready-to-run, internship-level and resume-ready NLP project for **Aspect-Level Opinion Mining** on user reviews.

## What makes this recruiter-friendly
- ✅ Explainable NLP pipeline (no black-box-only logic)
- ✅ 4 progressively stronger versions (easy to discuss in interviews)
- ✅ Clean Streamlit UI for mentor/recruiter demos
- ✅ Modular code structure with readable components

## 4 versions (different approaches)

### **V1 — Proximity Rules (Baseline)**
- Extract aspects (noun chunks + noun dependencies)
- Find nearby opinion words
- Score with lexicon + negation + intensifier
- Best for understanding fundamentals quickly

### **V2 — Dependency-Aware Association**
- Same extraction/scoring core as V1
- Associates aspect-opinion using dependency links first
- Falls back to proximity if syntax relation is weak
- Better for multi-aspect sentences than plain windowing

### **V3 — Contrast-Aware Clauses**
- Adds clause split logic around `but`, `however`, `though`, etc.
- Tries to prevent wrong cross-clause attachment
- Useful for mixed opinions in one sentence

### **V4 — Ensemble (Recommended)**
- Runs V1 + V2 + V3 and averages scores
- Produces more stable and mentor-friendly outputs
- **Default recommended version** for demos

---

## Project structure

```text
.
├── app.py
├── requirements.txt
├── src/aspect_mining/
│   ├── preprocess.py
│   ├── features.py
│   ├── aspect_extractor.py
│   ├── sentiment.py
│   ├── association.py      # V1/V2/V3/V4 association logic
│   ├── pipeline.py         # versioned orchestration
│   ├── lexicon.py
│   ├── schemas.py
│   └── __init__.py
└── tests/
    ├── conftest.py
    └── test_pipeline.py
```

## How to download and run

### Option A: Download ZIP from GitHub
1. Open repo on GitHub
2. Click **Code → Download ZIP**
3. Extract ZIP
4. Open terminal inside the extracted folder

### Option B: Clone with git
```bash
git clone <YOUR_REPO_URL>
cd Aspect-level
```

### Setup + run
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## UI usage
- Paste review text
- Choose:
  - **Run recommended version (V4)**, or
  - **Compare all 4 versions**
- Click **Analyze review**
- View results in table + JSON

## Example review
> The battery life is amazing, but the camera quality is not good. I love the display, though the speakers are very weak.

## Design choices (short)
- Keep extraction/association/scoring separate for transparency.
- Use deterministic rules to remain explainable for internship screening.
- Support multi-aspect + mixed-opinion sentences with syntax/clause heuristics.
- Provide version comparison to show product thinking and experimentation.

## Limitations
- Rule systems can miss sarcasm and subtle context.
- Lexicon coverage is finite and domain-sensitive.
- Dependency parsing errors may propagate.

## Future improvements
- Domain lexicons (electronics/hospitality, etc.)
- Confidence estimates + evaluation on labeled ABSA datasets
- Optional ML reranker over rule outputs
- API endpoint + export workflows
