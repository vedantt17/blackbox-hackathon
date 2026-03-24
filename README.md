# BlackBox Hackathon — Digital Twin Pipeline

> Given everything we know about 233 real people from their survey responses, predict how each specific person would answer a brand new question they have never seen.

Built for the BlackBox Hackathon by UC Davis MSBA.

---

## What We Built

A hybrid ML + LLM pipeline that creates a "digital twin" for each of 233 people. The system routes each test question to the best prediction strategy based on how well it matches our training data.

```
Test Question
      │
      ▼
Semantic Similarity Check (all-MiniLM-L6-v2)
      │
      ├── sim >= 0.08 ──► ML Track (Random Forest ensemble)
      │                         └── 297 trained models × 15 personality scales
      │                         └── Top-50 correlated features per model
      │                         └── 5x bootstrap ensemble (person-level)
      │
      └── sim < 0.08  ──► LLM Track (Gemini 2.5 Flash)
                                └── Rich 5,500-char persona per person
                                └── 700+ actual Q&A pairs in context
                                └── Distribution hints to prevent neutral bias
```

---

## Dataset

4 surveys × 2 formats (labels + numeric) = 8 CSV files. 233 people × 914 features.

| Scale | Items | What it measures |
|---|---|---|
| Big Five (OCEAN) | 44 | Core personality traits |
| Need for Cognition | 18 | Thinking style |
| Risk / Loss Aversion | 42 each | Economic risk preferences |
| Beck Anxiety / Depression | 25 | Psychological state |
| Need for Closure | 15 | Tolerance for ambiguity |
| Cognitive Biases (S4) | ~50 | Sunk cost, framing, Linda problem, Allais |
| Trust / Ultimatum / Dictator | ~20 | Social trust behavior |
| Self-Monitoring | 13 | Social adaptability |

---

## Results

**ML Track (5-fold cross-validation):**

| Question Type | Majority Baseline | RF Accuracy | Lift |
|---|---|---|---|
| Need for Uniqueness | 27.9% | 39.9% | +12.0% |
| Extraversion | 27.5% | 38.6% | +11.1% |
| Agreeableness | 35.2% | 52.4% | +17.2% |
| Need for Closure | 45.5% | 55.4% | +9.9% |
| Maximization | 29.6% | 42.5% | +12.9% |
| **Average** | **33.1%** | **45.8%** | **+12.6%** |

**Key technical decisions validated by benchmarking:**
- Random Forest beats LightGBM on this dataset: 45.6% vs 44.6% (n=233 too small for boosting)
- Bootstrap ensemble adds +1.1% over single RF
- Sentence embeddings (all-MiniLM-L6-v2) better than TF-IDF for question routing

---

## Architecture

### ML Pipeline (`ml_pipeline.py`)

For each of 297 survey items across 15 personality scales:
1. Build feature matrix excluding same-scale columns (prevents leakage)
2. Select top-50 features by correlation with target
3. Train 5 Random Forest classifiers, each on a different bootstrap sample of the 233 people
4. At test time: embed new question → find most similar trained question → run person's features → weighted vote across bootstrap models

### LLM Pipeline (`predictor.py`)

For each person, build a structured persona:
- Full demographics (age, region, education, income, politics, religion)
- Big Five scores with computed trait labels + verbatim answers to key items
- Trust game dollar amounts (actual behavior, not just scores)
- Risk/loss lottery choices
- Cognitive bias outcomes (sunk cost, Linda, framing, Allais)
- Overconfidence score

Then send persona + test question to Gemini 2.5 Flash with distribution hints to prevent neutral defaulting.

### Semantic Routing

Used `all-MiniLM-L6-v2` (384-dim embeddings) instead of TF-IDF. This correctly matches "I judge people by their coffee order" to "I find fault with others" (Agreeableness item) even with zero word overlap.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers lightgbm

# Build (trains 297 models + builds personas, ~5-10 min)
python run.py --stage build
```

Set API keys in PowerShell:
```powershell
$env:GEMINI_API_KEY="your_key_from_aistudio.google.com"
$env:TEAM_TOKEN="your_hackathon_token"
```

---

## Usage

```bash
# Build: train all 297 models + build 233 personas
python run.py --stage build

# Predict: run on test questions JSON
python run.py --stage predict

# Eval: honest cross-validation on held-out questions
python run.py --stage eval

# Submit: send predictions to scoring API
python run.py --stage submit

# Test: smoke test, no API calls needed
python run.py --stage test
```

**Predict workflow:** drop the test JSON file in the project folder, then run predict. The pipeline automatically finds the JSON, routes each question, and saves two files:
- `outputs/predictions_TIMESTAMP.json` — full predictions with labels
- `outputs/submission_TIMESTAMP.json` — API-ready format `{person_id, question_id, predicted_answer: int}`

---

## File Structure

```
BlackBox Hackathon/
├── data/                    ← 8 CSV survey files (not committed)
├── cache/                   ← Auto-generated (not committed)
│   ├── personas.json        ← 233 rich text profiles
│   └── ml_pipeline.pkl      ← 297 trained models + semantic index
├── outputs/                 ← Predictions + submission JSONs
├── data_loader.py           ← Loads and aligns all 8 survey files
├── persona_builder.py       ← Builds rich Q&A personas per person
├── ml_pipeline.py           ← 297 RF models + semantic embedding index
├── predictor.py             ← LLM call logic (Gemini 2.5 Flash)
├── run.py                   ← Pipeline stages: build / predict / eval / submit
└── requirements.txt
```

---

## Why Not LightGBM? Why Not Bootstrap to Inflate Data?

**LightGBM:** Benchmarked both. RF averaged 45.6% accuracy vs LightGBM 44.6% on this dataset. LightGBM's gradient boosting overfits more aggressively at n=233. RF's bagging is more robust in the high feature-to-sample regime (914 features, 233 samples).

**Bootstrapping for more data:** Resampling 233 people to create 2000 rows does not create new information. Every new row is a copy of an existing person. Cross-validation scores inflate because the test set contains people whose duplicates were in training. The right use of bootstrapping is model stabilization (which we do, via 5 person-level bootstrap models per question), not data inflation.

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn RandomForestClassifier |
| Semantic Routing | sentence-transformers all-MiniLM-L6-v2 |
| LLM Provider | Google Gemini 2.5 Flash |
| Data Processing | pandas, numpy, scikit-learn |
| Pipeline | Custom run.py with build/predict/eval/submit stages |
