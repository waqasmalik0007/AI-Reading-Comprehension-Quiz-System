# Intelligent Reading Comprehension & Quiz Generation System

An AI-powered system built on the **RACE dataset** that automatically generates comprehension questions, predicts correct answers, creates distractor options, evaluates user responses, and provides graduated hints.

## Project Structure

```
AI_Project/
├── data/
│   ├── raw/                  # Original RACE CSV files (train, val, test)
│   └── processed/            # Cleaned data, feature matrices, vectorizers
├── models/
│   ├── model_a/traditional/  # Model A trained classifiers (LR, SVM, XGBoost, etc.)
│   └── model_b/traditional/  # Model B distractor ranker & hint scorer
├── src/
│   ├── preprocessing.py      # Dataset loading & feature engineering
│   ├── model_a_train.py      # Training script for Model A (Q&A Verifier)
│   ├── model_b_train.py      # Training script for Model B (Distractor & Hint Gen)
│   ├── inference.py          # Unified inference API
│   └── evaluate.py           # Metric computation
├── ui/
│   └── app.py                # Streamlit application (4 screens)
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis
├── tests/
├── requirements.txt
├── TASKS.md                  # Project task checklist
└── README.md
```

## Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd AI_Project

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Training Pipeline

Run the following commands **in order**:

```bash
# Step 1: Preprocess data (builds features from RACE dataset)
python -m src.preprocessing

# Step 2: Train Model A (Q&A Generator / Verifier)
python -m src.model_a_train

# Step 3: Train Model B (Distractor & Hint Generator)
python -m src.model_b_train

# Step 4: Evaluate all models on test set
python -m src.evaluate
```

## Running the Application

```bash
streamlit run ui/app.py
```

The app opens at `http://localhost:8501` with 4 screens:
1. **Article Input** — Paste text or load random RACE sample
2. **Quiz View** — Answer MCQs with color-coded feedback
3. **Hint Panel** — Graduated hints (general → specific → near-explicit)
4. **Dashboard** — Model metrics, latency tracking, CSV export

## Models Implemented

### Model A — Q&A Generator / Verifier
| Model | Task |
|-------|------|
| Logistic Regression | Answer verification |
| SVM (LinearSVC) | Answer verification |
| Naive Bayes | Question type classification |
| Random Forest | Difficulty estimation |
| XGBoost | Answer verification |
| K-Means | Unsupervised clustering |
| GMM | Probabilistic clustering |
| Label Propagation | Semi-supervised learning |
| Ensemble (Soft Voting + Stacking) | Combined verification |

### Model B — Distractor & Hint Generator
| Component | Method |
|-----------|--------|
| Distractor Extraction | Frequency-based candidate phrases |
| Distractor Ranking | Random Forest on cosine similarity features |
| Hint Scoring | Logistic Regression on sentence features |
| Diversity Filtering | Overlap-based penalty |

## Evaluation Metrics

- **Accuracy, Macro F1, Precision, Recall** — all models
- **Exact Match** — answer verification
- **Confusion Matrix** — all classifiers
- **Silhouette Score, Clustering Purity** — unsupervised models
- **Precision@K** — hint generation
- **R² Score** — hint relevance regression

## Dataset

**RACE** (ReAding Comprehension from Examinations) — Lai et al., 2017
- ~87,866 training samples, 4,887 validation, 4,934 test
- Multiple-choice questions (A/B/C/D) from Chinese English exams
- Source: [Hugging Face](https://huggingface.co/datasets/ehovy/race)

## Tech Stack

- **Python 3.13**, scikit-learn, XGBoost, Gensim, pandas, NumPy
- **Streamlit** for the web UI
- **Plotly / Matplotlib** for visualization
- **joblib** for model persistence
