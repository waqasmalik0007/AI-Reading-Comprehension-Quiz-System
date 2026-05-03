# AI Lab Project - Task Checklist
## Intelligent Reading Comprehension & Quiz Generation System (RACE Dataset)
### Deadline: May 11, 2026 at 08:30 AM

---

## Phase 1: Setup & Data (Days 1-2) [10 marks]
- [x] 1.1 Download RACE dataset from Hugging Face (train.csv, test.csv, val.csv) → placed in `data/raw/`
- [x] 1.2 Install all dependencies (scikit-learn, pandas, numpy, gensim, xgboost, streamlit, plotly, joblib, matplotlib, sentence-transformers)
- [x] 1.3 Create `requirements.txt` with pinned versions
- [x] 1.4 EDA notebook (`notebooks/EDA.ipynb`) — distributions, passage lengths, question types, answer balance, summary stats
- [x] 1.5 `src/preprocessing.py` — lowercasing, punctuation removal, One-Hot Encoding, TF-IDF (optional), cosine similarity matrix, train/val/test split management

---

## Phase 2: Model A — Q&A Generator / Verifier [15 + 20 + 5 = 40 marks]

### Traditional ML (15 marks)
- [x] 2.1 Feature engineering: One-Hot Encoding of (article + question + option), cosine similarity features, handcrafted lexical features
- [x] 2.2 Logistic Regression for answer verification — train, evaluate (Accuracy, F1)
- [x] 2.3 SVM for answer verification — train, evaluate
- [x] 2.4 Naive Bayes for question type classification
- [x] 2.5 Random Forest for difficulty estimation
- [x] 2.6 Comparison table of all Model A traditional ML models

### Unsupervised / Semi-Supervised (20 marks)
- [x] 2.7 K-Means Clustering on question-answer pairs (One-Hot or TF-IDF features)
- [x] 2.8 Label Propagation (semi-supervised) — small labeled set → propagate to unlabeled
- [x] 2.9 GMM (Gaussian Mixture Models) — probabilistic clustering
- [x] 2.10 Evaluate: clustering purity, silhouette score, semi-supervised F1 vs supervised baselines

### Template-Based Question Generation with ML Ranking
- [x] 2.11 Extract candidate sentences using keyword overlap
- [x] 2.12 Apply Wh-word templates (Who/What/Where/When/Why)
- [x] 2.13 Rank generated questions using SVM or Random Forest classifier

### Ensemble (5 marks)
- [x] 2.14 Implement ensemble (soft voting / hard voting / stacking)
- [x] 2.15 Show improvement over individual models

### Evaluation Metrics for Model A
- [x] 2.16 Report: Accuracy, Macro F1, Exact Match on test set

---

## Phase 3: Model B — Distractor & Hint Generator [15 + 10 = 25 marks]

### Distractor Generation (15 marks)
- [x] 3.1 Candidate extraction from passage (string matching, frequency-based)
- [x] 3.2 Feature engineering: One-Hot cosine similarity to answer, character match score, passage frequency
- [x] 3.3 ML Ranker (Logistic Regression or Random Forest) to select top-3 distractors
- [x] 3.4 Alternative: Word2Vec nearest neighbours for distractors (coded, optional download)
- [x] 3.5 Diversity penalty to ensure variation across distractors
- [x] 3.6 Evaluate: Precision, Recall, F1, Confusion Matrix, Accuracy

### Hint Generation (10 marks)
- [x] 3.7 Extractive hint generation — score sentences by relevance to question (cosine similarity)
- [x] 3.8 ML-scored hints — Logistic Regression on sentence features (keyword overlap, position, length)
- [x] 3.9 Graduated hints: Hint 1 (general), Hint 2 (specific), Hint 3 (near-explicit)
- [x] 3.10 Evaluate: Precision@K, R² score

---

## Phase 4: User Interface — Streamlit [15 marks]
- [x] 4.1 Screen 1 — Article Input (text area, load random RACE sample, Submit button)
- [x] 4.2 Screen 2 — Quiz View (question + 4 options, Check button, color-coded result)
- [x] 4.3 Screen 3 — Hint Panel (collapsible, graduated hints, Reveal Answer button)
- [x] 4.4 Screen 4 — Developer/Analytics Dashboard (metrics, confusion matrix, latency, CSV export)
- [x] 4.5 UX: loading indicators, error handling, accessible design

---

## Phase 5: Documentation & Submission [5 + 5 = 10 marks]

### Final Report (5 marks)
- [x] 5.1 Abstract (200 words max)
- [x] 5.2 Introduction & Motivation
- [x] 5.3 Related Work (cite ≥5 papers)
- [x] 5.4 Dataset Analysis
- [x] 5.5 Model A: Design, Training, Results
- [x] 5.6 Model B: Design, Training, Results
- [x] 5.7 User Interface Description
- [x] 5.8 Evaluation & Discussion
- [x] 5.9 Limitations & Future Work
- [x] 5.10 Conclusion + References

### Code Quality (5 marks)
- [x] 5.11 Clean, documented code
- [x] 5.12 README.md with setup, training, and run instructions
- [x] 5.13 Meaningful commit history (GitHub)

---

## Submission Checklist
- [x] GitHub repository (clean commit history)
- [x] requirements.txt with pinned versions
- [x] README.md with setup/training/run instructions
- [x] EDA notebook (notebooks/EDA.ipynb)
- [x] Trained model checkpoints (in models/ — gitignored, regenerated via pipeline)
- [x] Final report (report/final_report.md — convert to PDF before submission)
- [x] 10-minute live demo session ready (streamlit run ui/app.py)
