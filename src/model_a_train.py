"""
model_a_train.py — Training script for Model A: Q&A Generator / Verifier

This script trains and evaluates:
  1. Logistic Regression (answer verification)
  2. SVM (answer verification)
  3. Naive Bayes (question type classification)
  4. Random Forest (difficulty estimation)
  5. XGBoost (answer verification)
  6. K-Means Clustering (unsupervised)
  7. Label Propagation (semi-supervised)
  8. GMM (Gaussian Mixture Models)
  9. Ensemble (soft voting / stacking)
  10. Template-based question generation with ML ranking
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import load_processed_data, load_raw_data, clean_text

# Directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_a', 'traditional')
DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_a', 'checkpoints')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(name, model, metadata=None, scaler=None):
    """Save a model checkpoint with optional metadata and scaler."""
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    ckpt = {
        'model': model,
        'scaler': scaler,
        'metadata': metadata or {},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(CHECKPOINT_DIR, f'ckpt_{safe_name}.pkl')
    joblib.dump(ckpt, path)
    print(f"  [Checkpoint] Saved: {path}")
    return path


def load_checkpoint(name):
    """Load a previously saved checkpoint. Returns (model, scaler, metadata) or None."""
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(CHECKPOINT_DIR, f'ckpt_{safe_name}.pkl')
    if os.path.exists(path):
        ckpt = joblib.load(path)
        print(f"  [Checkpoint] Loaded: {path} (saved {ckpt.get('timestamp', 'unknown')})")
        return ckpt['model'], ckpt.get('scaler'), ckpt.get('metadata', {})
    return None, None, None


def load_features():
    """Load preprocessed feature matrices."""
    data = load_processed_data()
    return data


def evaluate_model(name, y_true, y_pred):
    """Print evaluation metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*50}")

    return {'name': name, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}


# ──────────────────────────────────────────────────────────────
# TASK 2.1: Feature engineering already done in preprocessing.py
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# TASK 2.2: LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression for answer verification."""
    print("\n[Training] Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val_scaled)
    metrics = evaluate_model("Logistic Regression", y_val, y_pred)
    metrics['train_time'] = train_time

    joblib.dump(model, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'lr_scaler.pkl'))
    save_checkpoint('logistic_regression', model, metadata=metrics, scaler=scaler)
    print(f"  Train time: {train_time:.2f}s")

    return model, scaler, metrics


# ──────────────────────────────────────────────────────────────
# TASK 2.3: SVM
# ──────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM for answer verification."""
    print("\n[Training] Support Vector Machine (LinearSVC)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LinearSVC(max_iter=2000, C=1.0, random_state=42, dual='auto')
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val_scaled)
    metrics = evaluate_model("SVM (LinearSVC)", y_val, y_pred)
    metrics['train_time'] = train_time

    joblib.dump(model, os.path.join(MODELS_DIR, 'svm.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'svm_scaler.pkl'))
    save_checkpoint('svm', model, metadata=metrics, scaler=scaler)
    print(f"  Train time: {train_time:.2f}s")

    return model, scaler, metrics


# ──────────────────────────────────────────────────────────────
# TASK 2.4: NAIVE BAYES (Question Type Classification)
# ──────────────────────────────────────────────────────────────

def train_naive_bayes_question_type(train_df, val_df, count_vec):
    """Train Naive Bayes for question type classification."""
    import re

    wh_words = ['what', 'which', 'who', 'where', 'when', 'why', 'how']

    def classify_q_type(question):
        q_lower = str(question).lower().strip()
        for wh in wh_words:
            if q_lower.startswith(wh):
                return wh
        if '_' in q_lower:
            return 'fill_blank'
        return 'other'

    print("\n[Training] Naive Bayes — Question Type Classification...")

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['q_type'] = train_df['question'].apply(classify_q_type)
    val_df['q_type'] = val_df['question'].apply(classify_q_type)

    # Vectorize questions
    X_train_q = count_vec.transform(train_df['question'].apply(clean_text).tolist())
    X_val_q = count_vec.transform(val_df['question'].apply(clean_text).tolist())

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_q = le.fit_transform(train_df['q_type'])
    y_val_q = le.transform(val_df['q_type'])

    model = MultinomialNB(alpha=1.0)
    start = time.time()
    model.fit(X_train_q, y_train_q)
    train_time = time.time() - start

    y_pred = model.predict(X_val_q)
    metrics = evaluate_model("Naive Bayes (Question Type)", y_val_q, y_pred)
    metrics['train_time'] = train_time

    print(f"  Classes: {le.classes_}")
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(model, os.path.join(MODELS_DIR, 'naive_bayes_qtype.pkl'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'nb_label_encoder.pkl'))
    save_checkpoint('naive_bayes_qtype', model, metadata=metrics)

    return model, le, metrics


# ──────────────────────────────────────────────────────────────
# TASK 2.5: RANDOM FOREST (Difficulty Estimation)
# ──────────────────────────────────────────────────────────────

def train_random_forest_difficulty(train_df, val_df, tfidf_vec):
    """Train Random Forest for difficulty estimation (middle vs high school)."""
    print("\n[Training] Random Forest — Difficulty Estimation...")

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df['difficulty'] = train_df['id'].apply(
        lambda x: 1 if str(x).startswith('high') else 0
    )
    val_df['difficulty'] = val_df['id'].apply(
        lambda x: 1 if str(x).startswith('high') else 0
    )

    # Features: article length, question length, option lengths, TF-IDF
    def build_diff_features(df):
        feats = pd.DataFrame()
        feats['article_len'] = df['article'].str.split().str.len()
        feats['question_len'] = df['question'].str.split().str.len()
        feats['avg_option_len'] = df[['A', 'B', 'C', 'D']].apply(
            lambda row: np.mean([len(str(x).split()) for x in row]), axis=1
        )
        feats['article_sentence_count'] = df['article'].apply(
            lambda x: len([s for s in str(x).split('.') if len(s.strip()) > 5])
        )
        return feats

    X_train_diff = build_diff_features(train_df)
    X_val_diff = build_diff_features(val_df)
    y_train_diff = train_df['difficulty'].values
    y_val_diff = val_df['difficulty'].values

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    start = time.time()
    model.fit(X_train_diff, y_train_diff)
    train_time = time.time() - start

    y_pred = model.predict(X_val_diff)
    metrics = evaluate_model("Random Forest (Difficulty)", y_val_diff, y_pred)
    metrics['train_time'] = train_time
    print(f"  Train time: {train_time:.2f}s")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_train_diff.columns)
    print(f"  Feature importances:\n{importances.sort_values(ascending=False).to_string()}")

    joblib.dump(model, os.path.join(MODELS_DIR, 'random_forest_difficulty.pkl'))
    save_checkpoint('random_forest_difficulty', model, metadata=metrics)

    return model, metrics


# ──────────────────────────────────────────────────────────────
# TASK 2.5b: XGBOOST (Answer Verification)
# ──────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost for answer verification."""
    print("\n[Training] XGBoost...")
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)
    metrics = evaluate_model("XGBoost", y_val, y_pred)
    metrics['train_time'] = train_time
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(model, os.path.join(MODELS_DIR, 'xgboost.pkl'))
    save_checkpoint('xgboost', model, metadata=metrics)

    return model, metrics


# ──────────────────────────────────────────────────────────────
# TASKS 2.7-2.10: UNSUPERVISED / SEMI-SUPERVISED
# ──────────────────────────────────────────────────────────────

def train_kmeans(X_train, y_train, n_clusters=2):
    """K-Means Clustering on verification features."""
    print("\n[Training] K-Means Clustering...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    start = time.time()
    model.fit(X_scaled)
    train_time = time.time() - start

    labels = model.labels_
    sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))

    # Clustering purity
    from collections import Counter
    purity = 0
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            most_common = Counter(y_train[cluster_mask]).most_common(1)[0][1]
            purity += most_common
    purity /= len(y_train)

    print(f"\n{'='*50}")
    print(f"  K-Means Clustering (k={n_clusters})")
    print(f"{'='*50}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Clustering Purity: {purity:.4f}")
    print(f"  Cluster sizes: {np.bincount(labels)}")
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(model, os.path.join(MODELS_DIR, 'kmeans.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'kmeans_scaler.pkl'))
    save_checkpoint('kmeans', model, metadata={'silhouette': sil_score}, scaler=scaler)

    return model, {'silhouette': sil_score, 'purity': purity, 'train_time': train_time}


def train_gmm(X_train, y_train, n_components=2):
    """Gaussian Mixture Model clustering."""
    print("\n[Training] Gaussian Mixture Model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full')
    start = time.time()
    model.fit(X_scaled)
    train_time = time.time() - start

    labels = model.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))

    from collections import Counter
    purity = 0
    for cluster_id in range(n_components):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            most_common = Counter(y_train[cluster_mask]).most_common(1)[0][1]
            purity += most_common
    purity /= len(y_train)

    print(f"\n{'='*50}")
    print(f"  GMM (n_components={n_components})")
    print(f"{'='*50}")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Clustering Purity: {purity:.4f}")
    print(f"  BIC: {model.bic(X_scaled):.2f}")
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(model, os.path.join(MODELS_DIR, 'gmm.pkl'))
    save_checkpoint('gmm', model, metadata={'silhouette': sil_score})

    return model, {'silhouette': sil_score, 'purity': purity, 'train_time': train_time}


def train_label_propagation(X_train, y_train, X_val, y_val, labeled_fraction=0.1):
    """Semi-supervised Label Propagation."""
    print(f"\n[Training] Label Propagation (labeled fraction={labeled_fraction})...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create semi-supervised labels: mask (1-labeled_fraction) as -1 (unlabeled)
    rng = np.random.RandomState(42)
    y_semi = y_train.copy()
    unlabeled_mask = rng.rand(len(y_semi)) > labeled_fraction
    y_semi[unlabeled_mask] = -1

    n_labeled = (y_semi != -1).sum()
    print(f"  Labeled: {n_labeled}, Unlabeled: {len(y_semi) - n_labeled}")

    # Subsample if dataset is large (Label Propagation is O(n^2))
    max_samples = 5000
    if len(X_train_scaled) > max_samples:
        idx = rng.choice(len(X_train_scaled), max_samples, replace=False)
        X_sub = X_train_scaled[idx]
        y_sub = y_semi[idx]
    else:
        X_sub = X_train_scaled
        y_sub = y_semi

    model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=100)
    start = time.time()
    model.fit(X_sub, y_sub)
    train_time = time.time() - start

    y_pred = model.predict(X_val_scaled)
    metrics = evaluate_model("Label Propagation (Semi-Supervised)", y_val, y_pred)
    metrics['train_time'] = train_time
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(model, os.path.join(MODELS_DIR, 'label_propagation.pkl'))
    save_checkpoint('label_propagation', model, metadata=metrics)

    return model, metrics


# ──────────────────────────────────────────────────────────────
# TASKS 2.11-2.13: TEMPLATE-BASED QUESTION GENERATION
# ──────────────────────────────────────────────────────────────

def generate_questions_from_article(article, correct_answer, tfidf_vec=None):
    """
    Template-based question generation:
      Step 1: Extract candidate sentences using keyword overlap
      Step 2: Apply Wh-word templates
      Step 3: Return ranked questions
    """
    import re

    sentences = re.split(r'[.!?\n]+', str(article))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return []

    answer_words = set(clean_text(correct_answer).split())

    # Step 1: Score sentences by keyword overlap with the correct answer
    scored_sents = []
    for sent in sentences:
        sent_words = set(clean_text(sent).split())
        overlap = len(answer_words & sent_words) / max(len(answer_words), 1)
        scored_sents.append((sent, overlap))

    scored_sents.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sents[:5]]

    # Step 2: Apply Wh-word templates
    templates = [
        ("What", "What is described in the following: {}?"),
        ("Who", "Who is mentioned in: {}?"),
        ("Where", "Where does the event in '{}' take place?"),
        ("When", "When does the event described in '{}' occur?"),
        ("Why", "Why does the situation in '{}' happen?"),
        ("Which", "Which of the following is true about: {}?"),
    ]

    questions = []
    for sent in top_sentences:
        sent_short = sent[:100] + "..." if len(sent) > 100 else sent
        for wh_type, template in templates:
            q = template.format(sent_short)
            questions.append({'question': q, 'type': wh_type, 'source_sentence': sent})

    return questions[:10]  # Return top 10


# ──────────────────────────────────────────────────────────────
# TASKS 2.14-2.15: ENSEMBLE
# ──────────────────────────────────────────────────────────────

def train_ensemble(X_train, y_train, X_val, y_val):
    """Train ensemble model (soft voting) combining LR, SVM, RF, XGBoost."""
    print("\n[Training] Ensemble (Soft Voting)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               random_state=42, eval_metric='logloss', n_jobs=-1)),
    ]

    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    start = time.time()
    ensemble.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = ensemble.predict(X_val_scaled)
    metrics = evaluate_model("Ensemble (Soft Voting)", y_val, y_pred)
    metrics['train_time'] = train_time
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble_voting.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'ensemble_scaler.pkl'))
    save_checkpoint('ensemble_voting', ensemble, metadata=metrics, scaler=scaler)

    return ensemble, scaler, metrics


def train_stacking_ensemble(X_train, y_train, X_val, y_val):
    """Train stacking ensemble."""
    print("\n[Training] Stacking Ensemble...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        n_jobs=-1
    )
    start = time.time()
    stacking.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = stacking.predict(X_val_scaled)
    metrics = evaluate_model("Stacking Ensemble", y_val, y_pred)
    metrics['train_time'] = train_time
    print(f"  Train time: {train_time:.2f}s")

    joblib.dump(stacking, os.path.join(MODELS_DIR, 'stacking_ensemble.pkl'))
    save_checkpoint('stacking_ensemble', stacking, metadata=metrics)

    return stacking, metrics


# ──────────────────────────────────────────────────────────────
# MAIN — Run all Model A training
# ──────────────────────────────────────────────────────────────

def run_all_model_a():
    """Run all Model A training tasks."""
    print("=" * 60)
    print("  MODEL A — Training Pipeline")
    print("=" * 60)

    # Load features
    print("\nLoading preprocessed features...")
    data = load_features()
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Train label dist: {np.bincount(y_train)}")

    # Load raw data for NB and RF tasks
    train_df, val_df, test_df = load_raw_data()
    tfidf_vec = data['tfidf_vec']
    count_vec = data['count_vec']

    results = []

    # ── Supervised Models ──
    _, _, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    results.append(lr_metrics)

    _, _, svm_metrics = train_svm(X_train, y_train, X_val, y_val)
    results.append(svm_metrics)

    _, _, nb_metrics = train_naive_bayes_question_type(train_df, val_df, count_vec)
    results.append(nb_metrics)

    _, rf_metrics = train_random_forest_difficulty(train_df, val_df, tfidf_vec)
    results.append(rf_metrics)

    _, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    results.append(xgb_metrics)

    # ── Unsupervised / Semi-Supervised ──
    _, kmeans_metrics = train_kmeans(X_train, y_train, n_clusters=2)
    _, gmm_metrics = train_gmm(X_train, y_train, n_components=2)
    _, lp_metrics = train_label_propagation(X_train, y_train, X_val, y_val)
    results.append(lp_metrics)

    # ── Ensemble ──
    _, _, ens_metrics = train_ensemble(X_train, y_train, X_val, y_val)
    results.append(ens_metrics)

    _, stack_metrics = train_stacking_ensemble(X_train, y_train, X_val, y_val)
    results.append(stack_metrics)

    # ── Final Test Set Evaluation ──
    print("\n" + "=" * 60)
    print("  FINAL TEST SET RESULTS")
    print("=" * 60)

    # Load best models and evaluate on test set
    scaler = joblib.load(os.path.join(MODELS_DIR, 'lr_scaler.pkl'))
    X_test_scaled = scaler.transform(X_test)

    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    evaluate_model("Logistic Regression (TEST)", y_test, lr_model.predict(X_test_scaled))

    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm.pkl'))
    svm_scaler = joblib.load(os.path.join(MODELS_DIR, 'svm_scaler.pkl'))
    evaluate_model("SVM (TEST)", y_test, svm_model.predict(svm_scaler.transform(X_test)))

    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    evaluate_model("XGBoost (TEST)", y_test, xgb_model.predict(X_test))

    ens_model = joblib.load(os.path.join(MODELS_DIR, 'ensemble_voting.pkl'))
    ens_scaler = joblib.load(os.path.join(MODELS_DIR, 'ensemble_scaler.pkl'))
    evaluate_model("Ensemble (TEST)", y_test, ens_model.predict(ens_scaler.transform(X_test)))

    # ── Comparison Table ──
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    comparison = pd.DataFrame([r for r in results if 'accuracy' in r])
    comparison = comparison.sort_values('f1', ascending=False)
    print(comparison.to_string(index=False))

    # Save comparison
    comparison.to_csv(os.path.join(DATA_PROC_DIR, 'model_a_comparison.csv'), index=False)
    print("\nComparison saved to data/processed/model_a_comparison.csv")

    # Question generation demo
    print("\n" + "=" * 60)
    print("  QUESTION GENERATION DEMO")
    print("=" * 60)
    sample = train_df.iloc[0]
    correct_text = sample[sample['answer']]
    questions = generate_questions_from_article(sample['article'], correct_text, tfidf_vec)
    for i, q in enumerate(questions[:5]):
        print(f"\n  Q{i+1} [{q['type']}]: {q['question']}")


if __name__ == '__main__':
    run_all_model_a()
