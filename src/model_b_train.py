"""
model_b_train.py — Training script for Model B: Distractor & Hint Generator

This script trains and evaluates:
  1. Distractor candidate extraction (frequency-based)
  2. Distractor feature engineering & ML ranking
  3. Word2Vec nearest-neighbour distractors
  4. Extractive hint generation with ML scoring
  5. Graduated hint generation
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import (
    load_raw_data, clean_text, extract_candidate_phrases,
    compute_distractor_features, score_sentences_for_hints,
    generate_graduated_hints
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_b', 'traditional')
DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_model(name, y_true, y_pred):
    """Print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
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

    return {'name': name, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}


# ──────────────────────────────────────────────────────────────
# TASK 3.1-3.3: DISTRACTOR GENERATION WITH ML RANKING
# ──────────────────────────────────────────────────────────────

def build_distractor_training_data(train_df, tfidf_vec, count_vec, n_samples=2000):
    """
    Build training data for distractor ranker.
    
    For each sample:
      - Extract candidate phrases from article
      - Compute features for each candidate
      - Label: 1 if candidate matches a real distractor, 0 otherwise
    """
    print(f"Building distractor training data from {n_samples} samples...")
    features_list = []
    labels = []

    sample_df = train_df.sample(min(n_samples, len(train_df)), random_state=42)

    for idx, row in sample_df.iterrows():
        article = str(row['article'])
        correct_answer = str(row[row['answer']])
        correct_label = row['answer']

        # Get the real distractors (the 3 wrong options)
        real_distractors = set()
        for opt in ['A', 'B', 'C', 'D']:
            if opt != correct_label:
                real_distractors.add(clean_text(str(row[opt])))

        # Extract candidates
        candidates = extract_candidate_phrases(article, correct_answer, top_n=15)

        for cand in candidates:
            feat = compute_distractor_features(
                cand, correct_answer, article, tfidf_vec, count_vec
            )

            # Label: is this candidate close to a real distractor?
            cand_clean = clean_text(cand)
            is_distractor = 0
            for rd in real_distractors:
                # Check if candidate overlaps significantly with real distractor
                cand_words = set(cand_clean.split())
                rd_words = set(rd.split())
                if len(cand_words & rd_words) / max(len(rd_words), 1) > 0.5:
                    is_distractor = 1
                    break

            features_list.append(feat)
            labels.append(is_distractor)

    X = pd.DataFrame(features_list)
    y = np.array(labels)
    print(f"  Built {len(X)} samples, label dist: {np.bincount(y)}")
    return X, y


def train_distractor_ranker(X_train, y_train):
    """Train a distractor ranking model."""
    print("\n[Training] Distractor Ranker (Random Forest)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        n_jobs=-1, class_weight='balanced'
    )
    start = time.time()
    model.fit(X_scaled, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_scaled)
    metrics = evaluate_model("Distractor Ranker (Train)", y_train, y_pred)
    print(f"  Train time: {train_time:.2f}s")

    # Feature importance
    feat_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]
    importances = pd.Series(model.feature_importances_, index=feat_names)
    print(f"  Feature importances:\n{importances.sort_values(ascending=False).to_string()}")

    joblib.dump(model, os.path.join(MODELS_DIR, 'distractor_ranker.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'distractor_scaler.pkl'))

    return model, scaler, metrics


def generate_distractors(article, question, correct_answer, tfidf_vec, count_vec,
                         ranker_model=None, ranker_scaler=None, top_k=3):
    """
    Generate top-K distractors for a given article/question/answer.
    Uses ML ranker if available, otherwise falls back to cosine similarity ranking.
    """
    candidates = extract_candidate_phrases(article, correct_answer, top_n=30)

    if not candidates:
        return ["Option A", "Option B", "Option C"]

    scored = []
    for cand in candidates:
        feat = compute_distractor_features(cand, correct_answer, article, tfidf_vec, count_vec)

        if ranker_model is not None and ranker_scaler is not None:
            feat_arr = np.array(list(feat.values())).reshape(1, -1)
            feat_scaled = ranker_scaler.transform(feat_arr)
            try:
                score = ranker_model.predict_proba(feat_scaled)[0][1]
            except:
                score = ranker_model.predict(feat_scaled)[0]
        else:
            # Fallback: use cosine similarity as score
            score = feat.get('tfidf_cos_cand_ans', feat.get('oh_cos_cand_ans', 0))

        scored.append((cand, score))

    # Sort by score descending, apply diversity penalty
    scored.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for cand, score in scored:
        if len(selected) >= top_k:
            break
        # Diversity: ensure new distractor is different from already selected
        is_diverse = True
        for prev in selected:
            prev_words = set(prev.split())
            cand_words = set(cand.split())
            overlap = len(prev_words & cand_words) / max(len(cand_words), 1)
            if overlap > 0.7:
                is_diverse = False
                break
        if is_diverse:
            selected.append(cand)

    # Pad if we don't have enough
    while len(selected) < top_k:
        selected.append(f"Option {len(selected) + 1}")

    return selected


# ──────────────────────────────────────────────────────────────
# TASK 3.4: WORD2VEC NEAREST NEIGHBOURS (OPTIONAL)
# ──────────────────────────────────────────────────────────────

def train_word2vec_distractors(train_df, n_samples=500):
    """Use pre-trained Word2Vec for distractor generation."""
    print("\n[Training] Word2Vec Nearest Neighbour Distractors...")

    try:
        import gensim.downloader as api
        print("  Loading Word2Vec model (this may take a while on first run)...")
        w2v_model = api.load('word2vec-google-news-300')
        print("  Word2Vec loaded successfully.")
    except Exception as e:
        print(f"  Warning: Could not load Word2Vec: {e}")
        print("  Skipping Word2Vec distractors.")
        return None, {}

    sample_df = train_df.sample(min(n_samples, len(train_df)), random_state=42)
    results = []

    for idx, row in sample_df.iterrows():
        correct_answer = str(row[row['answer']])
        answer_words = clean_text(correct_answer).split()

        # Get nearest neighbours for answer words
        neighbors = set()
        for word in answer_words:
            try:
                similar = w2v_model.most_similar(word, topn=10)
                neighbors.update([w for w, s in similar])
            except KeyError:
                continue

        # Filter: remove words that appear in the article
        article_words = set(clean_text(row['article']).split())
        distractors = [w for w in neighbors if w.lower() not in article_words][:3]

        results.append({
            'correct_answer': correct_answer,
            'w2v_distractors': distractors
        })

    print(f"  Generated distractors for {len(results)} samples")
    if results:
        for r in results[:3]:
            print(f"  Answer: {r['correct_answer'][:50]} -> Distractors: {r['w2v_distractors'][:3]}")

    return w2v_model, {'n_samples': len(results)}


# ──────────────────────────────────────────────────────────────
# TASKS 3.7-3.10: HINT GENERATION
# ──────────────────────────────────────────────────────────────

def build_hint_training_data(train_df, tfidf_vec, n_samples=2000):
    """Build training data for hint sentence scoring."""
    print(f"Building hint training data from {n_samples} samples...")

    features_list = []
    relevance_scores = []

    sample_df = train_df.sample(min(n_samples, len(train_df)), random_state=42)

    for idx, row in sample_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])

        sentences = re.split(r'[.!?\n]+', article)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        q_clean = clean_text(question)
        ans_clean = clean_text(correct_answer)
        q_words = set(q_clean.split())
        ans_words = set(ans_clean.split())

        for i, sent in enumerate(sentences):
            sent_clean = clean_text(sent)
            sent_words = set(sent_clean.split())

            # Features
            feat = {}
            feat['keyword_overlap_q'] = len(q_words & sent_words) / max(len(q_words), 1)
            feat['keyword_overlap_ans'] = len(ans_words & sent_words) / max(len(ans_words), 1)
            feat['position'] = i / max(len(sentences), 1)
            feat['sent_length'] = len(sent_clean.split())
            feat['sent_length_norm'] = min(feat['sent_length'] / 20.0, 1.0)

            # TF-IDF cosine similarity
            if tfidf_vec is not None:
                q_vec = tfidf_vec.transform([q_clean])
                s_vec = tfidf_vec.transform([sent_clean])
                feat['tfidf_cos_q_sent'] = cosine_similarity(q_vec, s_vec)[0, 0]

                a_vec = tfidf_vec.transform([ans_clean])
                feat['tfidf_cos_ans_sent'] = cosine_similarity(a_vec, s_vec)[0, 0]
            else:
                feat['tfidf_cos_q_sent'] = 0.0
                feat['tfidf_cos_ans_sent'] = 0.0

            features_list.append(feat)

            # Relevance label: how useful is this sentence as a hint?
            # Higher if it overlaps with question AND is close to answer
            relevance = (
                0.4 * feat['keyword_overlap_q'] +
                0.4 * feat['keyword_overlap_ans'] +
                0.2 * feat['tfidf_cos_q_sent']
            )
            relevance_scores.append(relevance)

    X = pd.DataFrame(features_list)
    y = np.array(relevance_scores)
    print(f"  Built {len(X)} sentence samples")
    return X, y


def train_hint_scorer(X_train, y_train):
    """Train a Logistic Regression model to score hint sentences."""
    print("\n[Training] Hint Sentence Scorer (Logistic Regression)...")

    # Binarize: top 30% sentences are "good hints"
    threshold = np.percentile(y_train, 70)
    y_binary = (y_train >= threshold).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    start = time.time()
    model.fit(X_scaled, y_binary)
    train_time = time.time() - start

    y_pred = model.predict(X_scaled)
    metrics = evaluate_model("Hint Scorer (Train)", y_binary, y_pred)
    print(f"  Train time: {train_time:.2f}s")

    # R² score on continuous relevance
    y_prob = model.predict_proba(X_scaled)[:, 1]
    r2 = r2_score(y_train, y_prob)
    print(f"  R² Score (continuous): {r2:.4f}")

    joblib.dump(model, os.path.join(MODELS_DIR, 'hint_scorer.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'hint_scaler.pkl'))

    return model, scaler, metrics


def evaluate_distractors(test_df, tfidf_vec, count_vec, ranker_model, ranker_scaler, n_samples=200):
    """Evaluate distractor quality on test set."""
    print(f"\n[Evaluating] Distractor quality on {n_samples} test samples...")

    sample_df = test_df.sample(min(n_samples, len(test_df)), random_state=42)

    correct_not_in_distractors = 0
    total = 0

    for idx, row in sample_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])

        distractors = generate_distractors(
            article, question, correct_answer,
            tfidf_vec, count_vec, ranker_model, ranker_scaler
        )

        # Check: correct answer should NOT be among distractors
        correct_clean = clean_text(correct_answer)
        distractor_match = False
        for d in distractors:
            if clean_text(d) == correct_clean:
                distractor_match = True
                break

        if not distractor_match:
            correct_not_in_distractors += 1
        total += 1

    accuracy = correct_not_in_distractors / max(total, 1)
    print(f"  Distractor Accuracy (correct answer NOT in distractors): {accuracy:.4f}")
    print(f"  Evaluated {total} samples")

    return {'distractor_accuracy': accuracy}


def evaluate_hints(test_df, tfidf_vec, n_samples=100):
    """Evaluate hint quality on test set."""
    print(f"\n[Evaluating] Hint quality on {n_samples} test samples...")

    sample_df = test_df.sample(min(n_samples, len(test_df)), random_state=42)

    hint_contains_answer_word = 0
    total_hints = 0

    for idx, row in sample_df.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        correct_answer = str(row[row['answer']])

        hints = generate_graduated_hints(article, question, tfidf_vec, n_hints=3)
        ans_words = set(clean_text(correct_answer).split())

        for hint in hints:
            hint_words = set(clean_text(hint).split())
            if len(ans_words & hint_words) > 0:
                hint_contains_answer_word += 1
            total_hints += 1

    precision_at_k = hint_contains_answer_word / max(total_hints, 1)
    print(f"  Hint Precision@K (hint contains answer keyword): {precision_at_k:.4f}")
    print(f"  Total hints evaluated: {total_hints}")

    return {'hint_precision_at_k': precision_at_k}


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def run_all_model_b():
    """Run all Model B training tasks."""
    print("=" * 60)
    print("  MODEL B — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_raw_data()
    tfidf_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'tfidf_vectorizer.pkl'))
    count_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'count_vectorizer.pkl'))

    # ── Distractor ranker ──
    X_dist, y_dist = build_distractor_training_data(train_df, tfidf_vec, count_vec, n_samples=1000)
    ranker, ranker_scaler, ranker_metrics = train_distractor_ranker(X_dist, y_dist)

    # ── Word2Vec distractors (optional) ──
    # Uncomment below to use Word2Vec (requires ~1.5GB download on first run)
    # w2v_model, w2v_metrics = train_word2vec_distractors(train_df, n_samples=200)

    # ── Hint scorer ──
    X_hint, y_hint = build_hint_training_data(train_df, tfidf_vec, n_samples=1000)
    hint_model, hint_scaler, hint_metrics = train_hint_scorer(X_hint, y_hint)

    # ── Evaluation ──
    dist_eval = evaluate_distractors(test_df, tfidf_vec, count_vec, ranker, ranker_scaler, n_samples=200)
    hint_eval = evaluate_hints(test_df, tfidf_vec, n_samples=100)

    # ── Demo ──
    print("\n" + "=" * 60)
    print("  DEMO: Distractor & Hint Generation")
    print("=" * 60)

    sample = test_df.iloc[0]
    article = sample['article']
    question = sample['question']
    correct_answer = sample[sample['answer']]

    print(f"\nArticle (first 200 chars): {article[:200]}...")
    print(f"Question: {question}")
    print(f"Correct Answer ({sample['answer']}): {correct_answer}")

    distractors = generate_distractors(
        article, question, correct_answer,
        tfidf_vec, count_vec, ranker, ranker_scaler
    )
    print(f"\nGenerated Distractors:")
    for i, d in enumerate(distractors):
        print(f"  {i+1}. {d}")

    hints = generate_graduated_hints(article, question, tfidf_vec, n_hints=3)
    print(f"\nGraduated Hints:")
    for i, h in enumerate(hints):
        level = ['General', 'Specific', 'Near-explicit'][i] if i < 3 else f'Hint {i+1}'
        print(f"  Hint {i+1} ({level}): {h[:100]}...")

    # ── Results summary ──
    print("\n" + "=" * 60)
    print("  MODEL B — Results Summary")
    print("=" * 60)
    print(f"  Distractor Ranker F1: {ranker_metrics.get('f1', 'N/A')}")
    print(f"  Distractor Accuracy: {dist_eval.get('distractor_accuracy', 'N/A')}")
    print(f"  Hint Precision@K: {hint_eval.get('hint_precision_at_k', 'N/A')}")


if __name__ == '__main__':
    run_all_model_b()
