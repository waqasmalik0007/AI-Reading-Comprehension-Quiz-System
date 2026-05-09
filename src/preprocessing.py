"""
preprocessing.py — Dataset loading, cleaning, and feature engineering for RACE dataset.

This module handles:
  - Loading raw CSV data (train/val/test)
  - Text cleaning (lowercasing, punctuation removal)
  - One-Hot Encoding and TF-IDF vectorization
  - Cosine similarity feature computation
  - Feature matrix construction for Model A and Model B
  - Saving/loading processed data
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_raw_data():
    """Load train, validation, and test CSVs from data/raw/."""
    train_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'test.csv'))
    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────
# 2. TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_text(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_dataframe(df):
    """Apply text cleaning to all relevant columns in-place and return the df."""
    df = df.copy()
    for col in ['article', 'question', 'A', 'B', 'C', 'D']:
        df[col + '_clean'] = df[col].apply(clean_text)
    return df


# ──────────────────────────────────────────────────────────────
# 3. LABEL ENCODING
# ──────────────────────────────────────────────────────────────

def encode_answer_labels(df):
    """Convert answer column (A/B/C/D) to numeric (0/1/2/3)."""
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df = df.copy()
    df['answer_label'] = df['answer'].map(label_map)
    return df


def get_correct_option_text(row):
    """Return the text of the correct answer option."""
    return str(row.get(row['answer'], ''))


# ──────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING — TF-IDF
# ──────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(corpus, max_features=10000, ngram_range=(1, 2)):
    """Fit a TF-IDF vectorizer on the given corpus."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        sublinear_tf=True,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def build_count_vectorizer(corpus, max_features=10000):
    """Fit a CountVectorizer (One-Hot / Bag-of-Words) on the given corpus."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        binary=True,  # One-Hot: 1 if present, 0 if absent
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


# ──────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING — VERIFICATION FEATURES (MODEL A)
# ──────────────────────────────────────────────────────────────

def build_verification_features(df, tfidf_vectorizer=None, count_vectorizer=None):
    """
    Build feature vectors for the answer verification task (Model A).

    For each row, creates 4 samples (one per option A/B/C/D).
    Features per sample:
      - TF-IDF cosine similarity: article vs option, question vs option, article vs question
      - One-Hot cosine similarity: article vs option, question vs option
      - Handcrafted: option word count, question word count, word overlap ratio
    Label: 1 if this option is the correct answer, 0 otherwise.
    """
    features_list = []
    labels = []

    for idx, row in df.iterrows():
        article_clean = row.get('article_clean', clean_text(row['article']))
        question_clean = row.get('question_clean', clean_text(row['question']))
        correct_answer = row['answer']

        for opt_label in ['A', 'B', 'C', 'D']:
            option_clean = row.get(opt_label + '_clean', clean_text(row[opt_label]))

            feat = {}

            # --- TF-IDF cosine similarities ---
            if tfidf_vectorizer is not None:
                art_vec = tfidf_vectorizer.transform([article_clean])
                q_vec = tfidf_vectorizer.transform([question_clean])
                opt_vec = tfidf_vectorizer.transform([option_clean])

                feat['tfidf_cos_art_opt'] = cosine_similarity(art_vec, opt_vec)[0, 0]
                feat['tfidf_cos_q_opt'] = cosine_similarity(q_vec, opt_vec)[0, 0]
                feat['tfidf_cos_art_q'] = cosine_similarity(art_vec, q_vec)[0, 0]

            # --- One-Hot cosine similarities ---
            if count_vectorizer is not None:
                art_oh = count_vectorizer.transform([article_clean])
                q_oh = count_vectorizer.transform([question_clean])
                opt_oh = count_vectorizer.transform([option_clean])

                feat['oh_cos_art_opt'] = cosine_similarity(art_oh, opt_oh)[0, 0]
                feat['oh_cos_q_opt'] = cosine_similarity(q_oh, opt_oh)[0, 0]

            # --- Handcrafted features ---
            art_words = set(article_clean.split())
            q_words = set(question_clean.split())
            opt_words = set(option_clean.split())

            feat['option_word_count'] = len(option_clean.split())
            feat['question_word_count'] = len(question_clean.split())
            feat['article_word_count'] = len(article_clean.split())

            # Word overlap ratios
            if len(opt_words) > 0:
                feat['overlap_art_opt'] = len(art_words & opt_words) / len(opt_words)
                feat['overlap_q_opt'] = len(q_words & opt_words) / len(opt_words)
            else:
                feat['overlap_art_opt'] = 0.0
                feat['overlap_q_opt'] = 0.0

            feat['overlap_art_q'] = (
                len(art_words & q_words) / len(q_words) if len(q_words) > 0 else 0.0
            )

            features_list.append(feat)
            labels.append(1 if opt_label == correct_answer else 0)

    X = pd.DataFrame(features_list)
    y = np.array(labels)
    return X, y


# ──────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING — DISTRACTOR FEATURES (MODEL B)
# ──────────────────────────────────────────────────────────────

def extract_candidate_phrases(article, correct_answer, top_n=20):
    """
    Extract candidate distractor phrases from the article.
    Uses frequency-based word selection and simple string matching.
    """
    sentences = re.split(r'[.!?\n]+', str(article))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Get important words from article (frequency-based)
    words = re.findall(r'\b[a-z]+\b', article.lower())
    stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'been', 'be',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'shall', 'can', 'need',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'as', 'into', 'through', 'during', 'before', 'after', 'and',
                  'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                  'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                  'than', 'too', 'very', 'just', 'because', 'if', 'when',
                  'where', 'how', 'what', 'which', 'who', 'whom', 'this',
                  'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
                  'them', 'his', 'her', 'their', 'my', 'your', 'our', 'i',
                  'me', 'we', 'you', 'him', 'us', 'about', 'up', 'out', 'one',
                  'also', 'then', 'there', 'here'}

    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    word_freq = pd.Series(content_words).value_counts()

    # Get candidate phrases (short segments from sentences)
    candidates = []
    correct_lower = clean_text(correct_answer)

    for sent in sentences:
        sent_clean = clean_text(sent)
        if correct_lower in sent_clean:
            continue  # Skip sentences containing the correct answer

        # Extract noun-phrase-like segments (sequences of content words)
        sent_words = sent_clean.split()
        for n in [1, 2, 3]:  # unigrams, bigrams, trigrams
            for i in range(len(sent_words) - n + 1):
                phrase = ' '.join(sent_words[i:i + n])
                if phrase != correct_lower and len(phrase) > 2:
                    candidates.append(phrase)

    # Deduplicate and rank by frequency
    candidate_freq = pd.Series(candidates).value_counts()
    return candidate_freq.head(top_n).index.tolist()


def compute_distractor_features(candidate, correct_answer, article,
                                 tfidf_vectorizer=None, count_vectorizer=None):
    """Compute features for a single distractor candidate."""
    feat = {}
    cand_clean = clean_text(candidate)
    ans_clean = clean_text(correct_answer)
    art_clean = clean_text(article)

    # Character-level match score
    common_chars = set(cand_clean) & set(ans_clean)
    feat['char_match'] = len(common_chars) / max(len(set(ans_clean)), 1)

    # Length ratio
    feat['len_ratio'] = len(cand_clean.split()) / max(len(ans_clean.split()), 1)

    # Passage frequency
    feat['passage_freq'] = art_clean.count(cand_clean)

    # Word overlap with correct answer
    cand_words = set(cand_clean.split())
    ans_words = set(ans_clean.split())
    feat['word_overlap'] = len(cand_words & ans_words) / max(len(ans_words), 1)

    # TF-IDF cosine similarity to correct answer
    if tfidf_vectorizer is not None:
        cand_vec = tfidf_vectorizer.transform([cand_clean])
        ans_vec = tfidf_vectorizer.transform([ans_clean])
        feat['tfidf_cos_cand_ans'] = cosine_similarity(cand_vec, ans_vec)[0, 0]

    # One-Hot cosine similarity
    if count_vectorizer is not None:
        cand_oh = count_vectorizer.transform([cand_clean])
        ans_oh = count_vectorizer.transform([ans_clean])
        feat['oh_cos_cand_ans'] = cosine_similarity(cand_oh, ans_oh)[0, 0]

    return feat


# ──────────────────────────────────────────────────────────────
# 7. HINT GENERATION FEATURES (MODEL B)
# ──────────────────────────────────────────────────────────────

def score_sentences_for_hints(article, question, tfidf_vectorizer=None):
    """
    Score each sentence in the article for relevance to the question.
    Returns list of (sentence, score) sorted by relevance descending.
    """
    sentences = re.split(r'[.!?\n]+', str(article))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return []

    question_clean = clean_text(question)

    scored = []
    for sent in sentences:
        sent_clean = clean_text(sent)

        # Keyword overlap score
        q_words = set(question_clean.split())
        s_words = set(sent_clean.split())
        keyword_overlap = len(q_words & s_words) / max(len(q_words), 1)

        # Position score (earlier sentences often more relevant)
        position = sentences.index(sent) / max(len(sentences), 1)
        position_score = 1.0 - position

        # Length score (medium-length sentences preferred)
        length_score = min(len(sent_clean.split()) / 20.0, 1.0)

        # TF-IDF cosine similarity
        tfidf_score = 0.0
        if tfidf_vectorizer is not None:
            q_vec = tfidf_vectorizer.transform([question_clean])
            s_vec = tfidf_vectorizer.transform([sent_clean])
            tfidf_score = cosine_similarity(q_vec, s_vec)[0, 0]

        # Combined score
        score = (0.4 * keyword_overlap + 0.2 * position_score +
                 0.1 * length_score + 0.3 * tfidf_score)

        scored.append((sent.strip(), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def generate_graduated_hints(article, question, tfidf_vectorizer=None, n_hints=3,
                             correct_answer=None):
    """
    Generate graduated hints (general → specific → near-explicit).

    Hint 1 — Contextual clue   : relevant sentence from article, answer word absent
    Hint 2 — Closer clue       : second most relevant sentence, answer word absent
    Hint 3 — Near-explicit     : first letter + character/word count (never the answer)
    """
    ans = (correct_answer or '').strip()
    ans_lower = ans.lower()

    # Score sentences by relevance to the question
    scored = score_sentences_for_hints(article, question, tfidf_vectorizer)

    # Keep only sentences that do NOT contain the answer word
    safe = [(s, sc) for s, sc in scored if ans_lower not in s.lower()] if ans else scored
    if not safe:
        safe = scored  # fallback: every sentence has the answer, use best available

    hints = []

    # ── Hint 1: Best relevant sentence without the answer ────────
    if safe:
        hints.append(safe[0][0])
    else:
        hints.append("Re-read the passage and focus on the surrounding context.")

    # ── Hint 2: Second best relevant sentence (different from H1) ─
    if len(safe) > 1:
        hints.append(safe[1][0])
    elif safe:
        hints.append(safe[0][0])
    else:
        hints.append("Look carefully at every sentence — the clue is nearby.")

    # ── Hint 3: Near-explicit — first letter + length ────────────
    if ans:
        first = ans[0].upper()
        nwords = len(ans.split())
        nchars = len(ans)
        if nwords > 1:
            hint3 = (f"The answer is {nwords} words long and starts with "
                     f"the letter '{first}'.")
        else:
            hint3 = (f"The answer has {nchars} character(s) and starts with "
                     f"'{first}'.")
        hints.append(hint3)
    elif len(safe) > 2:
        hints.append(safe[2][0])

    return hints[:n_hints]


# ──────────────────────────────────────────────────────────────
# 8. FULL PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────

def run_full_preprocessing(max_train_rows=None):
    """
    Run the complete preprocessing pipeline:
      1. Load raw data
      2. Clean text
      3. Encode labels
      4. Fit TF-IDF and One-Hot vectorizers on training articles
      5. Build verification features for Model A
      6. Save everything to data/processed/
    
    Args:
        max_train_rows: Limit training rows for faster iteration (None = all)
    """
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    print("Loading raw data...")
    train_df, val_df, test_df = load_raw_data()

    if max_train_rows:
        train_df = train_df.head(max_train_rows)
        print(f"  Using first {max_train_rows} training rows")

    print(f"  Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    print("Cleaning text...")
    train_df = clean_dataframe(train_df)
    val_df = clean_dataframe(val_df)
    test_df = clean_dataframe(test_df)

    print("Encoding labels...")
    train_df = encode_answer_labels(train_df)
    val_df = encode_answer_labels(val_df)
    test_df = encode_answer_labels(test_df)

    print("Fitting TF-IDF vectorizer on training articles...")
    tfidf_vec, X_train_tfidf = build_tfidf_vectorizer(
        train_df['article_clean'].tolist(), max_features=10000
    )

    print("Fitting One-Hot (CountVectorizer) on training articles...")
    count_vec, X_train_oh = build_count_vectorizer(
        train_df['article_clean'].tolist(), max_features=10000
    )

    print("Building verification features (Model A) — this may take a while...")
    print("  Processing training set...")
    X_train_feat, y_train = build_verification_features(train_df, tfidf_vec, count_vec)
    print(f"  Train features: {X_train_feat.shape}, labels: {y_train.shape}")

    print("  Processing validation set...")
    X_val_feat, y_val = build_verification_features(val_df, tfidf_vec, count_vec)
    print(f"  Val features: {X_val_feat.shape}, labels: {y_val.shape}")

    print("  Processing test set...")
    X_test_feat, y_test = build_verification_features(test_df, tfidf_vec, count_vec)
    print(f"  Test features: {X_test_feat.shape}, labels: {y_test.shape}")

    # Save everything
    print("Saving processed data...")
    train_df.to_csv(os.path.join(DATA_PROC_DIR, 'train_clean.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_PROC_DIR, 'val_clean.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_PROC_DIR, 'test_clean.csv'), index=False)

    joblib.dump(tfidf_vec, os.path.join(DATA_PROC_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(count_vec, os.path.join(DATA_PROC_DIR, 'count_vectorizer.pkl'))

    joblib.dump((X_train_feat, y_train), os.path.join(DATA_PROC_DIR, 'train_features.pkl'))
    joblib.dump((X_val_feat, y_val), os.path.join(DATA_PROC_DIR, 'val_features.pkl'))
    joblib.dump((X_test_feat, y_test), os.path.join(DATA_PROC_DIR, 'test_features.pkl'))

    print("Done! All files saved to data/processed/")
    return {
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'tfidf_vec': tfidf_vec, 'count_vec': count_vec,
        'X_train': X_train_feat, 'y_train': y_train,
        'X_val': X_val_feat, 'y_val': y_val,
        'X_test': X_test_feat, 'y_test': y_test,
    }


def load_processed_data():
    """Load previously saved processed data."""
    tfidf_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'tfidf_vectorizer.pkl'))
    count_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'count_vectorizer.pkl'))
    X_train, y_train = joblib.load(os.path.join(DATA_PROC_DIR, 'train_features.pkl'))
    X_val, y_val = joblib.load(os.path.join(DATA_PROC_DIR, 'val_features.pkl'))
    X_test, y_test = joblib.load(os.path.join(DATA_PROC_DIR, 'test_features.pkl'))
    return {
        'tfidf_vec': tfidf_vec, 'count_vec': count_vec,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run with a subset first for speed; set to None for full dataset
    result = run_full_preprocessing(max_train_rows=5000)
    print(f"\nFeature columns: {list(result['X_train'].columns)}")
    print(f"Label distribution (train): {np.bincount(result['y_train'])}")
