"""
inference.py — Unified inference API for Model A and Model B.

Loads trained models and provides functions for:
  - Answer verification (Model A)
  - Question generation (Model A)
  - Distractor generation (Model B)
  - Hint generation (Model B)
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import (
    clean_text, extract_candidate_phrases, compute_distractor_features,
    generate_graduated_hints, score_sentences_for_hints
)
from src.model_b_train import generate_distractors
from src.model_a_train import generate_questions_from_article

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_A_DIR = os.path.join(BASE_DIR, 'models', 'model_a', 'traditional')
MODEL_B_DIR = os.path.join(BASE_DIR, 'models', 'model_b', 'traditional')
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')


class InferenceEngine:
    """Unified inference engine for the Reading Comprehension system."""

    def __init__(self):
        self.models_loaded = False
        self.load_models()

    def load_models(self):
        """Load all trained models and vectorizers."""
        try:
            # Vectorizers
            self.tfidf_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'tfidf_vectorizer.pkl'))
            self.count_vec = joblib.load(os.path.join(DATA_PROC_DIR, 'count_vectorizer.pkl'))

            # Model A — Answer verification
            self.lr_model = joblib.load(os.path.join(MODEL_A_DIR, 'logistic_regression.pkl'))
            self.lr_scaler = joblib.load(os.path.join(MODEL_A_DIR, 'lr_scaler.pkl'))

            # Model A — XGBoost
            self.xgb_model = joblib.load(os.path.join(MODEL_A_DIR, 'xgboost.pkl'))

            # Model B — Distractor ranker
            self.dist_ranker = joblib.load(os.path.join(MODEL_B_DIR, 'distractor_ranker.pkl'))
            self.dist_scaler = joblib.load(os.path.join(MODEL_B_DIR, 'distractor_scaler.pkl'))

            # Model B — Hint scorer
            self.hint_scorer = joblib.load(os.path.join(MODEL_B_DIR, 'hint_scorer.pkl'))
            self.hint_scaler = joblib.load(os.path.join(MODEL_B_DIR, 'hint_scaler.pkl'))

            # Dataset for random samples
            self.test_df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'test.csv'))

            self.models_loaded = True
        except Exception as e:
            print(f"Warning: Could not load some models: {e}")
            self.models_loaded = False

    def get_random_sample(self):
        """Get a random sample from the test set."""
        if not hasattr(self, 'test_df'):
            return None
        sample = self.test_df.sample(1).iloc[0]
        return {
            'id': sample['id'],
            'article': sample['article'],
            'question': sample['question'],
            'options': {
                'A': sample['A'],
                'B': sample['B'],
                'C': sample['C'],
                'D': sample['D'],
            },
            'correct_answer': sample['answer'],
            'correct_text': sample[sample['answer']],
        }

    def get_next_question_same_article(self, current_article, current_question):
        """Get a different question from the same article. Falls back to new random sample."""
        if not hasattr(self, 'test_df'):
            return self.get_random_sample()
        same_article = self.test_df[
            (self.test_df['article'] == current_article) &
            (self.test_df['question'] != current_question)
        ]
        if len(same_article) > 0:
            sample = same_article.sample(1).iloc[0]
        else:
            return self.get_random_sample()
        return {
            'id': sample['id'],
            'article': sample['article'],
            'question': sample['question'],
            'options': {
                'A': sample['A'],
                'B': sample['B'],
                'C': sample['C'],
                'D': sample['D'],
            },
            'correct_answer': sample['answer'],
            'correct_text': sample[sample['answer']],
        }

    def verify_answer(self, article, question, selected_option_text):
        """
        Verify if the selected option is the correct answer.
        Returns: (is_correct_probability, features_dict)
        """
        start_time = time.time()

        article_clean = clean_text(article)
        question_clean = clean_text(question)
        option_clean = clean_text(selected_option_text)

        feat = {}

        # TF-IDF similarities
        art_vec = self.tfidf_vec.transform([article_clean])
        q_vec = self.tfidf_vec.transform([question_clean])
        opt_vec = self.tfidf_vec.transform([option_clean])

        feat['tfidf_cos_art_opt'] = cosine_similarity(art_vec, opt_vec)[0, 0]
        feat['tfidf_cos_q_opt'] = cosine_similarity(q_vec, opt_vec)[0, 0]
        feat['tfidf_cos_art_q'] = cosine_similarity(art_vec, q_vec)[0, 0]

        # One-Hot similarities
        art_oh = self.count_vec.transform([article_clean])
        q_oh = self.count_vec.transform([question_clean])
        opt_oh = self.count_vec.transform([option_clean])

        feat['oh_cos_art_opt'] = cosine_similarity(art_oh, opt_oh)[0, 0]
        feat['oh_cos_q_opt'] = cosine_similarity(q_oh, opt_oh)[0, 0]

        # Handcrafted
        art_words = set(article_clean.split())
        q_words = set(question_clean.split())
        opt_words = set(option_clean.split())

        feat['option_word_count'] = len(option_clean.split())
        feat['question_word_count'] = len(question_clean.split())
        feat['article_word_count'] = len(article_clean.split())
        feat['overlap_art_opt'] = len(art_words & opt_words) / max(len(opt_words), 1)
        feat['overlap_q_opt'] = len(q_words & opt_words) / max(len(opt_words), 1)
        feat['overlap_art_q'] = len(art_words & q_words) / max(len(q_words), 1)

        X = np.array(list(feat.values())).reshape(1, -1)
        X_scaled = self.lr_scaler.transform(X)

        prob = self.lr_model.predict_proba(X_scaled)[0][1]
        latency = time.time() - start_time

        return prob, feat, latency

    def rank_all_options(self, article, question, options):
        """
        Rank all 4 options by probability of being correct.
        Returns: list of (option_label, option_text, probability) sorted descending.
        """
        start_time = time.time()
        scored = []
        for label, text in options.items():
            prob, _, _ = self.verify_answer(article, question, text)
            scored.append((label, text, prob))

        scored.sort(key=lambda x: x[2], reverse=True)
        latency = time.time() - start_time
        return scored, latency

    def generate_distractors(self, article, question, correct_answer):
        """Generate 3 plausible distractors."""
        start_time = time.time()
        distractors = generate_distractors(
            article, question, correct_answer,
            self.tfidf_vec, self.count_vec,
            self.dist_ranker, self.dist_scaler
        )
        latency = time.time() - start_time
        return distractors, latency

    def generate_hints(self, article, question, n_hints=3):
        """Generate graduated hints."""
        start_time = time.time()
        hints = generate_graduated_hints(article, question, self.tfidf_vec, n_hints)
        latency = time.time() - start_time
        return hints, latency

    def generate_questions(self, article, correct_answer):
        """Generate questions from article using templates."""
        start_time = time.time()
        questions = generate_questions_from_article(article, correct_answer, self.tfidf_vec)
        latency = time.time() - start_time
        return questions, latency

    def full_inference(self, article, question=None, options=None):
        """
        Run full inference pipeline:
          1. If no question provided, generate one
          2. If no options provided, generate distractors
          3. Verify/rank all options
          4. Generate hints
        """
        result = {'latencies': {}}

        # Use provided or generate
        if question is None:
            # Need a correct answer for generation — use first sentence heuristic
            questions, q_lat = self.generate_questions(article, article.split('.')[0])
            result['latencies']['question_gen'] = q_lat
            if questions:
                question = questions[0]['question']
                result['generated_question'] = question
            else:
                question = "What is the main idea of this passage?"
                result['generated_question'] = question

        result['question'] = question

        if options is None:
            # Generate distractors (use first sentence as "correct answer" placeholder)
            correct_text = article.split('.')[0].strip()
            distractors, d_lat = self.generate_distractors(article, question, correct_text)
            result['latencies']['distractor_gen'] = d_lat
            options = {'A': correct_text}
            for i, d in enumerate(distractors[:3]):
                options[chr(66 + i)] = d  # B, C, D
            result['generated_options'] = options

        result['options'] = options

        # Rank options
        ranked, rank_lat = self.rank_all_options(article, question, options)
        result['ranked_options'] = ranked
        result['predicted_answer'] = ranked[0][0]
        result['latencies']['ranking'] = rank_lat

        # Generate hints
        hints, h_lat = self.generate_hints(article, question)
        result['hints'] = hints
        result['latencies']['hints'] = h_lat

        result['total_latency'] = sum(result['latencies'].values())
        return result
