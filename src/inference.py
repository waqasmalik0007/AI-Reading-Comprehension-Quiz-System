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

    def get_next_question_same_article(self, current_article, current_question, used_answers=None):
        """Get a different question from the same article.
        For dataset samples: queries test_df for another row with same article.
        For custom text: re-runs inference with a different key answer.
        Falls back to new random sample only if nothing else is possible.
        """
        # Try dataset first
        if hasattr(self, 'test_df'):
            same_article = self.test_df[
                (self.test_df['article'] == current_article) &
                (self.test_df['question'] != current_question)
            ]
            if len(same_article) > 0:
                sample = same_article.sample(1).iloc[0]
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

        # Custom text — generate a new question from same article with different answer
        result = self.full_inference(current_article, exclude_answers=used_answers or [])
        new_sample = {
            'id': 'custom',
            'article': current_article,
            'question': result['question'],
            'options': result['options'],
            'correct_answer': 'A',
            'correct_text': result['options'].get('A', ''),
            '_is_custom': True,
        }
        # Return if genuinely different question
        if new_sample['question'] != current_question:
            return new_sample

        # Same question — try once more with a broader exclusion list
        broader_exclude = (used_answers or []) + [new_sample['correct_text']]
        result2 = self.full_inference(current_article, exclude_answers=broader_exclude)
        new_sample2 = {
            'id': 'custom',
            'article': current_article,
            'question': result2['question'],
            'options': result2['options'],
            'correct_answer': 'A',
            'correct_text': result2['options'].get('A', ''),
            '_is_custom': True,
        }
        if new_sample2['question'] != current_question:
            return new_sample2

        # Still same — return new_sample (different options at minimum) rather than
        # changing the article to a random one
        return new_sample

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

    def _extract_key_answer(self, article, exclude_answers=None):
        """
        Extract a meaningful key phrase from the article to use as the correct answer.
        Prefers named entities (capitalized phrases), numbers, or important nouns.
        Avoids returning the full first sentence.
        """
        import re

        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
            'at', 'by', 'for', 'with', 'about', 'from', 'as', 'into', 'through',
            'he', 'she', 'it', 'they', 'we', 'i', 'you', 'his', 'her', 'its',
            'their', 'our', 'and', 'but', 'or', 'so', 'yet', 'both', 'either',
            'not', 'that', 'this', 'these', 'those', 'which', 'who', 'whom',
        }

        exclude_set = {e.lower() for e in (exclude_answers or [])}

        sentences = re.split(r'[.!?]+', article)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        def _ok(phrase, src_sent):
            return phrase.lower() not in STOPWORDS and phrase.lower() not in exclude_set and len(phrase) > 3

        # Priority 1: all number phrases — pick first not excluded
        num_matches = re.findall(
            r'\b(age of \w+|\w+ hours?|\w+ years?|\w+ percent|\w+ million|\w+ thousand|'
            r'\d{4}|\d+)\b', article, re.IGNORECASE
        )
        for phrase in num_matches:
            phrase = phrase.strip()
            src = next((s for s in sentences if phrase.lower() in s.lower()), sentences[0] if sentences else article)
            if _ok(phrase, src):
                return phrase, src

        # Priority 2: all capitalized proper nouns — pick first not excluded
        proper_nouns = re.findall(r'\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )?[A-Z][a-z]+)\b', article)
        for candidate in proper_nouns:
            if len(candidate.split()) >= 2:
                src = next((s for s in sentences if candidate in s), sentences[0] if sentences else article)
                if _ok(candidate, src):
                    return candidate, src

        # Priority 3: single capitalized names (not at sentence start)
        for sent in sentences:
            words = sent.split()
            for w in words[1:]:
                if w[0].isupper() and len(w) > 3 and w.lower() not in STOPWORDS and w.lower() not in exclude_set:
                    return w, sent

        # Fallback: meaningful words from different sentences
        for sent in sentences:
            words = [w for w in sent.split() if w.lower() not in STOPWORDS and len(w) > 4 and w.lower() not in exclude_set]
            if words:
                return ' '.join(words[:3]), sent

        return article.split('.')[0][:60].strip(), sentences[0] if sentences else article

    def _build_question(self, key_answer, source_sentence, article):
        """Build a natural question from the key answer and its source sentence."""
        import re

        s = source_sentence.strip()

        # Detect if answer is a number/age/duration
        if re.search(r'\d|years?|hours?|percent|million|age', key_answer, re.IGNORECASE):
            q = re.sub(re.escape(key_answer), '________', s, flags=re.IGNORECASE, count=1)
            if q != s:
                return f"Fill in the blank: {q}"
            return f"How much or how many? '{key_answer}' is mentioned in the passage — what does it refer to?"

        # Detect if answer is a person/proper noun
        if key_answer[0].isupper() and len(key_answer.split()) >= 2:
            q_who = re.sub(re.escape(key_answer), 'who', s, flags=re.IGNORECASE, count=1)
            q_blank = re.sub(re.escape(key_answer), '________', s, flags=re.IGNORECASE, count=1)
            if q_who != s and q_who.lower().startswith('who'):
                q_who = q_who.strip().rstrip('.').rstrip('?').rstrip('!')
                return q_who[0].upper() + q_who[1:] + '?'
            elif q_blank != s:
                return f"Fill in the blank: {q_blank}"

        # Default — cloze style
        q = re.sub(re.escape(key_answer), '________', s, flags=re.IGNORECASE, count=1)
        if q != s:
            return f"Fill in the blank: {q}"

        return f"According to the passage, what is '{key_answer}'?"

    def full_inference(self, article, question=None, options=None, exclude_answers=None):
        """
        Run full inference pipeline:
          1. If no question provided, generate one
          2. If no options provided, generate distractors
          3. Verify/rank all options
          4. Generate hints
        """
        import re
        result = {'latencies': {}}

        STOPWORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'he', 'she', 'it', 'they', 'we', 'his', 'her', 'its', 'their',
            'and', 'but', 'or', 'so', 'not', 'that', 'this', 'with', 'for',
            'of', 'in', 'on', 'at', 'to', 'by', 'as', 'from', 'also',
        }

        # ── Step 1: Question generation ──
        if question is None:
            start_q = time.time()
            key_answer, source_sent = self._extract_key_answer(article, exclude_answers=exclude_answers)
            question = self._build_question(key_answer, source_sent, article)
            result['latencies']['question_gen'] = time.time() - start_q
            result['generated_question'] = question
            result['_key_answer'] = key_answer
        else:
            key_answer = article.split('.')[0][:60].strip()

        result['question'] = question

        # ── Step 2: Distractor generation ──
        if options is None:
            correct_text = result.get('_key_answer', key_answer)
            distractors, d_lat = self.generate_distractors(article, question, correct_text)
            result['latencies']['distractor_gen'] = d_lat

            # ── Type-consistent distractors ──
            def _type_consistent_distractors(answer, article_text):
                import random
                answer = answer.strip()

                # Year (4-digit number like 2019)
                if re.match(r'^\d{4}$', answer):
                    base = int(answer)
                    offsets = [-3, -2, -1, 1, 2, 3]
                    random.shuffle(offsets)
                    return [str(base + o) for o in offsets[:3]]

                # Pure number (e.g. "3", "40", "six", "forty")
                word_numbers = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
                    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                    'nineteen': 19, 'twenty': 20, 'twenty-one': 21, 'twenty-two': 22,
                    'twenty-three': 23, 'twenty-four': 24, 'twenty-five': 25,
                    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
                    'seventy': 70, 'eighty': 80, 'ninety': 90,
                    'hundred': 100, 'thousand': 1000,
                }
                num_words_rev = {v: k for k, v in word_numbers.items()}
                if answer.lower() in word_numbers:
                    base = word_numbers[answer.lower()]
                    candidates = []
                    for delta in [-2, -1, 1, 2, 3]:
                        v = base + delta
                        if v > 0:
                            candidates.append(num_words_rev.get(v, str(v)))
                    return candidates[:3]
                if re.match(r'^\d+$', answer):
                    base = int(answer)
                    return [str(base + d) for d in [-2, 1, 3] if base + d > 0]

                # Age phrase like "age of sixteen"
                age_match = re.match(r'age of (\w+)', answer, re.IGNORECASE)
                if age_match:
                    word = age_match.group(1).lower()
                    if word in word_numbers:
                        base = word_numbers[word]
                        alts = []
                        for delta in [-2, -1, 1, 2]:
                            v = base + delta
                            if v > 0:
                                alt_word = num_words_rev.get(v, str(v))
                                alts.append(f'age of {alt_word}')
                        return alts[:3]

                # Duration phrase like "thirty hours", "six months"
                dur_match = re.match(r'(\w+) (hours?|days?|months?|years?|weeks?|minutes?)', answer, re.IGNORECASE)
                if dur_match:
                    qty_word = dur_match.group(1).lower()
                    unit = dur_match.group(2)
                    if qty_word in word_numbers:
                        base = word_numbers[qty_word]
                        alts = []
                        for delta in [-2, -1, 1, 2]:
                            v = base + delta
                            if v > 0:
                                alt_word = num_words_rev.get(v, str(v))
                                alts.append(f'{alt_word} {unit}')
                        return alts[:3]

                # Academic subject / department
                ACADEMIC_SUBJECTS = [
                    'Computer Science', 'Mathematics', 'Physics', 'Chemistry', 'Biology',
                    'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
                    'Software Engineering', 'Data Science', 'Artificial Intelligence',
                    'Business Administration', 'Economics', 'Psychology', 'Sociology',
                    'Political Science', 'Literature', 'History', 'Philosophy', 'Law',
                    'Medicine', 'Architecture', 'Environmental Science', 'Statistics',
                    'Information Technology', 'Accounting', 'Finance', 'Marketing',
                ]
                academic_kws = {
                    'science', 'engineering', 'technology', 'mathematics', 'studies',
                    'administration', 'economics', 'psychology', 'sociology', 'literature',
                    'history', 'philosophy', 'medicine', 'architecture', 'statistics',
                    'accounting', 'finance', 'marketing', 'physics', 'chemistry', 'biology',
                }
                if (answer in ACADEMIC_SUBJECTS or
                        any(kw in answer.lower() for kw in academic_kws)):
                    others = [s for s in ACADEMIC_SUBJECTS if s.lower() != answer.lower()]
                    random.shuffle(others)
                    return others[:3]

                # University / institution
                UNIVERSITIES = [
                    'FAST University', 'NUST', 'LUMS', 'Karachi University',
                    'Punjab University', 'Quaid-i-Azam University', 'IBA Karachi',
                    'GCU Lahore', 'UET Lahore', 'COMSATS University',
                    'Air University', 'Bahria University', 'NED University',
                ]
                if re.search(r'university|college|institute|school', answer, re.IGNORECASE):
                    others = [u for u in UNIVERSITIES if u.lower() != answer.lower()]
                    random.shuffle(others)
                    return others[:3]

                # City / place
                CITIES = [
                    'Lahore', 'Karachi', 'Islamabad', 'Peshawar', 'Quetta',
                    'Multan', 'Faisalabad', 'Rawalpindi', 'Hyderabad', 'Sialkot',
                    'London', 'New York', 'Paris', 'Dubai', 'Toronto',
                ]
                if answer in CITIES:
                    others = [c for c in CITIES if c != answer]
                    random.shuffle(others)
                    return others[:3]

                # Profession / job title
                PROFESSIONS = [
                    'engineer', 'doctor', 'teacher', 'scientist', 'professor',
                    'researcher', 'developer', 'designer', 'lawyer', 'accountant',
                    'manager', 'analyst', 'journalist', 'architect', 'pilot',
                ]
                if answer.lower() in PROFESSIONS:
                    others = [p for p in PROFESSIONS if p.lower() != answer.lower()]
                    random.shuffle(others)
                    return others[:3]

                # Adjective — antonym / synonym groups
                ADJECTIVE_GROUPS = [
                    {'naughty', 'mischievous', 'notorious', 'well-behaved', 'obedient', 'disciplined', 'troublesome'},
                    {'good', 'bad', 'excellent', 'poor', 'great', 'terrible', 'outstanding', 'mediocre'},
                    {'happy', 'sad', 'joyful', 'miserable', 'cheerful', 'gloomy', 'elated', 'depressed'},
                    {'smart', 'clever', 'intelligent', 'foolish', 'brilliant', 'dumb', 'wise', 'stupid'},
                    {'fast', 'slow', 'quick', 'rapid', 'sluggish', 'swift', 'leisurely', 'hasty'},
                    {'big', 'small', 'large', 'tiny', 'huge', 'little', 'enormous', 'miniature'},
                    {'old', 'young', 'ancient', 'modern', 'new', 'outdated', 'contemporary', 'elderly'},
                    {'hard', 'easy', 'difficult', 'simple', 'tough', 'straightforward', 'challenging', 'complex'},
                    {'rich', 'poor', 'wealthy', 'broke', 'affluent', 'impoverished', 'prosperous'},
                    {'famous', 'unknown', 'popular', 'obscure', 'renowned', 'celebrated', 'infamous'},
                ]
                for group in ADJECTIVE_GROUPS:
                    if answer.lower() in group:
                        others = list(group - {answer.lower()})
                        random.shuffle(others)
                        return others[:3]

                # Nationality / language
                NATIONALITIES = [
                    'Pakistani', 'Indian', 'American', 'British', 'Canadian',
                    'Australian', 'Chinese', 'Japanese', 'German', 'French',
                ]
                if answer in NATIONALITIES:
                    others = [n for n in NATIONALITIES if n != answer]
                    random.shuffle(others)
                    return others[:3]

                # Person name — single capitalized word not matched above
                if re.match(r'^[A-Z][a-z]+$', answer) and len(answer) > 2:
                    # Words that look capitalized but are NOT person names
                    NON_NAME_CAPS = {
                        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                        'Saturday', 'Sunday', 'January', 'February', 'March',
                        'April', 'May', 'June', 'July', 'August', 'September',
                        'October', 'November', 'December', 'Pakistan', 'India',
                        'America', 'Britain', 'University', 'College', 'Institute',
                        'Lahore', 'Karachi', 'Islamabad', 'Peshawar', 'Quetta',
                        'Multan', 'Faisalabad', 'Rawalpindi', 'London', 'Dubai',
                    }
                    # Step 1: extract other name-like capitalized words from article
                    raw = re.findall(r'\b([A-Z][a-z]{2,})\b', article_text)
                    from_article = list(dict.fromkeys(
                        n for n in raw
                        if n != answer
                        and n not in NON_NAME_CAPS
                        and not any(kw in n.lower() for kw in academic_kws)
                    ))
                    if len(from_article) >= 3:
                        return from_article[:3]
                    # Step 2: supplement with curated names similar in cultural context
                    PERSON_NAMES = [
                        'Ali', 'Ahmed', 'Hassan', 'Omar', 'Usman', 'Bilal',
                        'Tariq', 'Zaid', 'Hamza', 'Faisal', 'Imran', 'Asad',
                        'Sara', 'Fatima', 'Ayesha', 'Maryam', 'Sana', 'Nadia',
                        'Hina', 'Zara', 'Alia', 'Rania', 'John', 'Michael',
                        'David', 'James', 'Emma', 'Sophia', 'Olivia', 'Amelia',
                    ]
                    extras = [n for n in PERSON_NAMES
                              if n.lower() != answer.lower() and n not in from_article]
                    random.shuffle(extras)
                    return (from_article + extras)[:3]

                # Generic fallback — extract same-structure phrases from article
                # (same word count, same capitalization) so distractors are contextually plausible
                ans_words = answer.split()
                ans_len = len(ans_words)
                if answer[0].isupper():
                    if ans_len == 1:
                        caps = re.findall(r'\b([A-Z][a-zA-Z]{3,})\b', article_text)
                        candidates = list(dict.fromkeys(
                            c for c in caps if c.lower() != answer.lower()
                        ))
                    else:
                        word_pat = r'[A-Z][a-zA-Z]+'
                        pat = r'\b(' + r'\s+'.join([word_pat] * ans_len) + r')\b'
                        candidates = list(dict.fromkeys(
                            c for c in re.findall(pat, article_text)
                            if c.lower() != answer.lower()
                        ))
                    if len(candidates) >= 3:
                        random.shuffle(candidates)
                        return candidates[:3]
                else:
                    # Lowercase noun — find other lowercase words of similar length
                    words_in_art = re.findall(r'\b([a-z]{4,})\b', article_text.lower())
                    STOPWORDS_SET = {
                        'that', 'this', 'with', 'from', 'have', 'been', 'were',
                        'they', 'them', 'their', 'will', 'would', 'could', 'should',
                        'when', 'where', 'which', 'what', 'also', 'into', 'over',
                    }
                    candidates = list(dict.fromkeys(
                        w for w in words_in_art
                        if w != answer.lower() and w not in STOPWORDS_SET
                        and abs(len(w) - len(answer)) <= 3
                    ))
                    if len(candidates) >= 3:
                        random.shuffle(candidates)
                        return candidates[:3]

                return []

            type_distractors = _type_consistent_distractors(correct_text, article)
            # Remove any distractor that already appears in the question text
            if type_distractors:
                q_lower = question.lower()
                type_distractors = [d for d in type_distractors if d.lower() not in q_lower]
            if type_distractors:
                # Type-consistent distractors are already good — skip the stopword filter
                options = {'A': correct_text}
                for i, d in enumerate(type_distractors[:3]):
                    options[chr(66 + i)] = d
                result['generated_options'] = options
                result['_custom_correct'] = 'A'
                result['options'] = options
                ranked, rank_lat = self.rank_all_options(article, question, options)
                result['ranked_options'] = ranked
                result['predicted_answer'] = 'A'
                result['latencies']['ranking'] = rank_lat
                hints, h_lat = self.generate_hints(article, question)
                result['hints'] = hints
                result['latencies']['hints'] = h_lat
                result['total_latency'] = sum(result['latencies'].values())
                return result

            # Filter stopwords and too-short distractors
            GENERIC_WORDS = {
                'university', 'school', 'student', 'person', 'people', 'place',
                'thing', 'time', 'year', 'month', 'day', 'way', 'part', 'point',
                'work', 'life', 'hand', 'case', 'week', 'company', 'system',
                'program', 'class', 'course', 'group', 'home', 'room', 'world',
            }

            def _is_valid_distractor(d):
                words = d.strip().split()
                if len(words) < 1 or len(d.strip()) < 5:
                    return False
                if d.lower().strip() in STOPWORDS:
                    return False
                # Reject single generic words
                if len(words) == 1 and words[0].lower() in GENERIC_WORDS:
                    return False
                # Reject if starts with article (e.g. "a notorious", "the best")
                if words[0].lower() in {'a', 'an', 'the'}:
                    return False
                # Reject if last word is a stopword (e.g. "one of the")
                if words[-1].lower() in STOPWORDS:
                    return False
                # Reject if more than half words are stopwords
                sw_count = sum(1 for w in words if w.lower() in STOPWORDS)
                if sw_count > len(words) / 2:
                    return False
                return True

            distractors = [d for d in distractors if _is_valid_distractor(d)]

            # If not enough good distractors, extract noun phrases from article
            if len(distractors) < 3:
                noun_phrases = re.findall(
                    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+|'
                    r'\w+ (?:University|Institute|City|Street|Park|Hospital|School))\b',
                    article
                )
                extras = [np for np in noun_phrases if np.lower() != correct_text.lower()]
                # Also extract number-based phrases
                num_phrases = re.findall(
                    r'\b(?:age of \w+|\d+ \w+|twenty|thirty|forty|fifty|sixty|'
                    r'hundred|thousand|million)\b', article, re.IGNORECASE
                )
                extras += [n for n in num_phrases if n.lower() != correct_text.lower()]
                distractors += extras[:3 - len(distractors)]

            options = {'A': correct_text}
            for i, d in enumerate(distractors[:3]):
                options[chr(66 + i)] = d
            # Fill missing options — prefer proper nouns from article, never sentence fragments
            if len([k for k in ['B', 'C', 'D'] if k not in options]) > 0:
                all_proper = re.findall(
                    r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', article
                )
                all_proper = list(dict.fromkeys(
                    p for p in all_proper
                    if p.lower() != correct_text.lower() and p not in options.values()
                ))
                pn_idx = 0
                for letter in ['B', 'C', 'D']:
                    if letter not in options:
                        while pn_idx < len(all_proper):
                            cand = all_proper[pn_idx]
                            pn_idx += 1
                            if cand not in options.values():
                                options[letter] = cand
                                break
            # Last resort: label-only placeholders (never sentence fragments)
            fallback_labels = ['Option B', 'Option C', 'Option D']
            fb_idx = 0
            for letter in ['B', 'C', 'D']:
                if letter not in options:
                    options[letter] = fallback_labels[fb_idx]
                fb_idx += 1
            result['generated_options'] = options
            # We extracted the answer ourselves — A is always correct for custom text
            result['_custom_correct'] = 'A'

        result['options'] = options

        # ── Step 3: Rank options ──
        ranked, rank_lat = self.rank_all_options(article, question, options)
        result['ranked_options'] = ranked
        # For custom text, correct answer is always A (we generated it)
        # For dataset samples, use the ML ranker
        result['predicted_answer'] = result.get('_custom_correct', ranked[0][0])
        result['latencies']['ranking'] = rank_lat

        # ── Step 4: Hints ──
        hints, h_lat = self.generate_hints(article, question)
        result['hints'] = hints
        result['latencies']['hints'] = h_lat

        result['total_latency'] = sum(result['latencies'].values())
        return result
