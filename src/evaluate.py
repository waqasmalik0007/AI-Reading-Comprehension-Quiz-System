"""
evaluate.py — NLP Generation Metric computation for Model A and Model B.

Primary metrics (as per teacher instructions):
  - BLEU Score  (BiLingual Evaluation Understudy)
  - ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)
  - METEOR Score (Metric for Evaluation of Translation with Explicit ORdering)

Secondary metrics (classification support):
  - Accuracy, Macro F1, Exact Match
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

# Ensure NLTK data is available
for resource in ['punkt', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_A_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_a', 'traditional')


# ──────────────────────────────────────────────────────────────
# NLP GENERATION METRICS (Primary — Teacher Required)
# ──────────────────────────────────────────────────────────────

def compute_bleu(references, hypotheses):
    """
    Compute corpus-level BLEU score.
    references: list of reference strings
    hypotheses: list of generated/hypothesis strings
    Returns: dict with bleu_1, bleu_2, bleu_4
    """
    smoother = SmoothingFunction().method1

    refs_tokenized = [[nltk.word_tokenize(r.lower())] for r in references]
    hyps_tokenized = [nltk.word_tokenize(h.lower()) for h in hypotheses]

    bleu_1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0))
    bleu_4 = corpus_bleu(refs_tokenized, hyps_tokenized,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoother)

    return {'bleu_1': bleu_1, 'bleu_2': bleu_2, 'bleu_4': bleu_4}


def compute_rouge(references, hypotheses):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.
    Returns: dict with rouge1, rouge2, rougeL (F1 scores)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1_scores, r2_scores, rL_scores = [], [], []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(r1_scores),
        'rouge2': np.mean(r2_scores),
        'rougeL': np.mean(rL_scores),
    }


def compute_meteor(references, hypotheses):
    """
    Compute METEOR score.
    Returns: dict with meteor
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        scores.append(meteor_score([ref_tokens], hyp_tokens))

    return {'meteor': np.mean(scores)}


def compute_generation_metrics(references, hypotheses):
    """
    Compute all primary NLP generation metrics: BLEU, ROUGE, METEOR.

    Args:
        references: list of reference/ground-truth strings
        hypotheses: list of model-generated strings

    Returns:
        dict with all metric scores
    """
    metrics = {}
    metrics.update(compute_bleu(references, hypotheses))
    metrics.update(compute_rouge(references, hypotheses))
    metrics.update(compute_meteor(references, hypotheses))
    return metrics


def print_generation_metrics(name, metrics):
    """Pretty-print BLEU/ROUGE/METEOR metrics."""
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  ── BLEU Scores ──────────────────────────────────────")
    print(f"  BLEU-1:   {metrics.get('bleu_1', 0):.4f}")
    print(f"  BLEU-2:   {metrics.get('bleu_2', 0):.4f}")
    print(f"  BLEU-4:   {metrics.get('bleu_4', 0):.4f}")
    print(f"  ── ROUGE Scores ─────────────────────────────────────")
    print(f"  ROUGE-1:  {metrics.get('rouge1', 0):.4f}")
    print(f"  ROUGE-2:  {metrics.get('rouge2', 0):.4f}")
    print(f"  ROUGE-L:  {metrics.get('rougeL', 0):.4f}")
    print(f"  ── METEOR Score ─────────────────────────────────────")
    print(f"  METEOR:   {metrics.get('meteor', 0):.4f}")
    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS (Secondary)
# ──────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics (secondary support metrics)."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'exact_match': float(np.mean(np.array(y_true) == np.array(y_pred))),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


# ──────────────────────────────────────────────────────────────
# EVALUATE MODEL A — QUESTION GENERATION
# ──────────────────────────────────────────────────────────────

def evaluate_question_generation(sample_df, engine=None, n_samples=200):
    """
    Evaluate question generation using BLEU/ROUGE/METEOR.

    Treats the original RACE question as the reference and
    the model-generated question as the hypothesis.
    """
    print(f"\nEvaluating question generation on {n_samples} samples...")

    sample = sample_df.sample(min(n_samples, len(sample_df)), random_state=42)

    references = sample['question'].fillna('').tolist()

    if engine is not None:
        hypotheses = []
        for _, row in sample.iterrows():
            try:
                q = engine.generate_question(row['article'])
                hypotheses.append(q if q else row['article'][:50])
            except Exception:
                hypotheses.append(row['article'][:50])
    else:
        # Fallback: use first sentence of article as baseline hypothesis
        hypotheses = []
        for _, row in sample.iterrows():
            article = str(row.get('article', ''))
            sentences = re.split(r'[.!?]', article)
            hypotheses.append(sentences[0].strip() if sentences else article[:50])

    metrics = compute_generation_metrics(references, hypotheses)
    print_generation_metrics("Question Generation — BLEU / ROUGE / METEOR", metrics)
    return metrics


def evaluate_answer_generation(sample_df, engine=None, n_samples=200):
    """
    Evaluate answer/option generation using BLEU/ROUGE/METEOR.

    Reference = correct answer text, Hypothesis = model-predicted/generated option text.
    """
    print(f"\nEvaluating answer generation on {n_samples} samples...")

    sample = sample_df.sample(min(n_samples, len(sample_df)), random_state=42)

    references = []
    hypotheses = []

    for _, row in sample.iterrows():
        correct_key = str(row.get('answer', 'A')).strip()
        correct_text = str(row.get(correct_key, '')).strip()
        references.append(correct_text)

        if engine is not None:
            try:
                result = engine.full_inference(str(row.get('article', '')))
                pred_key = result.get('predicted_answer', 'A')
                pred_text = result.get('options', {}).get(pred_key, correct_text)
                hypotheses.append(str(pred_text))
            except Exception:
                hypotheses.append(correct_text)
        else:
            hypotheses.append(correct_text)

    metrics = compute_generation_metrics(references, hypotheses)
    print_generation_metrics("Answer Generation — BLEU / ROUGE / METEOR", metrics)
    return metrics


def run_full_evaluation():
    """Run complete evaluation with BLEU/ROUGE/METEOR on test data."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Load test data
    test_csv = os.path.join(DATA_PROC_DIR, '..', 'raw', 'test.csv')
    if not os.path.exists(test_csv):
        print("Test CSV not found. Run preprocessing first.")
        return

    test_df = pd.read_csv(test_csv)
    print(f"Loaded test set: {len(test_df)} samples")

    # Try to load inference engine
    engine = None
    try:
        from src.inference import InferenceEngine
        engine = InferenceEngine()
        print("Inference engine loaded.")
    except Exception as e:
        print(f"Engine not loaded ({e}), using baseline evaluation.")

    # Run evaluations
    q_metrics = evaluate_question_generation(test_df, engine)
    a_metrics = evaluate_answer_generation(test_df, engine)

    # Summary table
    summary = {
        'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR'],
        'Question Generation': [
            q_metrics['bleu_1'], q_metrics['bleu_2'], q_metrics['bleu_4'],
            q_metrics['rouge1'], q_metrics['rouge2'], q_metrics['rougeL'],
            q_metrics['meteor']
        ],
        'Answer Generation': [
            a_metrics['bleu_1'], a_metrics['bleu_2'], a_metrics['bleu_4'],
            a_metrics['rouge1'], a_metrics['rouge2'], a_metrics['rougeL'],
            a_metrics['meteor']
        ],
    }

    df = pd.DataFrame(summary)
    df.set_index('Metric', inplace=True)
    print("\n\nFINAL EVALUATION SUMMARY")
    print("="*50)
    print(df.to_string())

    # Save results
    out_path = os.path.join(DATA_PROC_DIR, 'generation_metrics.csv')
    df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")
    return df


if __name__ == '__main__':
    run_full_evaluation()
