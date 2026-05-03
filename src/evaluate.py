"""
evaluate.py — Metric computation for Model A and Model B.

Computes and reports:
  - Accuracy, Macro F1, Precision, Recall, Exact Match
  - Confusion Matrix
  - RMSE, MAE, MSE, R² Score
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_A_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_a', 'traditional')


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'exact_match': np.mean(y_true == y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_prob))
        metrics['mae'] = mean_absolute_error(y_true, y_prob)
        metrics['mse'] = mean_squared_error(y_true, y_prob)
        metrics['r2'] = r2_score(y_true, y_prob)

    return metrics


def print_metrics(name, metrics):
    """Pretty-print metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  Exact Match:  {metrics['exact_match']:.4f}")
    if 'rmse' in metrics:
        print(f"  RMSE:         {metrics['rmse']:.4f}")
        print(f"  MAE:          {metrics['mae']:.4f}")
        print(f"  MSE:          {metrics['mse']:.4f}")
        print(f"  R²:           {metrics['r2']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"{'='*60}")


def evaluate_all_models():
    """Evaluate all Model A classifiers on the test set."""
    print("Loading test features...")
    X_test, y_test = joblib.load(os.path.join(DATA_PROC_DIR, 'test_features.pkl'))

    models = {
        'Logistic Regression': ('logistic_regression.pkl', 'lr_scaler.pkl'),
        'SVM': ('svm.pkl', 'svm_scaler.pkl'),
        'XGBoost': ('xgboost.pkl', None),
    }

    results = []
    for name, (model_file, scaler_file) in models.items():
        model_path = os.path.join(MODEL_A_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"  Skipping {name}: model file not found")
            continue

        model = joblib.load(model_path)
        X = X_test.copy()

        if scaler_file:
            scaler = joblib.load(os.path.join(MODEL_A_DIR, scaler_file))
            X = scaler.transform(X)

        y_pred = model.predict(X)

        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]

        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        print_metrics(f"{name} (TEST SET)", metrics)

        results.append({
            'model': name,
            'accuracy': metrics['accuracy'],
            'f1': metrics['macro_f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'exact_match': metrics['exact_match'],
        })

    results_df = pd.DataFrame(results)
    print("\nFinal comparison:")
    print(results_df.to_string(index=False))
    return results_df


if __name__ == '__main__':
    evaluate_all_models()
