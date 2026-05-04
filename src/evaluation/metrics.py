# src/evaluation/metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (y_prob > lower) & (y_prob <= upper)

        if mask.sum() == 0:
            continue

        confidence = y_prob[mask].mean()
        accuracy = y_true[mask].mean()
        weight = mask.mean()

        ece += weight * abs(accuracy - confidence)

    return float(ece)


def binary_entropy(y_prob, eps=1e-12):
    y_prob = np.asarray(y_prob)
    y_prob = np.clip(y_prob, eps, 1 - eps)

    entropy = -(
        y_prob * np.log(y_prob)
        + (1 - y_prob) * np.log(1 - y_prob)
    )

    return entropy


def evaluate_binary_classifier(
    y_true,
    y_prob,
    threshold=0.5,
    n_bins=10,
):
    """
    Standard:
    y_true:
      1 = Fake
      0 = True / Real

    y_prob:
      probability of class 1 = Fake
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    entropy_values = binary_entropy(y_prob)

    metrics = {
        "threshold": float(threshold),

        # Core metrics
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),

        # Ranking / probability metrics
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),

        # Calibration / uncertainty
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=n_bins)),
        "entropy_mean": float(entropy_values.mean()),
        "entropy_std": float(entropy_values.std()),

        # Confusion matrix values
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        # Error rates
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }

    return metrics


def save_metrics(metrics: dict, output_dir: str | Path, prefix: str = "metrics"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    return json_path, csv_path


def metrics_to_latex_row(model_name: str, metrics: dict):
    return (
        f"{model_name} & "
        f"{metrics['accuracy']:.4f} & "
        f"{metrics['precision']:.4f} & "
        f"{metrics['recall']:.4f} & "
        f"{metrics['f1']:.4f} & "
        f"{metrics['roc_auc']:.4f} & "
        f"{metrics['pr_auc']:.4f} \\\\"
    )