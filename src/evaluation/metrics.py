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
    matthews_corrcoef,
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
    n_params=None,
    train_time_sec=None,
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

        # Additional summary metrics
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_params": float(n_params) if n_params is not None else None,
        "train_time_sec": float(train_time_sec) if train_time_sec is not None else None,
    }

    return metrics


def compute_topic_breakdown(y_true, y_prob, topics, threshold: float = 0.5) -> dict | None:
    """
    Per-Topic accuracy/F1 breakdown for a single split, to check whether the
    model depends on topics over-represented in train.

    Assumes y_true/y_prob and topics are already positionally aligned (same
    row order as the corpus PKL the predictions were made on) -- callers are
    responsible for that alignment; this function does not join by Id.
    Returns None if topics is empty/None or lengths don't match, so callers
    can store a null topic_breakdown rather than fail the whole run.
    """
    if topics is None:
        return None

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    topics = np.asarray(topics)

    if len(topics) != len(y_true):
        return None

    y_pred = (y_prob >= threshold).astype(int)

    breakdown = {}
    for topic in sorted(set(topics.tolist())):
        mask = topics == topic
        n = int(mask.sum())
        if n == 0:
            continue

        yt, yp = y_true[mask], y_pred[mask]
        breakdown[str(topic)] = {
            "n": n,
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }

    return breakdown or None


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