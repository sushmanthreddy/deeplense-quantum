"""Dependency-light classification metrics and paired prediction artifacts."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal ROC AUC with correct handling of tied score thresholds."""

    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    sorted_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positive = np.cumsum(y)[distinct]
    false_positive = 1 + distinct - true_positive
    tpr = np.r_[0.0, true_positive / positives]
    fpr = np.r_[0.0, false_positive / negatives]
    return float(np.trapz(tpr, fpr))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, class_names: List[str]
) -> Dict:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    classes = len(class_names)
    matrix = confusion_matrix(labels, predictions, classes)
    per_class = {}
    f1_values = []
    recalls = []
    auc_values = []
    for label, name in enumerate(class_names):
        tp = int(matrix[label, label])
        fp = int(matrix[:, label].sum() - tp)
        fn = int(matrix[label, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        auc = binary_auc(labels == label, probabilities[:, label])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_ovr": auc,
            "support": int(matrix[label].sum()),
        }
        f1_values.append(f1)
        recalls.append(recall)
        auc_values.append(auc)

    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(classes, dtype=np.float64)[labels]
    return {
        "samples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_auc_ovr": float(np.nanmean(auc_values)),
        "nll": float(-np.log(clipped).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15": expected_calibration_error(probabilities, labels, bins=15),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
