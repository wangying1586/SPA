from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == labels
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.mean() * abs(bin_accuracy - bin_confidence)
    return float(ece * 100.0)


def negative_log_likelihood(
    probabilities: np.ndarray, labels: np.ndarray
) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.log(clipped[np.arange(labels.size), labels]).mean())


def multiclass_brier(
    probabilities: np.ndarray, labels: np.ndarray
) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.square(probabilities - one_hot).sum(axis=1).mean())


def _macro_f1(confusion: np.ndarray) -> float:
    f1_values = []
    for class_index in range(confusion.shape[0]):
        tp = confusion[class_index, class_index]
        fp = confusion[:, class_index].sum() - tp
        fn = confusion[class_index, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
    return float(np.mean(f1_values) * 100.0)


def compute_classification_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError("probability matrix has an unexpected shape")
    if probabilities.shape[0] != labels.size:
        raise ValueError("probabilities and labels have different sample counts")

    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label, predicted_label] += 1

    counts = confusion.sum(axis=1)
    hits = np.diag(confusion)
    specificity = 100.0 * hits[0] / max(counts[0], 1)
    sensitivity = 100.0 * hits[1:].sum() / max(counts[1:].sum(), 1)
    score = 0.5 * (specificity + sensitivity)
    accuracy = 100.0 * np.trace(confusion) / max(confusion.sum(), 1)

    return {
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "score": float(score),
        "accuracy": float(accuracy),
        "macro_f1": _macro_f1(confusion),
        "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "nll": negative_log_likelihood(probabilities, labels),
        "brier": multiclass_brier(probabilities, labels),
        "samples": int(labels.size),
        "confusion_matrix": confusion.tolist(),
        "class_counts": counts.tolist(),
    }
