"""Pure prediction-scoring functions."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ..taxonomy import CLASS_NAMES, accepted


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [float("nan"), float("nan")]
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    ) / denominator
    return [centre - margin, centre + margin]


def _alternatives(value: object) -> tuple[str, ...]:
    return tuple(label.strip() for label in str(value or "").split("|") if label.strip())


def score(frame: pd.DataFrame, predictions: list[str]) -> dict:
    if len(frame) != len(predictions):
        raise ValueError("Prediction count differs from gold row count")
    truth = frame.primary_label.tolist()
    strict = [prediction == label for prediction, label in zip(predictions, truth)]
    lenient = [
        accepted(prediction, label, _alternatives(alternatives))
        for prediction, label, alternatives
        in zip(predictions, truth, frame.alternative_labels)
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=CLASS_NAMES, zero_division=0
    )
    return {
        "n": len(frame),
        "strict_accuracy": float(np.mean(strict)),
        "strict_accuracy_95ci": wilson_interval(sum(strict), len(frame)),
        "macro_f1": float(f1_score(
            truth, predictions, labels=CLASS_NAMES, average="macro", zero_division=0
        )),
        "accepted_accuracy": float(np.mean(lenient)),
        "accepted_accuracy_95ci": wilson_interval(sum(lenient), len(frame)),
        "per_class": {
            label: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_NAMES)
        },
        "labels": list(CLASS_NAMES),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CLASS_NAMES).tolist(),
    }


def examples(frame: pd.DataFrame, predictions: list[str], *, errors: int = 10, correct: int = 5) -> dict:
    rows = []
    for (_, row), prediction in zip(frame.iterrows(), predictions):
        rows.append({
            "event_id_cnty": str(row.event_id_cnty), "notes": str(row.notes),
            "primary_label": str(row.primary_label), "prediction": prediction,
            "strict_correct": prediction == row.primary_label,
            "accepted_correct": accepted(
                prediction, row.primary_label, _alternatives(row.alternative_labels)
            ),
        })
    return {
        "correct": [row for row in rows if row["strict_correct"]][:correct],
        "errors": [row for row in rows if not row["strict_correct"]][:errors],
    }
