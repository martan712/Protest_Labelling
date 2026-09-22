"""Loading of labelled training and development sets.

Locked test labels intentionally do not appear in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..taxonomy import CLASS_NAMES
from ..text import normalize_notes
from .contracts import ANNOTATED_COLUMNS, validate_labels


@dataclass(frozen=True)
class TrainingRelease:
    name: str
    train: pd.DataFrame
    dev: pd.DataFrame


def load_annotated(path: Path, *, context: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    validate_labels(frame, context=context)
    missing = set(ANNOTATED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)}")
    frame["notes"] = frame.notes.map(normalize_notes)
    if frame.notes.eq("").any():
        raise ValueError(f"{context} contains empty notes")
    invalid = sorted(set(frame.primary_label) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"{context} contains invalid labels: {invalid}")
    return frame


def load_training_release(train_path: Path, dev_path: Path) -> TrainingRelease:
    return TrainingRelease(
        name=train_path.stem,
        train=load_annotated(train_path, context="training"),
        dev=load_annotated(dev_path, context="development"),
    )
