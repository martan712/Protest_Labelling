"""Dataframe schemas and split invariants."""

from __future__ import annotations

import pandas as pd

from ..taxonomy import CLASS_NAMES, OTHER_REASONS
from ..text import normalize_notes, note_hash

LABEL_COLUMNS = (
    "event_id_cnty", "primary_label", "alternative_labels", "other_reason",
    "evidence", "annotator",
)
MANIFEST_COLUMNS = (
    "event_id_cnty", "event_date", "year", "country", "notes", "note_hash",
)
ANNOTATED_COLUMNS = (
    "event_id_cnty", "notes", "note_hash", "primary_label",
    "alternative_labels", "other_reason", "evidence", "annotator",
)


def validate_manifest(frame: pd.DataFrame, *, context: str = "manifest") -> pd.DataFrame:
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)}")
    if frame.event_id_cnty.duplicated().any():
        raise ValueError(f"{context} contains duplicate event IDs")
    normalized = frame.notes.map(normalize_notes)
    if normalized.eq("").any():
        raise ValueError(f"{context} contains empty notes")
    if not normalized.map(note_hash).eq(frame.note_hash.astype(str)).all():
        raise ValueError(f"{context} note hashes do not match normalized notes")
    return frame


def validate_labels(frame: pd.DataFrame, *, context: str = "annotations") -> pd.DataFrame:
    missing = set(LABEL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)}")
    if frame.event_id_cnty.duplicated().any():
        raise ValueError(f"{context} contains duplicate event IDs")
    invalid = sorted(set(frame.primary_label) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"{context} contains invalid labels: {invalid}")
    alternatives = {
        label.strip()
        for value in frame.alternative_labels.fillna("").astype(str)
        for label in value.split("|") if label.strip()
    }
    invalid_alternatives = sorted(alternatives - set(CLASS_NAMES))
    if invalid_alternatives:
        raise ValueError(f"{context} contains invalid alternatives: {invalid_alternatives}")
    for _, row in frame.iterrows():
        row_alternatives = [
            label.strip() for label in str(row.alternative_labels).split("|") if label.strip()
        ]
        if len(row_alternatives) > 1:
            raise ValueError(f"{context}: at most one alternative label is allowed")
        if row.primary_label in row_alternatives:
            raise ValueError(f"{context}: the primary label cannot be its own alternative")
    other = frame.primary_label.eq("other")
    reasons = frame.other_reason.fillna("").astype(str)
    if (~reasons[other].isin(OTHER_REASONS)).any():
        raise ValueError(f"{context}: every 'other' label needs a valid other_reason")
    if reasons[~other].ne("").any():
        raise ValueError(f"{context}: other_reason is only valid for 'other'")
    if frame.evidence.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{context}: evidence is required")
    if frame.annotator.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"{context}: annotator is required")
    return frame


def assert_disjoint(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> None:
    for column in ("event_id_cnty", "note_hash"):
        overlap = set(left[column].astype(str)) & set(right[column].astype(str))
        if overlap:
            raise ValueError(
                f"Leakage: {left_name} and {right_name} share {len(overlap)} "
                f"{column} value(s): {sorted(overlap)[:10]}"
            )


def assert_nested(
    smaller: pd.DataFrame,
    larger: pd.DataFrame,
    *,
    smaller_name: str,
    larger_name: str,
) -> None:
    missing = set(smaller.event_id_cnty) - set(larger.event_id_cnty)
    if missing:
        raise ValueError(
            f"{larger_name} is not a superset of {smaller_name}; missing {len(missing)} IDs"
        )
