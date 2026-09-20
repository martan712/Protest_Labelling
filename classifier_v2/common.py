"""Shared helpers for the standalone ModernBERT protest classifier.

This module is deliberately independent from the legacy keyword pipeline in
``pipeline/``.  Nothing here reads ``clean_notes``, trigger keywords, old labels
or model predictions.  The only text the model ever sees is the original
``notes`` string.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

# Make ``evaluation.taxonomy`` importable regardless of the working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.taxonomy import CLASS_NAMES, OTHER_REASONS, validate_label  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

DATA_DIR = REPO_ROOT / "data"
CLASSIFIER_DATA_DIR = DATA_DIR / "classifier_v2"
MODELS_DIR = REPO_ROOT / "models" / "classifier_v2"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

# Locked, frozen artefacts.  Only evaluate.py --split test may read the gold.
DEV_GOLD = DATA_DIR / "run" / "dev_gold.csv"
TEST_MANIFEST_840 = REPO_ROOT / "evaluation" / "manifests" / "test_manifest_840.csv"
TEST_GOLD = DATA_DIR / "review" / "test_gold_v1" / "test_gold.csv"

#: Development gold IDs that also occur in the locked test set and must be removed.
DEV_REMOVE_IDS = ("DEU24281", "FRA33651", "ITA21511")

#: Where labelled semantic annotations live.  Ordered: earlier sources are the
#: smaller releases, later sources extend them, preserving the nested property.
#: ``test_gold_v1`` is intentionally absent and must never be added here.
ANNOTATION_SOURCES: tuple[tuple[str, str], ...] = (
    ("unknown_starter", "data/review/unknown_starter_v1/chunks/chunk_*.csv"),
    ("expanded", "data/review/expanded_v1/chunks/chunk_*.csv"),
    ("sampled", "data/review/sampled_v1/chunks/chunk_*.csv"),
)

TRAINING_RELEASE_SIZES: tuple[int, ...] = (800, 1500, 3000, 6000)

REQUIRED_ANNOTATION_COLUMNS = (
    "event_id_cnty",
    "notes",
    "note_hash",
    "primary_label",
    "other_reason",
    "reviewer",
)

# The columns written for training/dev releases (the required set plus the
# independently reviewed alternatives, which are only used for the secondary
# legacy accepted-pair score).
RELEASE_COLUMNS = (
    "event_id_cnty",
    "notes",
    "note_hash",
    "primary_label",
    "alternative_labels",
    "other_reason",
    "reviewer",
)

# --------------------------------------------------------------------------- #
# Text handling
# --------------------------------------------------------------------------- #

# C0/C1 control characters except tab/newline/carriage return, plus DEL.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")


def normalize_notes(value: object) -> str:
    """Replace invalid control characters and collapse whitespace.

    Wording, punctuation and case are preserved.  Used for both training and
    inference so the two paths can never disagree.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def sha256_text(value: object) -> str:
    """The project-wide note hash convention: sha256 of the UTF-8 string."""
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def reject_empty_notes(frame: pd.DataFrame, *, column: str = "notes", context: str = "") -> None:
    empty = frame[column].isna() | (frame[column].astype(str).str.strip() == "")
    if empty.any():
        ids = frame.loc[empty, "event_id_cnty"].head(10).tolist()
        raise ValueError(f"Empty notes found{(' in ' + context) if context else ''}: {ids}")


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #


def label_maps(class_names: list[str] | None = None) -> tuple[dict[int, str], dict[str, int]]:
    """Return ``(id2label, label2id)`` for the 21 classes.

    Training and inference both call this, so the mapping cannot drift.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    id2label = {index: name for index, name in enumerate(names)}
    label2id = {name: index for index, name in id2label.items()}
    return id2label, label2id


def validate_annotations(frame: pd.DataFrame, *, context: str = "annotations") -> pd.DataFrame:
    """Validate annotation columns, labels, ``other_reason`` and note content."""
    missing = set(REQUIRED_ANNOTATION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {sorted(missing)}")

    labels = frame["primary_label"].astype(str)
    invalid = sorted(set(labels) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"{context} contains labels outside the taxonomy: {invalid}")

    reasons = frame["other_reason"].fillna("").astype(str)
    is_other = labels == "other"
    bad_other = is_other & ~reasons.isin(OTHER_REASONS)
    if bad_other.any():
        ids = frame.loc[bad_other, "event_id_cnty"].head(10).tolist()
        raise ValueError(f"{context}: 'other' rows need a valid other_reason: {ids}")
    bad_non_other = ~is_other & ~reasons.isin(["", "nan"])
    if bad_non_other.any():
        ids = frame.loc[bad_non_other, "event_id_cnty"].head(10).tolist()
        raise ValueError(f"{context}: other_reason is only valid for 'other': {ids}")

    reject_empty_notes(frame, context=context)
    return frame


def assert_valid_label(label: str, other_reason: str | None = None) -> None:
    """Single-label validity, delegating to the shared taxonomy contract."""
    validate_label(label, other_reason)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _resolve(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path


def load_annotations(
    sources: tuple[tuple[str, str], ...] = ANNOTATION_SOURCES,
) -> pd.DataFrame:
    """Load every labelled annotation chunk in source order.

    Unlabelled rows (queues) are ignored; chunks may therefore exist before
    they are annotated.  Rows are deduplicated by ``event_id_cnty`` keeping the
    first occurrence, so release ordering stays deterministic.
    """
    frames: list[pd.DataFrame] = []
    order = 0
    for rank, (name, pattern) in enumerate(sources):
        paths = sorted(_resolve(pattern).parent.glob(Path(pattern).name))
        for path in paths:
            frame = pd.read_csv(path, low_memory=False)
            if "primary_label" not in frame.columns:
                continue
            frame = frame[frame["primary_label"].notna()].copy()
            if frame.empty:
                continue
            frame["notes"] = frame["notes"].map(normalize_notes)
            frame["note_hash"] = frame["notes"].map(sha256_text)
            frame["other_reason"] = frame.get("other_reason", "").fillna("").astype(str)
            frame = frame[frame["other_reason"].isin([*OTHER_REASONS, "", "nan"])].copy()
            frame["alternative_labels"] = frame.get("alternative_labels", "").fillna("").astype(str)
            frame["reviewer"] = frame.get("reviewer", f"{name}-{path.stem}").fillna("").astype(str)
            frame["_source_rank"] = rank
            frame["_source_order"] = order
            frame["_row_order"] = range(len(frame))
            order += 1
            frames.append(frame)
        # end chunks
    if not frames:
        raise FileNotFoundError("No labelled annotation chunks were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["_source_rank", "_source_order", "_row_order"], kind="stable")
    combined = combined.drop_duplicates(subset="event_id_cnty", keep="first").reset_index(drop=True)
    validate_annotations(combined, context="training annotations")
    return combined[list(RELEASE_COLUMNS)].copy()


def load_dev_497(dev_gold: Path = DEV_GOLD) -> pd.DataFrame:
    """Development gold with the three test-overlapping IDs removed (497 rows)."""
    frame = pd.read_csv(dev_gold, low_memory=False)
    frame = frame[~frame["event_id_cnty"].isin(DEV_REMOVE_IDS)].copy()
    frame["notes"] = frame["notes"].map(normalize_notes)
    frame["note_hash"] = frame["note_hash"].astype(str).str.strip()
    recomputed = frame["notes"].map(sha256_text)
    if not recomputed.eq(frame["note_hash"]).all():
        bad = frame.loc[~recomputed.eq(frame["note_hash"]), "event_id_cnty"].head(10).tolist()
        raise ValueError(f"Development note hashes no longer match the manifest: {bad}")
    frame["other_reason"] = frame.get("other_reason", "").fillna("").astype(str)
    frame["alternative_labels"] = frame.get("alternative_labels", "").fillna("").astype(str)
    frame["reviewer"] = frame.get("reviewer", "").fillna("").astype(str)
    validate_annotations(frame, context="development gold")
    return frame[list(RELEASE_COLUMNS)].reset_index(drop=True)


def load_test_set() -> pd.DataFrame:
    """The locked 840-row test set: notes from the manifest + gold labels.

    This is the *only* place test labels are read.  It is imported by
    ``evaluate.py --split test`` and by the leakage tests; training code never
    calls it.
    """
    manifest = pd.read_csv(TEST_MANIFEST_840, low_memory=False)
    gold = pd.read_csv(TEST_GOLD, low_memory=False)
    if manifest["event_id_cnty"].duplicated().any() or gold["event_id_cnty"].duplicated().any():
        raise ValueError("Test manifest and gold must have unique event IDs")
    merged = manifest[["event_id_cnty", "notes", "note_hash"]].merge(
        gold[["event_id_cnty", "note_hash", "primary_label", "alternative_labels", "other_reason"]],
        on="event_id_cnty",
        how="inner",
        suffixes=("_manifest", "_gold"),
        validate="one_to_one",
    )
    if len(merged) != len(manifest):
        raise ValueError(
            f"Test gold covers {len(merged)} of {len(manifest)} manifest rows"
        )
    if not merged["note_hash_manifest"].eq(merged["note_hash_gold"]).all():
        raise ValueError("Test note hashes do not match between manifest and gold")
    merged["note_hash"] = merged["note_hash_manifest"]
    merged["notes"] = merged["notes"].map(normalize_notes)
    merged["other_reason"] = merged.get("other_reason", pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["alternative_labels"] = merged.get(
        "alternative_labels", pd.Series("", index=merged.index)
    ).fillna("").astype(str)
    merged["reviewer"] = merged.get("reviewer", pd.Series("test-gold", index=merged.index)).fillna("").astype(str)
    merged = merged[list(RELEASE_COLUMNS)]
    validate_annotations(merged, context="test gold")
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #


def assert_disjoint(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    on: tuple[str, ...] = ("event_id_cnty", "note_hash"),
) -> None:
    """Raise if any ID or note hash occurs in both frames."""
    for column in on:
        if column not in left.columns or column not in right.columns:
            continue
        overlap = set(left[column].astype(str)) & set(right[column].astype(str))
        if overlap:
            sample = sorted(overlap)[:10]
            raise ValueError(
                f"Leakage: {left_name} and {right_name} share {len(overlap)} {column} "
                f"value(s): {sample}"
            )


def assert_nested(smaller: pd.DataFrame, larger: pd.DataFrame, *, smaller_name: str, larger_name: str) -> None:
    """Raise unless every row of ``smaller`` also occurs in ``larger``."""
    missing = set(smaller["event_id_cnty"]) - set(larger["event_id_cnty"])
    if missing:
        raise ValueError(
            f"{larger_name} is not a superset of {smaller_name}; missing {len(missing)} IDs"
        )
