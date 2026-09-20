"""Selection and validation of nested, disjoint annotation manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..text import normalize_notes, note_hash
from .contracts import MANIFEST_COLUMNS, assert_disjoint, assert_nested, validate_manifest

SOURCE_COLUMNS = ["event_id_cnty", "event_date", "year", "country", "notes"]


def load_event_pool(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column in SOURCE_COLUMNS, low_memory=False)
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Event source is missing columns: {sorted(missing)}")
    if frame.event_id_cnty.duplicated().any():
        raise ValueError("Event source contains duplicate event IDs")
    frame["notes"] = frame.notes.map(normalize_notes)
    frame = frame[frame.notes.ne("")].copy()
    frame["note_hash"] = frame.notes.map(note_hash)
    return frame.reset_index(drop=True)


def select_random(
    events: pd.DataFrame,
    *,
    seed: int = 42,
    dev_size: int = 500,
    test_size: int = 840,
    train_sizes: tuple[int, ...] = (800, 1500, 3000, 6000),
) -> dict[str, pd.DataFrame]:
    unique_notes = events.drop_duplicates("note_hash", keep="first").reset_index(drop=True)
    required = dev_size + test_size + max(train_sizes)
    if len(unique_notes) < required:
        raise ValueError(f"Need {required} unique notes, found {len(unique_notes)}")
    shuffled = unique_notes.iloc[
        np.random.default_rng(seed).permutation(len(unique_notes))
    ].reset_index(drop=True)
    test = shuffled.iloc[:test_size].copy()
    dev = shuffled.iloc[test_size:test_size + dev_size].copy()
    pool = shuffled.iloc[test_size + dev_size:test_size + dev_size + max(train_sizes)]
    result = {"test_locked": test, "dev": dev}
    result.update({f"train_{size:04d}": pool.iloc[:size].copy() for size in train_sizes})
    validate_splits(result, train_sizes=train_sizes)
    return result


def preserve_selections(
    events: pd.DataFrame,
    selection_files: dict[str, Path],
    *,
    train_sizes: tuple[int, ...] = (800, 1500, 3000, 6000),
) -> dict[str, pd.DataFrame]:
    """Recreate manifests from prior IDs while deliberately discarding labels."""
    indexed = events.set_index("event_id_cnty", drop=False)
    result: dict[str, pd.DataFrame] = {}
    for name, source in selection_files.items():
        ids = pd.read_csv(source, usecols=["event_id_cnty"], dtype=str).event_id_cnty
        if ids.duplicated().any():
            raise ValueError(f"{source} contains duplicate event IDs")
        missing = set(ids) - set(indexed.index)
        if missing:
            raise ValueError(f"{len(missing)} selected IDs are absent from the event pool")
        result[name] = indexed.loc[ids, list(MANIFEST_COLUMNS)].reset_index(drop=True)
    validate_splits(result, train_sizes=train_sizes)
    return result


def validate_splits(
    splits: dict[str, pd.DataFrame],
    *,
    train_sizes: tuple[int, ...] = (800, 1500, 3000, 6000),
) -> None:
    for name, frame in splits.items():
        validate_manifest(frame, context=name)
    largest = splits[f"train_{max(train_sizes):04d}"]
    assert_disjoint(largest, splits["dev"], left_name="train", right_name="dev")
    assert_disjoint(largest, splits["test_locked"], left_name="train", right_name="test")
    assert_disjoint(splits["dev"], splits["test_locked"], left_name="dev", right_name="test")
    for smaller, larger in zip(train_sizes, train_sizes[1:]):
        assert_nested(
            splits[f"train_{smaller:04d}"], splits[f"train_{larger:04d}"],
            smaller_name=f"train_{smaller}", larger_name=f"train_{larger}",
        )


def write_manifests(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[list(MANIFEST_COLUMNS)].to_csv(output_dir / f"{name}.csv", index=False)
