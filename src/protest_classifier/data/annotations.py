"""Annotation queues and assembly against frozen manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..taxonomy import TAXONOMY_VERSION
from .contracts import (
    ANNOTATED_COLUMNS,
    LABEL_COLUMNS,
    assert_disjoint,
    assert_nested,
    validate_labels,
    validate_manifest,
)


def write_annotation_chunks(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    chunk_size: int = 100,
) -> list[Path]:
    validate_manifest(manifest)
    queue = manifest[["event_id_cnty", "notes"]].copy()
    for column in LABEL_COLUMNS[1:]:
        queue[column] = ""
    queue["taxonomy_version"] = TAXONOMY_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, start in enumerate(range(0, len(queue), chunk_size), 1):
        path = output_dir / f"chunk_{number:03d}.csv"
        queue.iloc[start:start + chunk_size].to_csv(path, index=False)
        paths.append(path)
    return paths


def read_completed_chunks(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "primary_label" in frame:
            frame = frame[frame.primary_label.ne("")].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No completed annotation rows found")
    labels = pd.concat(frames, ignore_index=True)
    validate_labels(labels, context="annotation chunks")
    return labels


def assemble(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    validate_manifest(manifest)
    validate_labels(labels)
    unexpected = set(labels.event_id_cnty) - set(manifest.event_id_cnty)
    if unexpected:
        raise ValueError(f"Labels contain {len(unexpected)} IDs outside the manifest")
    joined = manifest.merge(labels[list(LABEL_COLUMNS)], on="event_id_cnty", how="left", validate="one_to_one")
    if joined.primary_label.isna().any() or joined.primary_label.eq("").any():
        missing = int(joined.primary_label.isna().sum() + joined.primary_label.eq("").sum())
        raise ValueError(f"Manifest is missing {missing} annotations")
    return joined[list(ANNOTATED_COLUMNS)].copy()


def assemble_training_releases(
    manifests: dict[int, pd.DataFrame],
    labels: pd.DataFrame,
    dev_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Assemble every complete release and enforce nesting/leakage invariants."""
    assert_disjoint(dev_manifest, test_manifest, left_name="development", right_name="test")
    releases: dict[int, pd.DataFrame] = {}
    previous: tuple[int, pd.DataFrame] | None = None
    for size, manifest in sorted(manifests.items()):
        if len(manifest) != size:
            raise ValueError(f"train_{size:04d} has {len(manifest)} rows")
        assert_disjoint(manifest, dev_manifest, left_name=f"train_{size}", right_name="dev")
        assert_disjoint(manifest, test_manifest, left_name=f"train_{size}", right_name="test")
        if previous:
            assert_nested(
                previous[1], manifest,
                smaller_name=f"train_{previous[0]}", larger_name=f"train_{size}",
            )
        selected = labels[labels.event_id_cnty.isin(manifest.event_id_cnty)]
        if len(selected) == size:
            releases[size] = assemble(manifest, selected)
        previous = (size, manifest)
    return releases
