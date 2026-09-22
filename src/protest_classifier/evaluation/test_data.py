"""The sole loader for separately stored locked-test labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.contracts import ANNOTATED_COLUMNS, validate_labels, validate_manifest


def load_locked_test(manifest_path: Path, gold_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    gold = pd.read_csv(gold_path, dtype=str).fillna("")
    validate_manifest(manifest, context="locked test manifest")
    validate_labels(gold, context="locked test gold")
    if set(gold.event_id_cnty) != set(manifest.event_id_cnty):
        raise ValueError("Locked test gold must exactly cover the manifest IDs")
    joined = manifest.merge(
        gold[[column for column in gold.columns if column in ANNOTATED_COLUMNS and column != "notes"]],
        on="event_id_cnty", how="inner", validate="one_to_one", suffixes=("", "_gold"),
    )
    if "note_hash_gold" in joined and not joined.note_hash.eq(joined.note_hash_gold).all():
        raise ValueError("Locked test gold note hashes do not match the manifest")
    return joined[list(ANNOTATED_COLUMNS)].copy()
