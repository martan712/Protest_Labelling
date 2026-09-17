"""Freeze development/test event IDs before any label or rule changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_counts(frame: pd.DataFrame, n: int, columns: list[str]) -> pd.Series:
    """Allocate n proportionally and deterministically over a hierarchy."""
    groups = frame.groupby(columns, dropna=False).size().rename("available")
    raw = groups / groups.sum() * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder:
        order = (raw - counts).sort_values(ascending=False, kind="stable").index[:remainder]
        counts.loc[order] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("data/filtered_events.csv"))
    parser.add_argument("--legacy-dev", type=Path, default=Path("data/manual_labelled_data/random_unknown_labeled.csv"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/manifests"))
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events, usecols=["event_id_cnty", "event_date", "year", "country", "notes", "clean_notes"])
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    if events["event_date"].isna().any():
        raise ValueError("Cannot freeze evaluation data with missing event dates")
    events["year"] = events["event_date"].dt.year.astype(int)
    events["note_hash"] = events["notes"].fillna("").map(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    events["duplicate_group_id"] = "note-" + events["note_hash"].str[:16]
    if events["event_id_cnty"].duplicated().any():
        raise ValueError("Event IDs are not unique")

    legacy = pd.read_csv(args.legacy_dev, usecols=["event_id_cnty", "manual_label", "manual_label_alt"])
    dev = events.merge(legacy, on="event_id_cnty", how="inner", validate="one_to_one")
    if len(dev) != len(legacy):
        missing = sorted(set(legacy.event_id_cnty) - set(dev.event_id_cnty))
        raise ValueError(f"Legacy dev IDs missing from snapshot: {missing[:5]}")
    held_groups = set(dev.duplicate_group_id)
    eligible = events[~events.event_id_cnty.isin(dev.event_id_cnty) & ~events.duplicate_group_id.isin(held_groups)].copy()
    quotas = allocate_counts(eligible, args.test_size, ["year", "country"])
    rng = np.random.default_rng(args.seed)
    selected = []
    for key, quota in quotas.items():
        if quota == 0:
            continue
        mask = np.ones(len(eligible), dtype=bool)
        for col, value in zip(["year", "country"], key):
            mask &= eligible[col].eq(value).to_numpy()
        candidates = eligible.loc[mask]
        selected.append(candidates.iloc[rng.choice(len(candidates), size=int(quota), replace=False)])
    test = pd.concat(selected, ignore_index=True).sort_values(["year", "country", "event_id_cnty"])
    test.insert(0, "split", "test")
    dev.insert(0, "split", "dev")
    columns = ["split", "event_id_cnty", "event_date", "year", "country", "notes", "clean_notes", "note_hash", "duplicate_group_id"]
    dev[columns].to_csv(args.out / "dev_manifest.csv", index=False)
    test[columns].to_csv(args.out / "test_manifest.csv", index=False)
    pd.concat([dev[columns], test[columns]], ignore_index=True).to_csv(args.out / "heldout_manifest.csv", index=False)
    test.assign(primary_label="", alternative_labels="", other_reason="", annotation_status="unannotated")[
        columns + ["primary_label", "alternative_labels", "other_reason", "annotation_status"]
    ].to_csv(args.out / "test_annotation_queue.csv", index=False)
    metadata = {
        "snapshot": str(args.events), "snapshot_sha256": sha256(args.events),
        "legacy_dev": str(args.legacy_dev), "legacy_dev_sha256": sha256(args.legacy_dev),
        "seed": args.seed, "test_size_requested": args.test_size,
        "dev_rows": len(dev), "test_rows": len(test),
        "excluded_duplicate_groups": len(held_groups), "taxonomy_version": "21-class-v1",
    }
    (args.out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
