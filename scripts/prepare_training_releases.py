#!/usr/bin/env python3
"""Assemble all complete nested training releases from annotation chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protest_classifier.data.annotations import assemble_training_releases, read_completed_chunks

ROOT = Path(__file__).resolve().parents[1]
SIZES = (800, 1500, 3000, 6000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/annotations/train_chunks")
    parser.add_argument("--output", type=Path, default=ROOT / "data/annotations")
    args = parser.parse_args()
    manifests = {
        size: pd.read_csv(args.manifests / f"train_{size:04d}.csv", dtype=str).fillna("")
        for size in SIZES
    }
    labels = read_completed_chunks(sorted(args.chunks.glob("chunk_*.csv")))
    dev = pd.read_csv(args.manifests / "dev.csv", dtype=str).fillna("")
    test = pd.read_csv(args.manifests / "test_locked.csv", dtype=str).fillna("")
    releases = assemble_training_releases(manifests, labels, dev, test)
    args.output.mkdir(parents=True, exist_ok=True)
    for size, frame in releases.items():
        path = args.output / f"train_{size:04d}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {len(frame)} rows to {path}")
    missing = sorted(set(SIZES) - set(releases))
    if missing:
        print(f"not yet complete: {missing}")


if __name__ == "__main__":
    main()
