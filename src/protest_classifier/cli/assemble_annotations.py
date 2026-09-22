#!/usr/bin/env python3
"""Validate annotation labels and join them to one frozen manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protest_classifier.data.annotations import assemble, read_completed_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--labels", type=Path)
    inputs.add_argument("--chunks", type=Path, help="Directory containing chunk_*.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str).fillna("")
    labels = (
        pd.read_csv(args.labels, dtype=str).fillna("")
        if args.labels else read_completed_chunks(sorted(args.chunks.glob("chunk_*.csv")))
    )
    result = assemble(manifest, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
