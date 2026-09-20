#!/usr/bin/env python3
"""Turn a frozen manifest into blank annotation chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protest_classifier.data.annotations import write_annotation_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str).fillna("")
    paths = write_annotation_chunks(manifest, args.output, chunk_size=args.chunk_size)
    print(f"wrote {len(paths)} chunks to {args.output}")


if __name__ == "__main__":
    main()
