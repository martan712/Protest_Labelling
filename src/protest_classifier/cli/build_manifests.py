#!/usr/bin/env python3
"""Build active train/dev/test manifests from label-free event selections."""

from __future__ import annotations

import argparse
from pathlib import Path

from protest_classifier.data.manifests import (
    load_event_pool,
    preserve_selections,
    select_random,
    write_manifests,
)

ROOT = Path(__file__).resolve().parents[3]
SIZES = (800, 1500, 3000, 6000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=ROOT / "data/raw/events.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--fresh-random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    events = load_event_pool(args.events)
    if args.fresh_random:
        splits = select_random(events, seed=args.seed, dev_size=497, test_size=840, train_sizes=SIZES)
    else:
        sources = {
            **{
                f"train_{size:04d}": args.output / f"train_{size:04d}.csv"
                for size in SIZES
            },
            "dev": args.output / "dev.csv",
            "test_locked": args.output / "test_locked.csv",
        }
        splits = preserve_selections(events, sources, train_sizes=SIZES)
    write_manifests(splits, args.output)
    for name, frame in splits.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
