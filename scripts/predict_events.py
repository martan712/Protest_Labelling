#!/usr/bin/env python3
"""Stream full-corpus predictions from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from protest_classifier.modeling.inference import predict_csv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=ROOT / "data/raw/events.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/predictions/all_events_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--read-chunk-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    total = predict_csv(
        args.model, args.events, args.output,
        batch_size=args.batch_size, read_chunk_size=args.read_chunk_size,
        max_length=args.max_length, limit=args.limit,
        progress=lambda count: print(f"predicted {count} rows", flush=True),
    )
    print(f"wrote {total} predictions to {args.output}")


if __name__ == "__main__":
    main()
