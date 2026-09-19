"""Build the blind annotation queue for the frozen test set.

Reads the flagged frozen manifest, keeps the rows scorable by every compared
model, and writes a shuffled queue plus fixed-size chunks. The queue carries no
keyword, train-membership or prediction columns: the annotator must see the
note text and nothing else.

The shuffle is seeded so keyword-matched and unknown rows interleave; otherwise
an annotator could infer the stratum from the texture of the block and calibrate
differently across the two, silently biasing the comparison this set exists to
make.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy import TAXONOMY_VERSION  # noqa: E402

QUEUE_COLUMNS = [
    "annotation_order",
    "event_id_cnty",
    "event_date",
    "year",
    "country",
    "notes",
    "note_hash",
    "primary_label",
    "alternative_labels",
    "other_reason",
    "evidence",
]

FORBIDDEN_COLUMNS = [
    "rule_label",
    "keyword_matched",
    "class",
    "in_students_train",
    "in_release21_train",
    "pred_students",
    "pred_release21",
    "clean_notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/review/test_gold_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=60)
    args = parser.parse_args()

    source_sha = sha256(args.flagged)
    frame = pd.read_csv(args.flagged, keep_default_na=False)
    scorable = frame[frame.scorable_all_models].copy()

    source_columns = ["event_id_cnty", "event_date", "year", "country", "notes", "note_hash"]
    missing = [column for column in source_columns if column not in scorable.columns]
    if missing:
        raise KeyError(f"source manifest missing queue columns: {missing}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(scorable))
    scorable = scorable.iloc[order].reset_index(drop=True)
    scorable.insert(0, "annotation_order", np.arange(1, len(scorable) + 1))
    for column in ("primary_label", "alternative_labels", "other_reason", "evidence"):
        scorable[column] = ""

    queue = scorable[QUEUE_COLUMNS]
    leaked = [column for column in FORBIDDEN_COLUMNS if column in queue.columns]
    if leaked:
        raise ValueError(f"forbidden columns leaked into queue: {leaked}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    adjudication_dir = args.out_dir / "adjudication"
    adjudication_dir.mkdir(parents=True, exist_ok=True)

    queue.to_csv(args.out_dir / "annotation_queue.csv", index=False)

    chunk_count = 0
    for start in range(0, len(queue), args.chunk_size):
        chunk_count += 1
        chunk = queue.iloc[start:start + args.chunk_size]
        chunk.to_csv(chunk_dir / f"chunk_{chunk_count:02d}.csv", index=False)

    strata = scorable.assign(
        stratum=scorable.keyword_matched.map({True: "keyword", False: "unknown"})
    ).groupby("stratum").size().to_dict()

    metadata = {
        "queue_version": "test-gold-v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "source_manifest": str(args.flagged),
        "source_manifest_sha256": source_sha,
        "seed": args.seed,
        "rows": int(len(queue)),
        "strata": {k: int(v) for k, v in strata.items()},
        "chunk_size": args.chunk_size,
        "chunk_count": chunk_count,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
