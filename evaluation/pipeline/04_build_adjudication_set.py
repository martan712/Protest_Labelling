"""Select the stratified second-pass adjudication sample and split it into chunks.

The second pass exists to measure the annotator's own error rate. Rows are drawn
with a fixed seed, stratified on the keyword/unknown split at the corpus ratio, and
written to a directory that contains neither the first-pass labels nor the stratum
flags, so the second annotator set cannot see the first set's answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def allocate_stratified(flag: pd.Series, total: int) -> dict[object, int]:
    counts = flag.value_counts()
    raw = counts / counts.sum() * total
    alloc = np.floor(raw).astype(int)
    remainder = total - int(alloc.sum())
    if remainder:
        # Positional top-up, so boolean stratum labels are not read as masks.
        positions = np.argsort(-(raw - alloc).to_numpy(), kind="stable")[:remainder]
        alloc.iloc[positions] += 1
    return alloc.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("data/review/test_gold_v1/annotation_queue.csv"))
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/review/test_gold_v1/adjudication"))
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=30)
    args = parser.parse_args()

    queue = pd.read_csv(args.queue, keep_default_na=False)
    flagged = pd.read_csv(args.flagged, keep_default_na=False)
    stratum = (
        flagged.loc[flagged.scorable_all_models, ["event_id_cnty", "keyword_matched"]]
        .set_index("event_id_cnty")
        .keyword_matched
    )
    queue = queue.assign(keyword_matched=queue.event_id_cnty.map(stratum))
    if queue.keyword_matched.isna().any():
        raise ValueError("some queue rows are missing a stratum flag")

    rng = np.random.default_rng(args.seed)
    alloc = allocate_stratified(queue.keyword_matched, args.size)
    picks = []
    for value, n in alloc.items():
        pool = queue[queue.keyword_matched.eq(value)]
        order = rng.permutation(len(pool))
        picks.append(pool.iloc[order].head(int(n)))
    selected = pd.concat(picks, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["annotation_order"] = np.arange(1, len(selected) + 1)
    for column in ("primary_label", "alternative_labels", "other_reason", "evidence"):
        selected[column] = ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected[QUEUE_COLUMNS].to_csv(args.out_dir / "selection.csv", index=False)

    chunk_count = 0
    for start in range(0, len(selected), args.chunk_size):
        chunk_count += 1
        selected.iloc[start:start + args.chunk_size][QUEUE_COLUMNS].to_csv(
            args.out_dir / f"chunk_{chunk_count:02d}.csv", index=False
        )

    metadata = {
        "size": int(len(selected)),
        "seed": args.seed,
        "allocation": {str(k): int(v) for k, v in alloc.items()},
        "chunk_size": args.chunk_size,
        "chunk_count": chunk_count,
    }
    (args.out_dir / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
