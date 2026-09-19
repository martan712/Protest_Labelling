"""Assemble the frozen test gold set, its manifest and per-model predictions.

Pass 1 supplies the labels. Rows on which the independent second pass disagreed are
kept as pass 1 but flagged `disputed`, so the disagreement travels with the data
rather than being silently adjudicated away by a third model.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy import CLASS_NAMES, TAXONOMY_VERSION, validate_label  # noqa: E402

GOLD_COLUMNS = [
    "event_id_cnty",
    "note_hash",
    "primary_label",
    "alternative_labels",
    "other_reason",
    "evidence",
    "reviewer",
    "disputed",
    "taxonomy_version",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", default="data/review/test_gold_v1/annotated/chunk_*.csv")
    parser.add_argument("--pass2", default="data/review/test_gold_v1/adjudication/second_pass_chunk_*.csv")
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--frozen", type=Path, default=Path("evaluation/manifests/test_manifest.csv"))
    parser.add_argument("--manifest-out", type=Path, default=Path("evaluation/manifests/test_manifest_840.csv"))
    parser.add_argument("--gold-out", type=Path, default=Path("data/review/test_gold_v1/test_gold.csv"))
    parser.add_argument("--pred-out-dir", type=Path, default=Path("data/run"))
    parser.add_argument("--near-dup", type=Path, default=Path("data/review/test_gold_v1/near_dup_students.csv"))
    args = parser.parse_args()

    gold = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in sorted(glob.glob(args.pass1))],
        ignore_index=True,
    )
    gold = gold.sort_values("annotation_order").reset_index(drop=True)

    pass2 = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in sorted(glob.glob(args.pass2))],
        ignore_index=True,
    )
    pass2 = pass2[["event_id_cnty", "primary_label"]].rename(columns={"primary_label": "pass2"})
    compare = gold[["event_id_cnty", "primary_label"]].merge(pass2, on="event_id_cnty", how="left")
    disputed_ids = set(compare.loc[compare.pass2.notna() & compare.primary_label.ne(compare.pass2), "event_id_cnty"])
    gold["disputed"] = gold.event_id_cnty.isin(disputed_ids)
    gold["taxonomy_version"] = TAXONOMY_VERSION

    frozen = pd.read_csv(args.frozen, keep_default_na=False)
    flagged = pd.read_csv(args.flagged, keep_default_na=False)
    scorable = flagged[flagged.scorable_all_models].copy()
    manifest_840 = frozen[frozen.event_id_cnty.isin(scorable.event_id_cnty)].copy()

    # Validation, in the same spirit as the frozen scorer.
    assert len(gold) == 840, f"expected 840 gold rows, got {len(gold)}"
    if gold.event_id_cnty.duplicated().any():
        raise ValueError("duplicate gold event IDs")
    if set(gold.event_id_cnty) != set(scorable.event_id_cnty):
        raise ValueError("gold IDs do not equal the scorable set")
    merged_hash = gold[["event_id_cnty", "note_hash"]].merge(
        scorable[["event_id_cnty", "note_hash"]], on="event_id_cnty", suffixes=("_gold", "_flag")
    )
    if not merged_hash.note_hash_gold.eq(merged_hash.note_hash_flag).all():
        raise ValueError("gold note hashes differ from the manifest")
    for _, row in gold.iterrows():
        validate_label(row.primary_label, row.other_reason)
        for alt in str(row.alternative_labels).split("|"):
            if alt.strip():
                validate_label(alt)
        if not str(row.primary_label).strip() and not str(row.evidence).strip():
            raise ValueError(f"row {row.event_id_cnty} has neither label nor evidence")

    manifest_840.to_csv(args.manifest_out, index=False)
    gold[GOLD_COLUMNS].to_csv(args.gold_out, index=False)

    # Per-model predictions restricted to the 840 frozen IDs.
    args.pred_out_dir.mkdir(parents=True, exist_ok=True)
    predictions_written = {}
    for column, run_name in (("pred_students", "test_students"), ("pred_release21", "test_release21")):
        predictions = scorable[["event_id_cnty", column]].rename(columns={column: "predicted_class"})
        if predictions.predicted_class.isna().any():
            raise ValueError(f"{column} has missing predictions on the scorable set")
        path = args.pred_out_dir / f"{run_name}_predictions.csv"
        predictions.to_csv(path, index=False)
        predictions_written[run_name] = str(path)

    near_dup = pd.read_csv(args.near_dup) if args.near_dup.exists() else pd.DataFrame(
        columns=["event_id_cnty", "near_dup_students_train", "near_dup_jaccard"]
    )

    print(f"gold rows: {len(gold)}  disputed: {int(gold.disputed.sum())}")
    print(f"manifest: {args.manifest_out}  gold: {args.gold_out}")
    print(f"predictions: {predictions_written}")
    print(f"near-dup rows flagged: {int(near_dup.near_dup_students_train.sum())}")


if __name__ == "__main__":
    main()
