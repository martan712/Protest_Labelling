"""Score the frozen test predictions against the blind gold, stratified.

The 98% figure was agreement with the trigger rules that were still in the input;
the 77.2% figure was a hand-check graded non-blind. This script reports the first
honest numbers, separated by stratum and by training contamination, with intervals:

- keyword-matched and provably not in the model's training data: the clean 60%;
- keyword-matched and in the model's training data: the memorisation ceiling;
- unknown (no keyword fired): comparable to the legacy 55.0% / 68.5%;
- a corpus-weighted combined figure over clean rows only, with the stratum
  variances propagated into its interval.

The frozen `score_predictions.py` is left untouched; this is an additive companion.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy import accepted, ACCEPTED_PAIRS, TAXONOMY_VERSION  # noqa: E402

# Corpus composition of the ~keyword-matched / ~unknown split, used to weight the
# combined estimate rather than averaging over whichever rows survived filtering.
STRATUM_WEIGHTS = {"keyword": 0.612, "unknown": 0.388}


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (float("nan"), float("nan"))
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def score_frame(frame: pd.DataFrame, prediction_column: str) -> dict:
    n = len(frame)
    strict_flags = frame[prediction_column].eq(frame.primary_label)
    accepted_flags = [
        accepted(pred, primary, tuple(str(alt).split("|") if pd.notna(alt) else ()))
        for pred, primary, alt in zip(frame[prediction_column], frame.primary_label, frame.alternative_labels)
    ]
    result = {"n": n}
    for metric, flags in (("strict", strict_flags), ("accepted", accepted_flags)):
        correct = int(sum(flags)) if not isinstance(flags, pd.Series) else int(flags.sum())
        result[metric] = correct / n if n else float("nan")
        result[f"{metric}_95ci"] = list(wilson(correct, n))
        result[f"{metric}_correct"] = correct
    return result


def weighted_combined(clean: dict, unknown: dict) -> dict:
    """Weighted combined estimate with independent-stratum variance propagation."""
    wk, wu = STRATUM_WEIGHTS["keyword"], STRATUM_WEIGHTS["unknown"]
    out = {"weights": STRATUM_WEIGHTS, "n_keyword_clean": clean["n"], "n_unknown": unknown["n"]}
    for metric in ("strict", "accepted"):
        pk, pu = clean[metric], unknown[metric]
        nk, nu = clean["n"], unknown["n"]
        point = wk * pk + wu * pu
        var = wk ** 2 * (pk * (1 - pk) / nk if nk else 0) + wu ** 2 * (pu * (1 - pu) / nu if nu else 0)
        half = 1.96 * sqrt(var)
        out[metric] = point
        out[f"{metric}_95ci"] = [point - half, point + half]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/review/test_gold_v1/test_gold.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/manifests/test_manifest_840.csv"))
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--near-dup", type=Path, default=Path("data/review/test_gold_v1/near_dup_students.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--pred-column", default="predicted_class")
    args = parser.parse_args()

    gold = pd.read_csv(args.gold, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    flagged = pd.read_csv(args.flagged, keep_default_na=False)
    if set(gold.event_id_cnty) != set(manifest.event_id_cnty):
        raise ValueError("gold and manifest must cover identical IDs")

    strata = flagged.loc[
        flagged.scorable_all_models,
        ["event_id_cnty", "keyword_matched", "in_students_train", "in_release21_train"],
    ].copy()
    if args.near_dup.exists():
        near_dup = pd.read_csv(args.near_dup)
        strata = strata.merge(near_dup[["event_id_cnty", "near_dup_students_train"]], on="event_id_cnty", how="left")
    else:
        strata["near_dup_students_train"] = False
    strata["near_dup_students_train"] = strata.near_dup_students_train.fillna(False).astype(bool)

    models = {
        "test-students": {
            "predictions": Path("data/run/test_students_predictions.csv"),
            "train_column": "in_students_train",
            "apply_near_dup": True,
        },
        "test-release21": {
            "predictions": Path("data/run/test_release21_predictions.csv"),
            "train_column": "in_release21_train",
            "apply_near_dup": False,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run_name, config in models.items():
        pred = pd.read_csv(config["predictions"])
        frame = gold.merge(pred, on="event_id_cnty", validate="one_to_one")
        frame = frame.merge(strata, on="event_id_cnty", validate="one_to_one")
        frame["keyword"] = frame.keyword_matched.astype(bool)

        leak = frame[config["train_column"]].astype(bool)
        if config["apply_near_dup"]:
            leak = leak | frame.near_dup_students_train
        keyword_clean = frame[frame.keyword & ~leak]
        keyword_leaked = frame[frame.keyword & leak]
        unknown = frame[~frame.keyword]

        report = {
            "run": run_name,
            "taxonomy_version": TAXONOMY_VERSION,
            "accepted_pairs": sorted(map(sorted, ACCEPTED_PAIRS)),
            "manifest_sha256": sha256(args.manifest),
            "gold_sha256": sha256(args.gold),
            "n_total": int(len(frame)),
            "n_disputed": int(frame.disputed.astype(str).str.lower().eq("true").sum()),
            "keyword_clean_not_in_training": score_frame(keyword_clean, args.pred_column),
            "keyword_in_training": score_frame(keyword_leaked, args.pred_column),
            "unknown": score_frame(unknown, args.pred_column),
            "combined_clean_weighted": weighted_combined(
                score_frame(keyword_clean, args.pred_column), score_frame(unknown, args.pred_column)
            ),
            "all_rows_unweighted": score_frame(frame, args.pred_column),
        }

        # Coverage gap: classes with no clean keyword rows for this model.
        clean_classes = set(keyword_clean.primary_label)
        all_classes = set(frame.primary_label)
        report["keyword_clean_classes_with_zero_rows"] = sorted(all_classes - clean_classes)

        (args.out_dir / f"{run_name}_strata.json").write_text(json.dumps(report, indent=2) + "\n")

        clean_rows = pd.concat([keyword_clean.assign(stratum="keyword_clean"), unknown.assign(stratum="unknown")])
        clean_rows["strict_flag"] = clean_rows[args.pred_column].eq(clean_rows.primary_label)
        clean_rows["accepted_flag"] = [
            accepted(p, prim, tuple(str(alt).split("|")))
            for p, prim, alt in zip(clean_rows[args.pred_column], clean_rows.primary_label, clean_rows.alternative_labels)
        ]
        by_topic = (
            clean_rows.groupby("primary_label")
            .agg(
                n=("event_id_cnty", "size"),
                n_keyword_clean=("stratum", lambda values: int(values.eq("keyword_clean").sum())),
                n_unknown=("stratum", lambda values: int(values.eq("unknown").sum())),
                strict=("strict_flag", "mean"),
                accepted=("accepted_flag", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        by_topic.to_csv(args.out_dir / f"{run_name}_strata_by_topic.csv", index=False)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
