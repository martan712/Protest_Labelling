"""Flag the frozen test manifest with train-contamination and keyword strata.

The manifest itself stays frozen. This script only annotates it, so that every
model is scored on one identical row set and contamination is reported per
model instead of being removed from the data.

Rows are restricted to the oldest snapshot any compared model was built on
(by default the students' snapshot, which ends 2025-05-23), because a model
cannot be scored on events that postdate its own data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODELS = {
    "students": Path("data/filtered_events_class_with_predicted_students.csv"),
    "release21": Path("data/filtered_events_class_with_predicted.csv"),
}

TRAIN_SETS = {
    "students": Path("data/labeled_balanced_20.csv"),
    "release21": Path("data/release/labeled_balanced_21.csv"),
}


def note_column(frame: pd.DataFrame) -> str:
    if "clean_notes" in frame.columns:
        return "clean_notes"
    raise KeyError("no clean_notes column to match on")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/manifests/test_manifest.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/labeled.csv"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("evaluation/manifests/flag_metadata.json"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["event_date"] = pd.to_datetime(manifest["event_date"], errors="coerce")
    frame = manifest.copy()

    # 1. Restrict to the window every compared model actually covers.
    cutoffs = {}
    for name, path in MODELS.items():
        dates = pd.read_csv(path, usecols=["event_date"])["event_date"]
        cutoffs[name] = pd.to_datetime(dates, format="mixed", errors="coerce").max()
    cutoff = min(cutoffs.values())
    frame = frame[frame.event_date <= cutoff].copy()

    # 2. Keyword stratum: unknown means no trigger rule fired, nothing more.
    rules = pd.read_csv(args.labels, usecols=["event_id_cnty", "class"]).rename(columns={"class": "rule_label"})
    frame = frame.merge(rules, on="event_id_cnty", how="left")
    frame["keyword_matched"] = frame.rule_label.notna() & frame.rule_label.ne("unknown")

    # 3. Per-model contamination and predictions.
    for name, train_path in TRAIN_SETS.items():
        train = pd.read_csv(train_path, low_memory=False)
        notes = set(train[note_column(train)].dropna())
        frame[f"in_{name}_train"] = frame.clean_notes.isin(notes)

    for name, path in MODELS.items():
        preds = pd.read_csv(path, usecols=["event_id_cnty", "predicted_class"])
        preds = preds.drop_duplicates("event_id_cnty").rename(columns={"predicted_class": f"pred_{name}"})
        frame = frame.merge(preds, on="event_id_cnty", how="left")

    pred_columns = [f"pred_{name}" for name in MODELS]
    frame["scorable_all_models"] = frame[pred_columns].notna().all(axis=1)

    frame["gold_primary_label"] = pd.NA
    frame["gold_alternative_labels"] = pd.NA

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    scorable = frame[frame.scorable_all_models]
    print(f"cutoff applied: {cutoff.date()}  ({dict((k, str(v.date())) for k, v in cutoffs.items())})")
    print(f"manifest {len(manifest)} -> in window {len(frame)} -> scorable by all models {len(scorable)}\n")

    strata = (
        scorable.assign(stratum=scorable.keyword_matched.map({True: "keyword", False: "unknown"}))
        .groupby("stratum")
        .agg(n=("event_id_cnty", "size"),
             in_students_train=("in_students_train", "sum"),
             in_release21_train=("in_release21_train", "sum"))
    )
    print(strata.to_string())
    print(f"\ngold labels present: {scorable.gold_primary_label.notna().sum()} / {len(scorable)}")

    args.metadata.write_text(json.dumps({
        "cutoff": str(cutoff.date()),
        "model_cutoffs": {k: str(v.date()) for k, v in cutoffs.items()},
        "manifest_rows": int(len(manifest)),
        "in_window_rows": int(len(frame)),
        "scorable_rows": int(len(scorable)),
        "strata": strata.to_dict(orient="index"),
        "gold_annotated": int(scorable.gold_primary_label.notna().sum()),
    }, indent=2, default=int) + "\n")


if __name__ == "__main__":
    main()
