"""Score predictions with the frozen taxonomy and produce comparable run reports."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import accepted, ACCEPTED_PAIRS, TAXONOMY_VERSION


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (float("nan"), float("nan"))
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def score(frame: pd.DataFrame, prediction_column: str) -> dict:
    frame = frame.dropna(subset=["manual_label", prediction_column]).copy()
    frame["strict"] = frame[prediction_column].eq(frame.manual_label)
    frame["accepted"] = [
        accepted(pred, primary, tuple(str(alt).split("|") if pd.notna(alt) else ()))
        for pred, primary, alt in zip(frame[prediction_column], frame.manual_label, frame.manual_label_alt)
    ]
    result = {"n": len(frame)}
    for metric in ("strict", "accepted"):
        correct = int(frame[metric].sum())
        result[metric] = correct / len(frame) if len(frame) else float("nan")
        result[f"{metric}_95ci"] = wilson(correct, len(frame))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manual", type=Path, default=Path("data/manual_labelled_data/random_unknown_labeled.csv"))
    parser.add_argument("--prediction-column", default="predicted_class")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)
    manual = pd.read_csv(args.manual, usecols=["event_id_cnty", "manual_label", "manual_label_alt"])
    joined = manual.merge(pred, on="event_id_cnty", how="inner", validate="one_to_one")
    if len(joined) != len(manual):
        raise ValueError("Predictions do not cover the complete locked development set")
    joined["strict"] = joined[args.prediction_column].eq(joined.manual_label)
    joined["accepted"] = [
        accepted(pred, primary, tuple(str(alt).split("|") if pd.notna(alt) else ()))
        for pred, primary, alt in zip(joined[args.prediction_column], joined.manual_label, joined.manual_label_alt)
    ]
    metrics = score(joined, args.prediction_column)
    metrics.update({"run": args.run_name, "taxonomy_version": TAXONOMY_VERSION, "accepted_pairs": sorted(map(sorted, ACCEPTED_PAIRS))})
    (args.out / f"{args.run_name}.json").write_text(json.dumps(metrics, indent=2, default=list) + "\n")
    joined["year"] = pd.to_datetime(joined.get("event_date"), errors="coerce").dt.year if "event_date" in joined else pd.NA
    joined["error"] = ~joined.accepted
    joined.groupby("manual_label").agg(n=("event_id_cnty", "size"), strict=("strict", "mean"), accepted=("accepted", "mean")).reset_index().to_csv(args.out / f"{args.run_name}_by_topic.csv", index=False)
    print(json.dumps(metrics, indent=2, default=list))


if __name__ == "__main__":
    main()
