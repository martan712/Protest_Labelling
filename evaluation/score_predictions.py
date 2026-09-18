"""Score predictions with the frozen taxonomy and produce comparable run reports."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from taxonomy import accepted, ACCEPTED_PAIRS, TAXONOMY_VERSION, validate_label


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


def score(frame: pd.DataFrame, prediction_column: str) -> dict:
    frame = frame.dropna(subset=["primary_label", prediction_column]).copy()
    frame["strict"] = frame[prediction_column].eq(frame.primary_label)
    frame["accepted"] = [
        accepted(pred, primary, tuple(str(alt).split("|") if pd.notna(alt) else ()))
        for pred, primary, alt in zip(frame[prediction_column], frame.primary_label, frame.alternative_labels)
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
    parser.add_argument("--gold", type=Path, default=Path("data/run/dev_gold.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/run/dev_manifest.csv"))
    parser.add_argument("--prediction-column", default="predicted_class")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)
    gold = pd.read_csv(args.gold, keep_default_na=False)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    if gold.event_id_cnty.duplicated().any() or pred.event_id_cnty.duplicated().any():
        raise ValueError("Gold and predictions must have unique event IDs")
    expected = set(manifest.event_id_cnty)
    if set(gold.event_id_cnty) != expected or set(pred.event_id_cnty) != expected:
        raise ValueError("Gold and predictions must exactly cover the frozen manifest IDs")
    frozen = manifest[["event_id_cnty", "note_hash"]].merge(
        gold[["event_id_cnty", "note_hash"]], on="event_id_cnty", suffixes=("_manifest", "_gold"), validate="one_to_one"
    )
    if not frozen.note_hash_manifest.eq(frozen.note_hash_gold).all():
        raise ValueError("Gold note hashes do not match the frozen manifest")
    for _, row in gold.iterrows():
        validate_label(row.primary_label, row.other_reason)
    joined = gold.merge(pred, on="event_id_cnty", how="inner", validate="one_to_one")
    joined["strict"] = joined[args.prediction_column].eq(joined.primary_label)
    joined["accepted"] = [
        accepted(pred, primary, tuple(str(alt).split("|") if pd.notna(alt) else ()))
        for pred, primary, alt in zip(joined[args.prediction_column], joined.primary_label, joined.alternative_labels)
    ]
    metrics = score(joined, args.prediction_column)
    metrics.update({"run": args.run_name, "taxonomy_version": TAXONOMY_VERSION,
                    "accepted_pairs": sorted(map(sorted, ACCEPTED_PAIRS)),
                    "manifest_sha256": sha256(args.manifest), "gold_sha256": sha256(args.gold)})
    (args.out / f"{args.run_name}.json").write_text(json.dumps(metrics, indent=2, default=list) + "\n")
    joined["year"] = pd.to_datetime(joined.get("event_date"), errors="coerce").dt.year if "event_date" in joined else pd.NA
    joined["error"] = ~joined.accepted
    joined.groupby("primary_label").agg(n=("event_id_cnty", "size"), strict=("strict", "mean"), accepted=("accepted", "mean")).reset_index().to_csv(args.out / f"{args.run_name}_by_topic.csv", index=False)
    print(json.dumps(metrics, indent=2, default=list))


if __name__ == "__main__":
    main()
