"""Measure agreement between the two independent blind annotation passes.

An LLM annotation set with no measured error rate is not ground truth. This script
compares pass 1 (the main run) and pass 2 (an independent re-annotation of a
stratified 150-row sample) on the primary label only, overall and by stratum, and
writes the confusion structure plus the uncertainty band that downstream scores
must carry.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from math import sqrt

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (float("nan"), float("nan"))
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return centre - margin, centre + margin


def load(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, keep_default_na=False) for path in sorted(glob.glob(pattern))]
    return pd.concat(frames, ignore_index=True)


def kappa(a: pd.Series, b: pd.Series) -> float:
    if len(a) == 0 or a.nunique() + b.nunique() < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def summarise(frame: pd.DataFrame, label: str) -> dict:
    n = len(frame)
    agree = int(frame.pass1.eq(frame.pass2).sum())
    lo, hi = wilson(agree, n)
    return {
        "n": n,
        "agreement": agree / n if n else float("nan"),
        "agreement_95ci": [lo, hi],
        "disagreement": 1 - agree / n if n else float("nan"),
        "cohen_kappa": kappa(frame.pass1, frame.pass2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", default="data/review/test_gold_v1/annotated/chunk_*.csv")
    parser.add_argument("--pass2", default="data/review/test_gold_v1/adjudication/second_pass_chunk_*.csv")
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/annotation_quality.json"))
    args = parser.parse_args()

    first = load(args.pass1)[["event_id_cnty", "primary_label", "alternative_labels"]].rename(
        columns={"primary_label": "pass1", "alternative_labels": "pass1_alt"}
    )
    second = load(args.pass2)[["event_id_cnty", "primary_label"]].rename(columns={"primary_label": "pass2"})
    if second.event_id_cnty.duplicated().any():
        raise ValueError("pass 2 contains duplicate event IDs")
    merged = first.merge(second, on="event_id_cnty", how="inner", validate="one_to_one")
    if len(merged) != len(second):
        raise ValueError("pass 2 covers rows absent from pass 1")

    flagged = pd.read_csv(args.flagged, keep_default_na=False)
    stratum = (
        flagged.loc[flagged.scorable_all_models, ["event_id_cnty", "keyword_matched"]]
        .set_index("event_id_cnty")
        .keyword_matched
    )
    merged["stratum"] = merged.event_id_cnty.map(stratum).map({True: "keyword", False: "unknown"})

    overall = summarise(merged, "overall")
    by_stratum = {name: summarise(group, name) for name, group in merged.groupby("stratum")}

    disagreements = merged[~merged.pass1.eq(merged.pass2)]
    confusion = (
        disagreements.groupby(["pass1", "pass2"]).size().sort_values(ascending=False)
    )
    confusion_records = [
        {"pass1": a, "pass2": b, "n": int(n)} for (a, b), n in confusion.items()
    ]

    # Uncertainty band: two independent blind passes disagree on `rate` of rows, so a
    # downstream score inherits at least that much label noise. Report the overall rate
    # and its Wilson interval, plus the worse of the two strata.
    band = {
        "label_disagreement_rate": overall["disagreement"],
        "label_disagreement_95ci": [1 - overall["agreement_95ci"][1], 1 - overall["agreement_95ci"][0]],
        "worst_stratum_disagreement": max(v["disagreement"] for v in by_stratum.values()),
        "worst_stratum": max(by_stratum, key=lambda k: by_stratum[k]["disagreement"]),
        "interpretation": (
            "Absolute accuracy figures below should be read with a +/- band of this "
            "size: two independent blind passes disagree on this fraction of rows, so "
            "some of any score gap is annotator noise rather than model signal."
        ),
    }

    result = {
        "pass1_source": args.pass1,
        "pass2_source": args.pass2,
        "sample_size": int(len(merged)),
        "overall": overall,
        "by_stratum": by_stratum,
        "confusion_of_disagreements": confusion_records,
        "uncertainty_band": band,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
