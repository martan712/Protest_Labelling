#!/usr/bin/env python3
"""Compare two independent annotation sets over the same manifests.

Two agreement measures, and nothing else:

  strict  -- the two primary labels are identical.
  lenient -- the {primary, secondary} sets share at least one label, so a
             label one annotator made primary and the other made secondary
             counts as agreement.

`alternative_labels` holds at most one label and is never split: several class
names contain commas (for example `crime, violence and victim justice`).

Cohen's kappa is reported for the strict measure only. Kappa needs exactly one
categorical choice per rater, which the lenient measure does not provide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train_0800", "train_1500", "train_3000", "train_6000", "dev", "test_locked")
KEY = "event_id_cnty"


def resolve(target: Path, split: str) -> Path:
    """A run directory resolves to its assembled split; a CSV is used as given."""
    if target.is_file():
        return target
    for candidate in (target / "assembled" / f"{split}.csv", target / f"{split}.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No {split}.csv under {target}")


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = {KEY, "primary_label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[KEY].duplicated().any():
        raise ValueError(f"{path} contains duplicate {KEY} values")
    if "alternative_labels" not in frame:
        frame["alternative_labels"] = ""
    return frame[[KEY, "primary_label", "alternative_labels"]]


def label_set(primary: str, secondary: str) -> set[str]:
    """The labels an annotator put forward: the primary, plus a secondary if given."""
    return {primary} | ({secondary.strip()} if secondary.strip() else set())


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    merged = a.merge(b, on=KEY, how="inner", suffixes=("_a", "_b"))
    only_a = len(a) - len(merged)
    only_b = len(b) - len(merged)
    if merged.empty:
        raise ValueError("The two sets share no event IDs")

    left = merged.primary_label_a
    right = merged.primary_label_b
    strict = left == right

    sets_a = [label_set(p, s) for p, s in zip(left, merged.alternative_labels_a)]
    sets_b = [label_set(p, s) for p, s in zip(right, merged.alternative_labels_b)]
    lenient = pd.Series(
        [bool(x & y) for x, y in zip(sets_a, sets_b)], index=merged.index
    )

    labels = sorted(set(left) | set(right))
    per_label = []
    for label in labels:
        in_a = left == label
        in_b = right == label
        both = int((in_a & in_b).sum())
        either = int((in_a | in_b).sum())
        per_label.append({
            "label": label,
            "n_a": int(in_a.sum()),
            "n_b": int(in_b.sum()),
            "agreed": both,
            "jaccard": round(both / either, 4) if either else 0.0,
        })

    disagreements = (
        merged.loc[~strict, ["primary_label_a", "primary_label_b"]]
        .value_counts().head(15).reset_index(name="count")
        .to_dict("records")
    )

    return {
        "n_compared": len(merged),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "strict_overlap": round(float(strict.mean()), 4),
        "lenient_overlap": round(float(lenient.mean()), 4),
        "cohen_kappa_strict": round(float(cohen_kappa_score(left, right, labels=labels)), 4),
        "n_secondary_a": int((merged.alternative_labels_a.str.strip() != "").sum()),
        "n_secondary_b": int((merged.alternative_labels_b.str.strip() != "").sum()),
        "n_labels_used": len(labels),
        "per_label": sorted(per_label, key=lambda row: row["jaccard"]),
        "top_disagreements": disagreements,
    }


def render(split: str, result: dict, *, name_a: str, name_b: str, top: int) -> None:
    print(f"\n=== {split} ===")
    print(f"compared {result['n_compared']} events "
          f"(only in {name_a}: {result['only_in_a']}, only in {name_b}: {result['only_in_b']})")
    print(f"  strict overlap  (primary == primary)   {result['strict_overlap']:>8.2%}")
    print(f"  lenient overlap (primary or secondary) {result['lenient_overlap']:>8.2%}")
    print(f"  Cohen's kappa   (strict)               {result['cohen_kappa_strict']:>8.4f}"
          f"   [{result['n_labels_used']} labels]")
    print(f"  secondary labels given: {name_a} {result['n_secondary_a']}, "
          f"{name_b} {result['n_secondary_b']}")

    print(f"\n  weakest labels by strict agreement (Jaccard):")
    print(f"    {'label':<58}{name_a:>7}{name_b:>7}{'agree':>7}{'jacc':>8}")
    for row in result["per_label"][:top]:
        print(f"    {row['label'][:56]:<58}{row['n_a']:>7}{row['n_b']:>7}"
              f"{row['agreed']:>7}{row['jaccard']:>8.3f}")

    print(f"\n  most frequent disagreements ({name_a} -> {name_b}):")
    for row in result["top_disagreements"][:top]:
        print(f"    {row['count']:>5}  {row['primary_label_a'][:34]:<36} -> "
              f"{row['primary_label_b'][:34]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Run directory or CSV")
    parser.add_argument("--b", type=Path, required=True, help="Run directory or CSV")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--splits", nargs="+", default=["dev", "test_locked", "train_6000"],
                        help=f"Any of: {', '.join(SPLITS)}")
    parser.add_argument("--top", type=int, default=10, help="Rows per detail table")
    parser.add_argument("--output", type=Path, help="Write the full report as JSON")
    args = parser.parse_args()

    report = {"a": str(args.a), "b": str(args.b), "splits": {}}
    for split in args.splits:
        result = compare(load(resolve(args.a, split)), load(resolve(args.b, split)))
        report["splits"][split] = result
        render(split, result, name_a=args.name_a, name_b=args.name_b, top=args.top)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
