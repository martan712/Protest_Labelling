"""Create the reviewed starter set from previously keyword-unknown events.

This pass is explicitly marked ``self_annotated`` so later human reviewers can replace it;
it never touches the frozen dev/test manifests or old labels silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/release/training_candidates.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data/release/reviewed_labels.csv"))
    parser.add_argument("--heldout", type=Path, default=Path("evaluation/manifests/heldout_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/release"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--other", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    heldout = pd.read_csv(args.heldout)
    eligible = candidates[
        candidates.old_label.eq("unknown")
        & ~candidates.event_id_cnty.isin(set(heldout.event_id_cnty))
        & ~candidates.duplicate_group_id.isin(set(heldout.duplicate_group_id))
    ].copy()
    rule_rows = eligible[eligible.rule_label.notna()]
    other_rows = eligible[eligible.rule_label.isna()]
    if len(rule_rows) < args.n - args.other or len(other_rows) < args.other:
        raise ValueError("The unknown pool cannot satisfy the requested starter annotation size")
    rule_rows = rule_rows.sample(n=args.n - args.other, random_state=args.seed)
    other_rows = other_rows.sample(n=args.other, random_state=args.seed)
    chosen = pd.concat([rule_rows, other_rows], ignore_index=True)
    outside_terms = r"disarmament|military|weapon|peace|veteran|war"
    chosen["primary_label"] = chosen.rule_label.fillna("other")
    chosen["other_reason"] = ""
    chosen.loc[chosen.rule_label.isna(), "other_reason"] = chosen.loc[chosen.rule_label.isna(), "notes"].str.contains(outside_terms, case=False, regex=True, na=False).map(lambda x: "outside_taxonomy" if x else "insufficient_information")
    chosen["alternative_labels"] = ""
    chosen["evidence"] = chosen.apply(lambda row: row.notes if pd.isna(row.rule_label) else f"Matched rule(s): {row.matched_rules}", axis=1)
    chosen["reviewer"] = "codex"
    chosen["review_status"] = "self_annotated"
    chosen["taxonomy_version"] = "21-class-v1"
    review_cols = ["event_id_cnty", "note_hash", "primary_label", "alternative_labels", "other_reason", "evidence", "reviewer", "review_status", "taxonomy_version"]
    prior = pd.read_csv(args.reviews)
    prior = prior[~prior.event_id_cnty.isin(set(chosen.event_id_cnty))]
    pd.concat([prior[review_cols], chosen[review_cols]], ignore_index=True).drop_duplicates("event_id_cnty").to_csv(args.reviews, index=False)
    updates = candidates.set_index("event_id_cnty")
    for event_id, row in chosen.set_index("event_id_cnty").iterrows():
        updates.loc[event_id, "final_label"] = row.primary_label
        updates.loc[event_id, "label_source"] = "self_annotated"
        updates.loc[event_id, "review_status"] = "self_annotated"
    updates.reset_index().to_csv(args.candidates, index=False)
    chosen.to_csv(args.out / "starter_annotations.csv", index=False)
    print(f"annotated {len(chosen)} unknown events: {len(rule_rows)} rule-supported and {len(other_rows)} other")


if __name__ == "__main__":
    main()
