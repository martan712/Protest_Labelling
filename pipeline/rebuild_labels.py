"""Recompute weak labels for the full event pool and produce auditable candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from finetuner.triggers import RULE_VERSION, match_rules, rule_label

TAXONOMY_VERSION = "21-class-v1"
VALID_LABELS = {
    "animal welfare", "blm", "climate", "culture", "discrimination", "education", "environment",
    "farmers", "health care", "housing", "immigration", "labor rights", "lgbtq", "other",
    "palestine-israel conflict", "pandemic", "policies & politics", "public services",
    "ukraine-russia war", "unjust law enforcement", "women rights",
}


def note_hash(value: object) -> str:
    return hashlib.sha256(str(value or "").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("data/filtered_events.csv"))
    parser.add_argument("--legacy", type=Path, default=Path("data/labeled.csv"))
    parser.add_argument("--dev", type=Path, default=Path("data/manual_labelled_data/random_unknown_labeled.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/release"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    old = pd.read_csv(args.legacy, usecols=["event_id_cnty", "class"])
    base = events.merge(old, on="event_id_cnty", how="left", validate="one_to_one").rename(columns={"class": "old_label"})
    base["event_date"] = pd.to_datetime(base["event_date"], errors="coerce")
    if base["event_date"].isna().any():
        raise ValueError("Every candidate must have a valid event_date")
    base["year"] = base["event_date"].dt.year.astype(int)
    base["note_hash"] = base["notes"].map(note_hash)
    base["duplicate_group_id"] = "note-" + base["note_hash"].str[:16]
    matched = base["notes"].map(match_rules)
    base["matched_rules"] = matched.map(lambda x: json.dumps(x, sort_keys=True))
    base["matched_classes"] = matched.map(lambda x: "|".join(sorted(x)))
    base["rule_label"] = matched.map(rule_label)
    base["rule_conflict"] = matched.map(lambda x: len(x) > 1)
    base["final_label"] = base["rule_label"]
    base["label_source"] = base["rule_label"].map(lambda x: "rule" if x else "unresolved")
    base["review_status"] = "unreviewed"

    legacy_dev = pd.read_csv(args.dev, usecols=["event_id_cnty", "manual_label", "manual_label_alt"])
    review = events.merge(legacy_dev, on="event_id_cnty", how="inner", validate="one_to_one")
    review["note_hash"] = review["notes"].map(note_hash)
    review["primary_label"] = review["manual_label"]
    review["alternative_labels"] = review["manual_label_alt"].fillna("")
    review["evidence"] = review["notes"].fillna("")
    review["reviewer"] = "legacy_manual_review"
    review["review_status"] = "adjudicated"
    review["taxonomy_version"] = TAXONOMY_VERSION
    review["other_reason"] = ""
    bad = ~review["primary_label"].isin(VALID_LABELS)
    if bad.any():
        raise ValueError(f"Invalid reviewed labels: {review.loc[bad, 'primary_label'].unique().tolist()}")
    overrides = review.set_index("event_id_cnty")
    for event_id, row in overrides.iterrows():
        mask = base["event_id_cnty"].eq(event_id)
        base.loc[mask, "final_label"] = row["primary_label"]
        base.loc[mask, "label_source"] = "reviewed_override"
        base.loc[mask, "review_status"] = "adjudicated"

    columns = ["event_id_cnty", "notes", "clean_notes", "event_date", "year", "country", "old_label", "rule_label", "final_label", "label_source", "review_status", "matched_rules", "matched_classes", "duplicate_group_id", "note_hash", "rule_conflict"]
    candidates = base[columns]
    candidates.to_csv(args.out / "training_candidates.csv", index=False)
    review[["event_id_cnty", "note_hash", "primary_label", "alternative_labels", "other_reason", "evidence", "reviewer", "review_status", "taxonomy_version"]].to_csv(args.out / "reviewed_labels.csv", index=False)

    changes = candidates[candidates.old_label.fillna("") != candidates.final_label.fillna("")].copy()
    changes["change_reason"] = changes.apply(lambda r: "reviewed_override" if r.review_status == "adjudicated" else ("rule_conflict" if r.rule_conflict else "recomputed_rule"), axis=1)
    changes[["event_id_cnty", "old_label", "rule_label", "final_label", "change_reason", "matched_classes", "review_status"]].to_csv(args.out / "label_changes.csv", index=False)
    audit = candidates.groupby(["label_source", "final_label"], dropna=False).size().rename("rows").reset_index()
    audit["rule_version"] = RULE_VERSION
    audit["taxonomy_version"] = TAXONOMY_VERSION
    audit.to_csv(args.out / "label_audit.csv", index=False)
    (args.out / "release_manifest.json").write_text(json.dumps({
        "events": str(args.events), "legacy_labels": str(args.legacy), "reviewed_labels": str(args.dev),
        "taxonomy_version": TAXONOMY_VERSION, "rule_version": RULE_VERSION,
        "rows": len(candidates), "resolved_rows": int(candidates.final_label.notna().sum()),
        "unresolved_rows": int(candidates.final_label.isna().sum()), "conflict_rows": int(candidates.rule_conflict.sum()),
    }, indent=2) + "\n")
    print(f"wrote {len(candidates):,} candidates; {candidates.final_label.notna().sum():,} resolved; {candidates.final_label.isna().sum():,} unresolved")


if __name__ == "__main__":
    main()
