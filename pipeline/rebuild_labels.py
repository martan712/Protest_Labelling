"""Recompute weak labels for the full event pool and produce auditable candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from finetuner.triggers import RULE_VERSION, match_rules, rule_label
from evaluation.taxonomy import CLASS_NAMES, TAXONOMY_VERSION, validate_label

REVIEW_COLUMNS = [
    "event_id_cnty", "note_hash", "primary_label", "alternative_labels", "other_reason",
    "evidence", "reviewer", "annotation_status", "taxonomy_version",
]
FINAL_REVIEW_STATUSES = {"human_review", "adjudicated"}


def note_hash(value: object) -> str:
    return hashlib.sha256(str(value or "").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=Path("data/filtered_events.csv"))
    parser.add_argument("--legacy", type=Path, default=Path("data/labeled.csv"))
    parser.add_argument("--reviews", type=Path, default=Path("data/reviews/reviewed_labels.csv"))
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
    base["label_source"] = base["rule_label"].notna().map({True: "rule", False: "unresolved"})
    base["review_status"] = "unreviewed"
    base["alternative_labels"] = ""
    base["other_reason"] = ""
    base["taxonomy_version"] = TAXONOMY_VERSION
    base["rule_version"] = RULE_VERSION

    # Evaluation labels are deliberately never imported here. Only the append-only
    # training review ledger may affect candidates.
    if args.reviews.exists():
        review = pd.read_csv(args.reviews, keep_default_na=False)
        missing = set(REVIEW_COLUMNS) - set(review.columns)
        if missing:
            raise ValueError(f"Review ledger is missing columns: {sorted(missing)}")
    else:
        review = pd.DataFrame(columns=REVIEW_COLUMNS)
    active = review.drop_duplicates("event_id_cnty", keep="last")
    event_index = base.set_index("event_id_cnty")
    issues: list[dict[str, str]] = []
    for _, row in active.iterrows():
        event_id = row["event_id_cnty"]
        if event_id not in event_index.index:
            issues.append({"event_id_cnty": event_id, "issue": "event_missing"})
            continue
        expected_hash = event_index.loc[event_id, "note_hash"]
        if row["note_hash"] != expected_hash:
            issues.append({"event_id_cnty": event_id, "issue": "note_hash_changed"})
            continue
        if row["taxonomy_version"] != TAXONOMY_VERSION:
            issues.append({"event_id_cnty": event_id, "issue": "taxonomy_version_changed"})
            continue
        try:
            validate_label(row["primary_label"], row["other_reason"])
            alternatives = [value.strip() for value in row["alternative_labels"].split("|") if value.strip()]
            for alternative in alternatives:
                validate_label(alternative)
            if row["primary_label"] in alternatives:
                raise ValueError("primary label repeated as alternative")
            if not row["evidence"].strip() or not row["reviewer"].strip():
                raise ValueError("evidence and reviewer are required")
        except ValueError as error:
            issues.append({"event_id_cnty": event_id, "issue": f"invalid_review: {error}"})
            continue
        mask = base["event_id_cnty"].eq(event_id)
        status = row["annotation_status"]
        if status in FINAL_REVIEW_STATUSES:
            base.loc[mask, "final_label"] = row["primary_label"]
            base.loc[mask, "label_source"] = "reviewed_override"
            base.loc[mask, "review_status"] = status
        elif status == "independent_llm_review":
            # These are provisional labels produced blind from the note. They are
            # suitable for this experimental run and remain explicitly distinct
            # from later human review/adjudication.
            base.loc[mask, "final_label"] = row["primary_label"]
            base.loc[mask, "label_source"] = "independent_llm_review"
            base.loc[mask, "review_status"] = "independent_review"
        else:
            issues.append({"event_id_cnty": event_id, "issue": f"unsupported_status: {status}"})
            continue
        base.loc[mask, "alternative_labels"] = row["alternative_labels"]
        base.loc[mask, "other_reason"] = row["other_reason"]

    columns = ["event_id_cnty", "notes", "clean_notes", "event_date", "year", "country", "old_label", "rule_label", "final_label", "alternative_labels", "other_reason", "label_source", "review_status", "matched_rules", "matched_classes", "duplicate_group_id", "note_hash", "rule_conflict", "taxonomy_version", "rule_version"]
    candidates = base[columns]
    candidates.to_csv(args.out / "training_candidates.csv", index=False)
    pd.DataFrame(issues, columns=["event_id_cnty", "issue"]).to_csv(args.out / "review_issues.csv", index=False)

    changes = candidates[candidates.old_label.fillna("") != candidates.final_label.fillna("")].copy()
    changes["change_reason"] = changes.apply(
        lambda r: "reviewed_override" if r.review_status in FINAL_REVIEW_STATUSES
        else ("rule_conflict" if r.rule_conflict else ("unresolved" if pd.isna(r.final_label) else "recomputed_rule")),
        axis=1,
    )
    changes[["event_id_cnty", "old_label", "rule_label", "final_label", "change_reason", "matched_classes", "review_status"]].to_csv(args.out / "label_changes.csv", index=False)
    audit = candidates.groupby(["label_source", "final_label"], dropna=False).size().rename("rows").reset_index()
    audit["rule_version"] = RULE_VERSION
    audit["taxonomy_version"] = TAXONOMY_VERSION
    audit.to_csv(args.out / "label_audit.csv", index=False)
    (args.out / "release_manifest.json").write_text(json.dumps({
        "status": "draft", "events": str(args.events), "legacy_labels": str(args.legacy),
        "reviewed_labels": str(args.reviews),
        "taxonomy_version": TAXONOMY_VERSION, "rule_version": RULE_VERSION,
        "rows": len(candidates), "resolved_rows": int(candidates.final_label.notna().sum()),
        "unresolved_rows": int(candidates.final_label.isna().sum()), "conflict_rows": int(candidates.rule_conflict.sum()),
        "review_issues": len(issues), "evaluation_labels_imported": False,
    }, indent=2) + "\n")
    print(f"wrote {len(candidates):,} candidates; {candidates.final_label.notna().sum():,} resolved; {candidates.final_label.isna().sum():,} unresolved")


if __name__ == "__main__":
    main()
