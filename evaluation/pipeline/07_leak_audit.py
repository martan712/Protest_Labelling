"""Post-hoc audit of annotation blindness and of near-duplicate train contamination.

Blindness was instructed, not enforced, so this script checks it after the fact:

1. Agreement between the blind primary label and the hidden trigger-rule label on the
   keyword stratum. The rule is, by the students' own audit, about 92.7% correct at
   best; annotation that agrees with it near or above that rate would suggest the
   annotator recovered the keyword rather than reading the grievance. Agreement well
   below the ceiling is consistent with genuine blind reading.
2. A fuzzy-duplicate pass between the 297 keyword-matched rows that are *not* exact
   training matches and the students' training notes. `in_students_train` used exact
   `clean_notes` equality; ACLED notes are templated, so near-duplicates can leak even
   when the exact string does not match.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy import accepted  # noqa: E402

RULE_CEILING = 0.927


def shingles(text: str, k: int = 4) -> set[str]:
    tokens = str(text).split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def fuzzy_matches(queries: list[str], corpus: list[str], threshold: float = 0.9) -> list[int]:
    """Return, for each query, whether some corpus entry reaches the Jaccard threshold."""
    query_sets = [shingles(text) for text in queries]
    corpus_sets = [shingles(text) for text in corpus]
    index: dict[str, list[int]] = defaultdict(list)
    for i, shard in enumerate(corpus_sets):
        for sh in shard:
            index[sh].append(i)
    hits: list[int] = []
    for qs in query_sets:
        counts: dict[int, int] = defaultdict(int)
        for sh in qs:
            for i in index.get(sh, ()):
                counts[i] += 1
        best = 0.0
        for i, inter in counts.items():
            union = len(qs) + len(corpus_sets[i]) - inter
            if union and inter / union >= threshold:
                best = max(best, inter / union)
                break
        hits.append(best)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", default="data/review/test_gold_v1/annotated/chunk_*.csv")
    parser.add_argument("--flagged", type=Path, default=Path("evaluation/manifests/test_manifest_flagged.csv"))
    parser.add_argument("--train", type=Path, default=Path("data/labeled_balanced_20.csv"))
    parser.add_argument("--quality", type=Path, default=Path("evaluation/results/annotation_quality.json"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/leak_audit.json"))
    parser.add_argument("--near-dup-out", type=Path, default=Path("data/review/test_gold_v1/near_dup_students.csv"))
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    gold = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in sorted(glob.glob(args.pass1))],
        ignore_index=True,
    )
    flagged = pd.read_csv(args.flagged, keep_default_na=False)
    scorable = flagged[flagged.scorable_all_models].copy()
    merged = gold.merge(
        scorable[["event_id_cnty", "clean_notes", "rule_label", "keyword_matched", "in_students_train"]],
        on="event_id_cnty",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gold):
        raise ValueError("gold covers rows outside the scorable manifest")

    keyword = merged[merged.keyword_matched]
    unknown = merged[~merged.keyword_matched]
    exact = keyword.primary_label.eq(keyword.rule_label)
    keyword_agreement = float(exact.mean())
    keyword_accepted = float(
        sum(
            accepted(rule, primary, str(alt).split("|"))
            for rule, primary, alt in zip(keyword.rule_label, keyword.primary_label, keyword.alternative_labels)
        )
        / len(keyword)
    )

    # Near-duplicate pass: the clean keyword rows that are not exact training matches.
    candidates = keyword[~keyword.in_students_train]
    train = pd.read_csv(args.train, low_memory=False)
    corpus = train.clean_notes.fillna("").astype(str).tolist()
    scores = fuzzy_matches(candidates.clean_notes.fillna("").astype(str).tolist(), corpus, args.threshold)
    candidates = candidates.assign(near_dup_students_train=[score >= args.threshold for score in scores],
                                    near_dup_jaccard=scores)
    affected = int(candidates.near_dup_students_train.sum())
    candidates[["event_id_cnty", "near_dup_students_train", "near_dup_jaccard"]].to_csv(args.near_dup_out, index=False)

    quality = json.loads(args.quality.read_text())
    verdict = "contaminated" if keyword_agreement >= RULE_CEILING else "consistent_with_blind_reading"
    result = {
        "keyword_stratum": {
            "n": int(len(keyword)),
            "rule_exact_agreement": keyword_agreement,
            "rule_accepted_agreement": keyword_accepted,
            "rule_accuracy_ceiling": RULE_CEILING,
            "above_ceiling": keyword_agreement >= RULE_CEILING,
        },
        "unknown_stratum": {
            "n": int(len(unknown)),
            "rule_label": "unknown",
            "rule_exact_agreement": 0.0,
            "note": (
                "Structurally degenerate control: 'unknown' is not a taxonomy class, so no "
                "agreement with the rule is possible. The recovery control is instead the "
                "pass-1/pass-2 agreement within each stratum and the fact that the models "
                "reproduce the unknown-stratum gold at only ~55%."
            ),
            "pass1_pass2_agreement": quality["by_stratum"]["unknown"]["agreement"],
            "share_labelled_other": float(unknown.primary_label.eq("other").mean()),
        },
        "keyword_stratum_pass1_pass2_agreement": quality["by_stratum"]["keyword"]["agreement"],
        "near_duplicate_students_train": {
            "candidate_rows": int(len(candidates)),
            "affected": affected,
            "threshold": args.threshold,
            "note": "Keyword-matched scorable rows not caught by exact clean_notes match that are near-duplicates of the students' training notes.",
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
