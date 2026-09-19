# Annotation protocol — frozen test set

**Status: executed.** Results in `test_results.md`. This file is the protocol record: the
rules the gold in `data/review/test_gold_v1/` was produced under, and the spec to follow when
extending or re-running the annotation. Paths reflect the current `evaluation/` layout.

## Why this exists

`class == 'unknown'` in `data/labeled.csv` is not a model output and not a kind of protest.
It is the trigger-word dictionary's residue: no keyword fired. The corpus is ~60%
keyword-matched / ~40% unknown.

Before this work, the unknown 40% had one honest number (55.0% strict) and the keyword-matched
60% had none — only agreement with the rules the model was trained to reproduce.

The failure mode being guarded against is not a bug. It is an annotator who sees the answer
before deciding. That is precisely how the students' 77.2% became 55.0% under blind
re-annotation. **Blindness is the entire point**; compromising it produces a number that is
worse than none, because it will be believed.

## Fixed inputs — never modify

- `manifests/test_manifest.csv` — 1,000 events, frozen before any rule change.
- `taxonomy.py` — `CLASS_NAMES` (21), `ACCEPTED_PAIRS`, `validate_label`.
- `taxonomy_guide.md` — what annotators are given.
- `scoring/score_predictions.py` — frozen scorer; enforces exact manifest coverage and
  note-hash equality.
- `data/manual_labelled_data/` — the legacy 200-event dev set.

## Protocol

### 1. Queue (`pipeline/03_build_annotation_queue.py`)

Selects `scorable_all_models` rows from `manifests/test_manifest_flagged.csv` and writes
`data/review/test_gold_v1/annotation_queue.csv` with exactly:

```
annotation_order, event_id_cnty, event_date, year, country, notes, note_hash,
primary_label, alternative_labels, other_reason, evidence
```

**Must not appear:** `rule_label`, `keyword_matched`, `class`, `in_students_train`,
`in_release21_train`, `pred_students`, `pred_release21`, `clean_notes`.

`clean_notes` is excluded deliberately — it is the lemmatised text the keyword matcher ran
on, and therefore a route back to the rule.

Shuffled under seed 42 so keyword and unknown rows interleave. In blocks, an annotator
notices the texture difference between strata and calibrates differently across them, which
biases the exact comparison the test set exists to make. Split into chunks of 60.

### 2. Annotation

One subagent per chunk, 4–6 in flight. Each prompt contains the **full inlined text** of
`taxonomy_guide.md` — inlining is what makes it true that the annotator saw the guide and
nothing else — plus its chunk path and this instruction verbatim:

> Read only this chunk file. Do not open `data/labeled.csv`, any `filtered_events_*` file,
> any file under `evaluation/manifests/`, any other chunk, or any file containing existing
> labels or predictions. Do not grep the repository for these event IDs or for any phrase
> from the notes. Your judgement must come from the note text alone. If you cannot decide
> from the note, that is a finding — record `other` with
> `other_reason=insufficient_information`. It is not a reason to look something up.

Output contract:

- `primary_label` — exactly one of the 21 `CLASS_NAMES`.
- `alternative_labels` — pipe-separated or empty; only a genuinely independently-supported
  reading. **Never populated from the accepted-pair table** — the scorer applies those pairs
  itself, and doing both double-counts. (The legacy gold did this on 21% of rows, which is
  the entire 68.5% → 58.5% correction.)
- `other_reason` — `outside_taxonomy` or `insufficient_information`, required when and only
  when `primary_label == other`.
- `evidence` — a short quote or paraphrase. Not optional; it is the audit trail and it
  measurably slows careless labelling.
- `reviewer` — chunk id, e.g. `gold-chunk-03`.

The two rules carrying most of the error: label the **expressed grievance**, not the
participant, venue or tactic (police demanding pay is `labor rights`); and decide
`climate`/`environment` and `labor rights`/`policies & politics` on the grievance as stated
rather than hedging into the vaguer class.

### 3. Second pass (`pipeline/04_build_adjudication_set.py`)

150 rows, seed 7, stratified to preserve the 61/39 split. Re-annotated by a **separate**
subagent set with no sight of pass 1. `pipeline/06_analyse_annotation_quality.py` records
agreement, Cohen's κ, per-stratum figures, and the disagreement confusion matrix.

Disagreements keep the pass-1 label and are flagged `disputed`. They are **not** resolved by
a third model — the disagreement is carried into the reported uncertainty instead.

An LLM annotation set with no measured error rate is not ground truth, and calling it gold is
the students' mistake in a new costume.

### 4. Leak audit (`pipeline/07_leak_audit.py`)

Blindness is instructed, not enforced, so verify after the fact. Compare gold against the
hidden `rule_label` on the keyword stratum: the rule labels are at best ~92.7% correct by the
students' own audit, so agreement near or above that suggests the annotator recovered the
keyword rather than reading the grievance. Measured: **79.4%** — consistent with blind
reading.

Note the unknown stratum is a degenerate control (`unknown` is not a class, so agreement is
structurally 0); pass-1/pass-2 agreement serves instead.

Also runs a fuzzy-duplicate pass (Jaccard ≥ 0.9) between the clean keyword rows and
`data/labeled_balanced_20.csv`, since `in_*_train` is exact-match only and ACLED notes are
templated. Result: 0 affected.

### 5. Gold and scoring (`pipeline/05_assemble_test_gold.py`, `08_score_strata.py`)

Gold validates: every label passes `validate_label`; IDs exactly match the scorable set; no
duplicates; every `note_hash` matches the frozen manifest; alternatives are valid class names.
The scorer reads gold with `keep_default_na=False`, so blanks are empty strings.

`08_score_strata.py` reports four row sets per model — keyword-clean, keyword-in-training
(memorisation ceiling), unknown, and combined over clean rows only. The combined figure is
**corpus-weighted** (0.612/0.388), never a raw mean, because dropping contaminated rows is
non-random: the leaked rows skew pre-2022.

The leaked-row score never enters the headline. It is reported adjacent and labelled.

## Standing constraints

- These annotations must never be used to fix trigger rules or train anything. The set is
  spent the moment it influences a model.
- Everything is additive and reproducible from a seed.
- Contamination is flagged, never removed from the data — it is a property of the model, not
  of the test set, and carving the set per model destroys cross-model comparability.
