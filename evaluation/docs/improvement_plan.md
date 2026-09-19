# Improve accepted-pair accuracy with keywords and T5

Keep keyword supervision, keep keywords in the input, and keep T5. Use the existing 20 topics
plus **`other`**. Target **90%, then 95%+ accepted-pair accuracy**; these are experimental goals,
not promised results. The main work is correcting training labels and covering the full time period.

## 1. Fix the label and scoring definitions

Use this exact set in one shared evaluator, replacing the broader `ACCEPTANCE_GROUPS` in the
checkpoint and prediction notebooks:

~~~python
ACCEPTED_PAIRS = {frozenset(p) for p in [
    ("climate", "environment"),
    ("unjust law enforcement", "blm"),
    ("public services", "health care"), ("public services", "education"), ("public services", "housing"),
    ("discrimination", "women rights"), ("discrimination", "lgbtq"),      ("discrimination", "blm"),
]}

def accepted(prediction, primary, alternatives=()):
    return any(
        prediction == gold or frozenset((prediction, gold)) in ACCEPTED_PAIRS
        for gold in {primary, *alternatives}
    )
~~~

Pairs are symmetric, not transitive: education–housing and women rights–lgbtq remain wrong.
Event-specific alternatives must be independently annotated and frozen before scoring.
`other` passes only an exact match; never add it as an alternative to a supported topic.

Write a short 21-class guide before relabeling. Classify the **expressed grievance**, not the
participant or venue: police demanding pay are labor rights; students protesting coal are
climate/environment; residents demanding a hospital reopen are health care.

Use `other` when no existing topic is supported. Record `other_reason` as `outside_taxonomy`
or `insufficient_information`. An unrelated disarmament event may be the former; an explicitly
unreported protest motive is the latter. `unknown` only means no keyword matched: it must not
automatically become `other`.

## 2. Reserve evaluation data before touching training labels

- Review the existing 200 examples under the new guide and keep them for development. They were
  already used to select checkpoints. The old 57.5% strict / 69.0% accepted score is a legacy
  reference; re-score the old model against the revised labels for a comparable baseline.
- Grow dev to about 400–600 reviewed events. Reserve **500–1,000 fresh unknown events** for the
  final test, sampled representatively from a fixed data/rule snapshot. Include recent years;
  the old sample contains no 2026 events.
- Have humans label without model predictions, independently double-label a substantial subset,
  and adjudicate disagreements. Add separate Dutch, rare-class, and keyword-matched audits.
- Freeze event IDs and duplicate groups before relabeling. Exclude them from every training
  source even if changed rules would now match them. Do not use test notes to improve rules.
- Keep test sampling representative. Training's year balancing and targeted cases must not change
  the headline test distribution; report enriched diagnostic sets separately.

**Deliverable:** versioned guide, shared scorer, and fixed train/dev/test exclusion manifests.

## 3. Rebuild, verify, and correct the existing training labels

Do not simply add `other` to the model or patch the old balanced CSV. Rebuild from the **full
event pool**, with stable event IDs, original notes, dates, and countries. The current balanced
file has already discarded most recent examples of large classes.

### A. Recompute every keyword label

1. Snapshot the old labels. Centralize the rule dictionary currently duplicated in
   `pipeline/1-preprocess.ipynb` and `pipeline/finetuner/triggers.py`.
2. Start each event's rule assignment afresh and run the corrected rules over **all notes**.
   The current classifier only visits rows marked `unknown`, so reusing populated labels would
   leave old mistakes untouched.
3. Fix and regression-check pension/suspension, labour conditions, environment, and nurseries.
   The old balanced file contains 112 labor rows mentioning suspension and 46 animal-welfare
   rows mentioning labour conditions.
4. Record **every matched rule**, not just the first. About 21% of currently rule-labeled events
   match multiple classes. Use documented grievance-based precedence or review unresolved
   conflicts; dictionary order must not silently decide them.
5. Add precise phrases for missed pay, contract, dismissal, factory-closure, and workload disputes,
   with negative examples such as hunger strikes. Add `other` rules only for verified narrow
   patterns; initially obtain most `other` examples through review.

### B. Verify the rows that will actually train T5

Use the year samplers in section 4 to propose the training rows, then verify them before release.
When comparing samplers, verify the union of their proposed rows so both use reviewed candidates:

- Run an independent LLM annotation pass over **every proposed training row**, using the original
  note, year/country, and 21-class guide, without showing the old label, rule label, or T5 prediction.
  Ask for the primary topic and supporting text. This is a review aid, not ground truth.
- Agreement retains the keyword label as weak supervision. Queue disagreements, conflicting rules,
  all proposed `other` assignments, and inadequate explanations for human review. An accepted-pair
  disagreement is lower priority, but still does not justify merging the primary classes.
- Human-audit agreement cases too: start with 10 random rows per populated class-year cell, plus
  extra examples from high-volume and suspect rules. This small initial audit locates problems;
  it does not establish precise per-cell accuracy.
- Review both keyword-matched and unknown events for `other`. Some existing policy/war labels
  should become `other`; many unmatched events should become ordinary topics.
- If a repeated mistake comes from a rule, fix the rule and rerun **all affected events**, rather
  than correcting only the inspected examples. Recheck the changed cohort.
- Human decisions override weak labels. Unresolved disagreements remain outside the training
  release until reviewed; unreviewed LLM suggestions never silently overwrite existing labels.
  Report excluded counts and resulting coverage gaps, then review replacements.

### C. Persist corrections and check the resulting dataset

Keep an append-only `reviewed_labels.csv`, keyed by event ID, with note hash, primary/alternative
labels, evidence, reviewer, and taxonomy version. Apply current, adjudicated overrides after every
rule rerun; flag changed notes or taxonomy versions for re-review.

Build a versioned `training_candidates.csv` containing old label, new rule label, final label,
label source/review status, matched rules, date/year, country, and duplicate-group ID.
Keep `rule_label` separate from the final `class`: this preserves the meaning of keyword-unknown.

Produce:

| File/report | What it establishes |
|---|---|
| `label_changes.csv` | Every old → final label change, including moves into/out of other, with its reason |
| `label_audit.csv` | Human decisions and error rates by rule, class, year, and annotation source |
| `labeled_balanced_21.csv` | The verified/review-gated selection actually consumed by training |
| Release manifest | Source snapshot, taxonomy/rule versions, seed, exclusions, row counts, and file hashes |

After corrections, reapply the year quotas: moving rows between classes can create shortfalls.
Verify any replacement rows before release. Draw a **fresh random human audit of at least 500 final
training rows** and report strict and accepted-pair label correctness with confidence intervals.
Use 95% accepted correctness as an initial data-quality gate; if it fails, fix the responsible
cohorts and audit again. Freeze the release only after that gate and the structural checks below
pass. Passing this sample is not proof that every rule or year is 95% correct.

**Release checks:** valid labels from the 21-class vocabulary; reviewed examples of `other`;
known-bug regressions pass; every row has provenance and a valid event date; no held-out ID/group
leakage; unresolved rows are excluded and counted; recorded class/year quotas match the actual file.

## 4. Replace oldest-first selection with explicit year quotas

The existing `groupby("class").head(3000)` leaves labor rights in 2018–2021, 75% from 2020,
and pandemic entirely in 2020. Shuffling that old balanced file cannot restore missing years.

Use the corrected full candidate pool, after held-out exclusions and duplicate handling:

1. Derive `year` from structured `event_date`. Count eligible, distinct training examples
   `n[class, year]`. Preserve distinct source events for maps; remove redundant text copies
   from training and keep related groups together across splits.
2. Keep an initial budget `B = min(3000, available examples)` per class for comparable run cost.
3. Allocate **half of B evenly across the years present in that class**. Cap each allocation at
   availability and redistribute unused slots among years with remaining examples.
4. Allocate the remaining slots **proportionally to remaining availability by year**. Round
   deterministically and redistribute rounding/cap shortfalls until the budget is filled.
5. Sample without replacement using seed 42 within each class-year quota, spreading selection
   across countries and limiting repeated campaign templates. Log the selected IDs.
6. Recalculate availability/quotas after label corrections or exclusions. If a class-year lacks
   enough good examples, send that cell to the annotation queue; never fill it with duplicates
   or knowingly bad labels.

This 50/50 recipe guarantees a coverage component while retaining some natural frequency
variation. It is a starting configuration: compare it with proportional random sampling on dev.
Do not invent 2018 pandemic examples or require a class in years where it has no supported events.

Write a `year_coverage.csv` and class-by-year heatmap showing **available, requested, selected,
and human-reviewed counts**, plus selected country shares. Flag absent years, unmet quotas, and
any year's disproportionate contribution. Missing dates must be resolved before sampling.

**Done when:** each populated class-year receives its feasible allocation; large classes span
their supported years in the current 2018–2026 snapshot; and remaining gaps have explicit reasons.
The same coverage process applies to newly annotated unknown events and `other`, not just keywords.

Update preprocessing, `finetuner/data.py`, the trainer, and active notebook paths to consume
`labeled_balanced_21.csv` and retain IDs, dates, countries, label sources, and split/group metadata.

## 5. Add missing examples and improve T5's input

Start with about **500 reviewed unknown training events**, including 50–100 `other` examples
across its two reasons. Count suitable examples already reviewed in section 3 toward this budget.
Grow nested sets to approximately 1,500, 3,000, and 5,000 as the learning curve justifies it.

Draw roughly 60% randomly and 40% from identified errors and class-year gaps. Prioritize labor
disputes, energy prices, police-as-employees, students protesting non-education issues, unrelated
war/peace events, and supported topics that resemble `other`. Collect independent examples;
never train on the dev cases that revealed the problem.

Keep keywords visible. Compare current cleaned text with minimally cleaned original notes that
preserve negation and grammatical relationships. Replace blind removal of the first three tokens
with explicit handling of the opening date.

Test note only, note + year, note + country, and both, after fixing sampling. For example:

~~~text
Event year: 2022
Country: France
Note: Foundry workers blocked the station to protest the closure of their factory.
~~~

Use structured metadata and one input builder for training/inference. Measure token lengths before
increasing the limit; none of the current 200 cleaned evaluation notes was truncated at 128.
Keep metadata only if it improves dev results without harmful year/country regressions; test a
later-period diagnostic separately.

## 6. Train, measure, and iterate

Initialize T5 for **21 labels**, checking the ID mapping when loading an old 20-class checkpoint.
Compare corrected keyword training followed by reviewed-data fine-tuning with mixed training
using 25%, 50%, or 75% reviewed examples per batch. The completed model must have trained on
`other`; keyword-only warm-up is not the finished candidate.

Select every checkpoint by accepted-pair dev accuracy. Start with ordinary cross-entropy and
argmax; use fixed seeds and confirm finalists over three seeds. Later, separately test choosing
the output that maximizes probability mass over its directly accepted gold labels. Evaluate that
decoder on dev and monitor broad-class prediction shares; it must not add transitive pairs.

| Run | Change being measured |
|---|---|
| Baseline | Old checkpoint re-scored under the new guide |
| A | Corrected, audited 21-class labels and reviewed starter set, with proportional random sampling |
| B | Explicit year sampling versus proportional random sampling, using the same verified pool |
| C | Original-note and year/country input variants, using the same rows |
| D | More reviewed unknown examples and mixture/finishing-stage choices |
| E | Targeted correction rounds and optional accepted-pair decoding |

For every run, record training-label quality, class/year coverage, accepted accuracy, strict
accuracy, and errors by topic/year/country. Report `other` precision and recall separately.
Keep a winning baseline when an experiment fails; do not assume every proposed change helps.

Once the recipe is selected, score the locked test at full coverage, including `other`, with
confidence intervals. Report Dutch and keyword-matched audits separately. Changing the gold labels,
excluding difficult cases, or widening accepted pairs cannot count as a model improvement.

Save data/model/scoring versions with predictions. Regenerate the downstream CSV and maps,
including `other` in legends and summaries. Further work follows the largest remaining rejected
confusions and the learning curve toward 90%, 95%+, and lower residual error counts.
