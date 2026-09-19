# evaluation

Honest measurement of the protest topic classifiers.

**Headline: 72% strict / 76% accepted, corpus-weighted, on blind gold.** The project's
previously reported 98.27% measured the model against the keyword labels it was trained to
reproduce, with the keywords still in its input. Full numbers and corrections:
[`docs/test_results.md`](docs/test_results.md).

## The one thing to understand first

`class == 'unknown'` in `data/labeled.csv` is **not** a model output and **not** a kind of
protest. It means the trigger-word dictionary matched nothing. The corpus is ~60%
keyword-matched / ~40% unknown, and the two need separate scores:

- on **keyword-matched** rows the label came from a keyword that is still in the note, so
  scoring a model there against that label measures memorisation, not understanding;
- on **unknown** rows the model gets no keyword to copy and the row was never trained on, so
  that is where it does the actual task.

A single blended accuracy hides this. Every score here is reported per stratum, with a
corpus-weighted combination.

## Layout

```
evaluation/
├── taxonomy.py              # 21 classes, accepted pairs, validation — the shared contract
├── taxonomy_guide.md        # what annotators are given
├── pipeline/                # run in order; all default to repo-root-relative paths
├── scoring/                 # score_predictions.py (frozen), score_checkpoint.py
├── manifests/               # frozen event ID sets + flag metadata
├── results/                 # JSON/CSV metrics, committed
├── notebooks/               # verify_students.ipynb, verify_ours.ipynb + error reviews
└── docs/                    # results, protocol, approach notes
```

## Reproduce

From the repository root, with `.venv/bin/python`. Steps 01–05 rebuild the frozen test set
and gold; **normally you only need 06–08**, which recompute all metrics from existing
annotations.

```bash
.venv/bin/python evaluation/pipeline/02_flag_manifest.py            # strata + contamination flags
.venv/bin/python evaluation/pipeline/06_analyse_annotation_quality.py  # agreement, Cohen's kappa
.venv/bin/python evaluation/pipeline/07_leak_audit.py               # blindness + near-duplicate check
.venv/bin/python evaluation/pipeline/08_score_strata.py             # the four-way breakdown
```

Steps 03–05 rebuild the annotation queue and gold; 03 and 04 only regenerate inputs for a new
annotation round. All scripts take `--help` and default their paths.

Notebooks run from `evaluation/notebooks/`:

```bash
cd evaluation/notebooks
../../.venv/bin/python -m nbconvert --to notebook --execute --inplace verify_students.ipynb
```

`verify_ours.ipynb` reloads a T5 checkpoint and re-runs inference; it needs
`models/attempt7-t5-leave-keywords-in-epoch-1/` and a GPU to be quick.

## Pipeline

| Step | Produces |
|---|---|
| `01_build_manifests.py` | Freezes dev/test event IDs before any rule change |
| `02_flag_manifest.py` | `test_manifest_flagged.csv` — strata, per-model contamination, predictions, common date window |
| `03_build_annotation_queue.py` | Blind queue + chunks under `data/review/test_gold_v1/` |
| `04_build_adjudication_set.py` | 150-row second-pass sample (seed 7) |
| `05_assemble_test_gold.py` | `test_gold.csv`, `test_manifest_840.csv`, per-model prediction CSVs |
| `06_analyse_annotation_quality.py` | `results/annotation_quality.json` — agreement, κ, disagreements |
| `07_leak_audit.py` | `results/leak_audit.json` — blindness verification, near-duplicates |
| `08_score_strata.py` | `results/test-*_strata.json` + `_by_topic.csv` |

## Rules this folder is built on

1. **The manifest is frozen.** `manifests/test_manifest.csv` was sampled before any label or
   rule change. Never modify it. Same for `taxonomy.py`, `scoring/score_predictions.py`, and
   `data/manual_labelled_data/`.
2. **Annotation is blind.** The queue carries only note, date, year, country. No rule label,
   no prediction, and no `clean_notes` — that is the lemmatised text the matcher ran on, and
   so a route back to the rule. This is not ceremony: the students' unblinded self-grading
   returned 77.2% where blind annotation returns 55.0%.
3. **Contamination is flagged, never removed.** Rows in a model's training data are scored
   separately as a memorisation ceiling. Removing them would bias the set (the leaked rows
   skew pre-2022) and would make the set model-specific, destroying cross-model comparison.
4. **Combined scores are corpus-weighted** (0.612 keyword / 0.388 unknown), never a raw mean,
   because dropping contaminated rows is non-random.
5. **The test set is spent once.** These annotations must never be used to fix trigger rules
   or train anything.

## Gold data and `.gitignore`

`data/` is gitignored. The irreplaceable annotation artifacts under
`data/review/test_gold_v1/` are force-added, following the precedent set for
`data/manual_labelled_data/`. Regenerable artifacts (queue, chunks) are not tracked.

## Caveats

Gold is LLM-annotated with LLM-measured agreement (κ=0.912). Before this goes to the client,
a human should adjudicate a subsample — step 04 accepts a human pass with no other change.
Seven classes can never have a clean keyword score for the students' model, and `other` is
currently never predicted. See [`docs/test_results.md`](docs/test_results.md) §7.

## Documents

| File | What it is |
|---|---|
| [`docs/test_results.md`](docs/test_results.md) | **The results.** Headline numbers, head-to-head, corrections to previously published figures |
| [`docs/annotation_task.md`](docs/annotation_task.md) | The annotation protocol, as executed |
| [`docs/current_approach.md`](docs/current_approach.md) | The 21-class training workflow |
| [`docs/improvement_plan.md`](docs/improvement_plan.md) | Plan that led to the 21-class redesign |
| [`docs/from_scratch_approach.md`](docs/from_scratch_approach.md) | Alternative clean-slate design |
