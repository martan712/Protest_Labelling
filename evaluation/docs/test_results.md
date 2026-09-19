# Test results — frozen 840-row test set, blind gold

Taxonomy `21-class-v1`. Gold: `data/review/test_gold_v1/test_gold.csv`.
Reproduce with `evaluation/pipeline/` steps 02–08. See `README.md` for commands.

## Summary

The project's headline accuracy was **98.27%**. That number measured the model against the
keyword labels it was trained to reproduce, with the keywords still in its input. The honest
number, on blind gold across the whole corpus, is **72%**.

The two models evaluated — the students' T5 and the 21-class release — are **statistically
indistinguishable**.

| | students' T5 | release21 |
|---|---:|---:|
| Combined, corpus-weighted, strict | **72.2%** [68.9, 75.6] | **70.0%** [67.0, 73.0] |
| Combined, corpus-weighted, accepted | **75.6%** [72.5, 78.7] | **73.1%** [70.2, 76.1] |

Those two columns are *not* directly comparable — "clean" means 297 rows for the students'
model and 505 for release21, because contamination differs. On identical rows they are level
(§3).

## 1. What is being measured

`class == 'unknown'` in `data/labeled.csv` is not a model output and not a kind of protest.
It is the trigger-word dictionary's residue: no keyword fired. The corpus splits
**60.8% keyword-matched / 39.2% unknown**; in the test draw, 61.2% / 38.8%.

Every previously reported figure was one of two broken things:

| Figure | Source | Problem |
|---|---|---|
| 98.27% | report Table 5 | Scored against keyword labels the model was trained to reproduce, keywords still in the input. A pure keyword lookup scores 81.0% on the same split. Agreement with a regex. |
| 77.2% | report Table 6 | The authors graded their own model's output while looking at the prediction. Blind re-annotation of that population returns 55.0%. |

So the `unknown` 40% had one honest number and the keyword-matched 60% had none. This test
set produces one for both.

## 2. Method

- **Frozen manifest.** `evaluation/manifests/test_manifest.csv`, 1,000 events, sampled before
  any label or rule change and never modified.
- **Common window.** Restricted to events on or before **2025-05-23**, the students'
  snapshot end, so no model is scored on events postdating its own data. 1,000 → 862 in
  window → **840 scorable by both models**. 160 manifest rows sit outside and are unused here.
- **Strata.** 514 keyword-matched, 326 unknown.
- **Contamination flagged, not removed.** 217 of the keyword rows are in the students'
  training data, 9 in release21's. They are scored separately as a memorisation ceiling and
  excluded from the headline. Removing them from the *data* was rejected: the leaked rows are
  63% pre-2022 while the clean replacement pool has no 2018–19 rows at all, so substitution
  would bias the set toward recent events; and contamination is a property of the model, not
  of the test set.
- **Blind annotation.** The queue carried only `event_id_cnty`, `event_date`, `year`,
  `country`, `notes`. No rule label, no prediction, no `clean_notes` (the lemmatised text the
  matcher ran on, and so a route back to the rule). Strata interleaved under seed 42 so
  annotators could not calibrate differently across them.
- **Combined figure** is corpus-weighted (keyword 0.612 / unknown 0.388), not a raw mean,
  because dropping contaminated rows is non-random.

## 3. Results

### Students' T5

| Stratum | n | strict | 95% CI | accepted |
|---|---:|---:|---|---:|
| keyword, not in training | 297 | 82.8% | [78.1, 86.7] | 87.2% |
| keyword, in training (ceiling) | 217 | 74.7% | [68.5, 80.0] | 78.3% |
| no-keyword | 326 | 55.5% | [50.1, 60.8] | 57.4% |
| **combined, weighted** | 623 | **72.2%** | [68.9, 75.6] | **75.6%** |

### release21

| Stratum | n | strict | 95% CI | accepted |
|---|---:|---:|---|---:|
| keyword, not in training | 505 | 78.4% | [74.6, 81.8] | 82.0% |
| keyword, in training (ceiling) | 9 | 100% | — | 100% |
| no-keyword | 326 | 56.7% | [51.3, 62.0] | 59.2% |
| **combined, weighted** | 831 | **70.0%** | [67.0, 73.0] | **73.1%** |

### Head-to-head, identical rows

Restricted to rows clean for both models:

| Stratum | n | students | release21 |
|---|---:|---:|---:|
| keyword, clean for both — strict | 291 | 82.5% | 81.4% |
| keyword, clean for both — accepted | 291 | 86.9% | 84.9% |
| no-keyword — strict | 326 | 55.5% | 56.7% |
| no-keyword — accepted | 326 | 57.4% | 59.2% |
| **combined weighted — strict** | 617 | **72.0%** | **71.9%** |
| **combined weighted — accepted** | 617 | **75.5%** | **74.9%** |

```
McNemar (strict, n=617): release21-only-right 42, students-only-right 41
chi2 = 0.00   (3.84 = p<0.05)   ->  no detectable difference
```

Per-topic, release21 gains `unjust law enforcement` (+21pp), `women rights` (+12) and
`climate` (+10), and loses `health care` (−13), `environment` (−12), `education` (−7) and
`pandemic` (−5). The `climate` gain against the `environment` loss is movement across an
accepted pair, not better classification.

**Interpretation.** The redesign fixed the evaluation — a defensible 72% now exists where an
indefensible 98% stood — but has not yet moved the model. This is what
`docs/current_approach.md` predicted: reviewed rows are 1.1% of the training release, so
supervision is still overwhelmingly the rules.

## 4. Annotation quality

150 rows (seed 7; 92 keyword / 58 unknown) were independently re-annotated by a separate
subagent set with no sight of the first pass.

| | n | agreement | Cohen's κ |
|---|---:|---:|---:|
| overall | 150 | 92.0% | 0.912 |
| keyword | 92 | 93.5% | 0.928 |
| unknown | 58 | 89.7% | 0.880 |

12 rows disagreed and are flagged `disputed`, carried at the pass-1 label rather than
adjudicated by a third pass. **Every figure in this document carries roughly a ±8% band from
annotator disagreement, ±10.3% in the unknown stratum.** Stratum agreement differs by 3.8pp,
which is small enough not to confound the keyword-vs-unknown comparison.

## 5. Leak audit

Blindness was instructed, not enforced, so it was verified after the fact.

| Check | Result |
|---|---|
| Gold vs hidden rule label, keyword stratum | 79.4% exact / 83.5% accepted |
| Students' self-graded ceiling for that rule | 92.7% |
| Verdict | Well below ceiling — consistent with blind reading, not recovery |
| Near-duplicates, 297 clean rows vs training | **0** at Jaccard ≥ 0.9 |

The unknown stratum is a degenerate control: `unknown` is not a taxonomy class, so agreement
with the rule is structurally zero there. Pass-1/pass-2 agreement within each stratum serves
as the control instead.

### Side finding: the trigger-word labels are ~79% correct, not 92.7%

The audit incidentally produces a blind measurement of the keyword labeller itself — the
honest version of the report's Table 2:

| Trigger-word label accuracy | strict | lenient |
|---|---:|---:|
| Students, self-graded (report Table 2) | 92.7% | 95.8% |
| **Blind, this annotation (n=514)** | **79.4%** | **83.5%** |

Roughly **21% of the training labels are wrong** — a ~13pp inflation, the same direction and
magnitude as the 77.2 → 55.0 gap on their other self-graded number.

## 6. Corrections to previously published figures

| Figure | Where | Corrected |
|---|---|---|
| 98.27% "test accuracy" | report Tables 4–5, abstract | Not accuracy. Agreement with the generating rule. |
| 92.7% trigger-word accuracy | report Table 2 | **79.4%** blind |
| 77.2% T5 on unlabelled | report Table 6 | **55.0%** blind, on that population |
| 68.5% "lenient", students | `plan.md`, `docs/current_approach.md` | **58.5%** |
| 69.0% "lenient", ours | `notebooks/verify_ours.ipynb` §Summary | **60.0%** |

The two "lenient" corrections have one cause: the legacy gold carried annotator-supplied
alternatives on **21.0%** of rows, the new gold on **0.9%**, because the annotation brief
forbade populating alternatives from the accepted-pair table (the scorer applies those pairs
itself; doing both double-counts). Re-scored on accepted pairs alone the legacy set gives
58.5%, consistent with the new no-keyword stratum at 57.4%. **The drop is a protocol
correction, not a regression.**

## 7. Known limits

- **Gold is LLM-annotated**, with LLM-measured agreement. κ=0.912 is strong, but two passes
  of the same model family can share a bias. Before this goes to Clingendael, a human should
  adjudicate at least a subsample; the pipeline supports swapping a human into step 04 with
  no other change.
- **Seven classes can never have a clean keyword score for the students' model.**
  `animal welfare`, `blm`, `culture`, `discrimination`, `housing`, `immigration`,
  `unjust law enforcement` each had a corpus pool under 3,000, so the `.head(3000)` cap never
  bit and the entire pool went into training. This is structural and permanent.
- **Nine classes have no clean keyword rows in this particular draw** — the seven above plus
  `lgbtq` and `public services`, which do have clean rows in the corpus (303 and 235) but
  none that landed here. That is a draw artifact, fixable by sampling more.
- **`other` is dead.** 22 gold rows are `other`; release21 predicts it **0 times in 840**.
  The class exists in the taxonomy and in training at 48 rows out of 46,849 (0.1%), too rare
  to learn. Those 22 rows (2.6% of the test set) are unreachable for both models.
- **The 2025-05-23 cutoff** leaves 160 manifest rows unused. They remain available for
  scoring release21 alone, and for any future model built on the newer snapshot.
- **One test set, spent once.** These annotations must not be used to fix rules or train
  anything.

## 8. What the numbers point at

The lever is supervision, not architecture. Reviewed rows are 1.1% of the training release
and `other` is 0.1%; both need an order of magnitude more before another checkpoint is worth
evaluating. The `.head(3000)` truncation — which takes the *oldest* rows per class, leaving
`labor rights` at 75% from 2020 and nothing after 2021 — is a one-line fix (`.sample(n)`) and
should land before the next training run.
