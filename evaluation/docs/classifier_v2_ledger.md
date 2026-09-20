# Classifier v2 ledger

**Instructions: KEEP THIS VERY BRIEF. Append only. Record what happened, the evidence, and the decision it motivated. Do not turn this into a plan, report, or technical guide.**

## 2026-09-20 — Scope reset

- Decided: predict exactly one of the existing 21 themes from raw event `notes`.
- Excluded: keyword logic, metadata features, LLM fallback, multi-label output, and trend analysis.
- Motivation: measure a standalone semantic classifier against human labels.

## 2026-09-20 — Fixed comparison data

- Created: 800 training rows and 497 development rows; kept the existing 840 test rows locked.
- Decision: use the same development set for model/data choices; use test only for the final comparison.
- Motivation: prevent leakage and preserve a meaningful comparison with the previous approach.

## 2026-09-20 — First model

- Implemented: `answerdotai/ModernBERT-base`, raw notes only, 512-token maximum, 21-class softmax head.
- Motivation: a simple pretrained semantic text classifier appropriate for limited labelled data.

## 2026-09-20 — 800-label baseline, seed 42

- Result: early-stopped at epoch 8; best epoch 6 = 60.8% accuracy and 40.6% macro-F1 on development.
- Evidence: training loss reached about 0.22 while development loss stayed about 1.38; several classes have only 6–19 training examples.
- Interpretation: strong overfitting and severe rare-class data shortage.
- Decision: collect more labels and compare a 1,500-row run using the same model and development set.

## 2026-09-20 — Baseline stability check

- In progress: seeds 17 and 83 on the same 800 rows; seed 42 is preserved separately.
- Motivation: establish whether the baseline varies substantially by random seed.
- Decision: use one seed for quick later experiments; repeat seeds only when confirming a chosen model.

## 2026-09-20 — Issues found

- `metrics.json` misses results because `train.py` reads `macro_f1` instead of `eval_macro_f1`; checkpoint selection itself worked.
- `sample_more_events.py` incorrectly counts legacy/already-labelled queue rows and may sample zero new events.
- Decision: fix both before preparing and training the 1,500-row release.

## 2026-09-20 — Baseline seeds complete

- Result: seed 17 best epoch 9 = 70.6% accuracy / 49.1% macro-F1; seed 83 best epoch 4 = 64.8% / 43.2% development.
- Evidence: checkpoints under `models/classifier_v2/run-0800/seed-*`. Mean development strict accuracy across the three seeds is about 65%.
- Decision: keep collecting labels; the spread across seeds confirms the 800-row set is data-limited, not seed-limited.

## 2026-09-20 — Fixes and annotation expansion started

- Fixed: `train.py` now reads `eval_macro_f1`; `sample_more_events.queued_rows` counts only unlabelled annotation-source rows.
- Prepared: 700 already-queued `expanded_v1` rows are clean against train/dev/test (0 overlap), so labelling them yields the 1,500-row release.
- Started: LLM annotation of `expanded_v1` chunks 04–10 (1,500 release) and sampling for 3,000/6,000.
- Evidence: `evaluation/docs/classifier_v2_ledger.md`, `models/classifier_v2/run-0800/seed-*/metrics.json`.

## 2026-09-20 — Annotation expansion complete

- Result: 700 `expanded_v1` rows plus 4,500 new `sampled_v1` rows annotated by LLM subagents; 6,000 labelled rows total.
- Evidence: `data/classifier_v2/train_*.csv`; all 22 leakage/data tests pass; sampled queue had 0 ID/hash overlap with train/dev/test.
- Decision: the nested 800/1,500/3,000/6,000 releases are ready.

## 2026-09-20 — Development learning curve

- Result (seed 42 strict accuracy / macro-F1): 800 = 0.610/0.365; 1,500 = 0.686/0.567; 3,000 = 0.751/0.612; 6,000 = 0.769/0.658.
- Evidence: doubling gains +7.6, +6.4, +1.8 points; `evaluation/results/classifier_v2_learning_curve.{json,png}`.
- Decision: keep 6,000 as the winning release; the final doubling still exceeded one point, but the trend shows the next would not.

## 2026-09-20 — Fixes to selection metric

- Fixed: training selected on macro-F1 over classes present in development (19), while reporting averaged all 21; `compute_metrics` now uses the full 21-class label set.
- Evidence: two dev classes (`palestine-israel conflict`, `ukraine-russia war`) are absent from dev, so the old selection macro was inflated (0.727 vs the true 0.658).
- Decision: seeds 17/83 were retrained on 6,000 with the corrected metric; seed 42's epoch-4 checkpoint was verified best under the corrected metric.

## 2026-09-20 — Final 840-row test (spent once)

- Result: ModernBERT seeds 17/42/83 = 85.0%, 86.0%, 85.8% strict (mean 85.6%); macro-F1 81.5%; accepted 88.2%.
- Evidence: `evaluation/results/classifier_v2_test.json`; compared with Students' T5 70.1% and release21 70.2% on the same IDs.
- Interpretation: +15.5 points strict over both existing models; remaining weak classes are `discrimination` (F1 0.38) and `other` (F1 0.35).
- Decision: freeze `run-6000` ModernBERT as the new classifier; do not tune further on test.


## 2026-09-20 — 1,500-label run, seed 42

- Result: early-stopped after epoch 6; best epoch 4 = 69.0% accuracy / 59.8% macro-F1.
- Evidence: +19.2 macro-F1 points over the same-seed 800-row run (40.6%).
- Decision: more annotations clearly help; continue the learning curve without changing the model or development set.

## 2026-09-20 — Ledger correction

- Correction: seed 17's selected 800-row checkpoint was epoch 7 = 68.8% accuracy / 50.6% macro-F1; the earlier entry mixed final-epoch accuracy with macro-F1.

## 2026-09-20 — Climate, environment, and farmers review

- Climate: 29/31 strict; 31/31 when climate/environment interchange is accepted.
- Environment: 27/38 strict; 29/38 with interchange. Of 11 misses, 2 were climate, about 4 were defensible/underspecified, and 5 were clear errors.
- Farmers: 1/2 strict; all three farmer-related errors were plausible boundary cases. Two development examples are insufficient to judge this class.
- Decision: finish the 3,000/6,000 learning curve; if errors remain, add boundary examples and label the stated grievance rather than actors or nouns.

## 2026-09-20 — Students model on v2 development rows

- Compared on the 461/497 development events present in the students' snapshot; zero exact note overlap with student training data.
- Students' T5: 49.5% strict / 56.4% accepted. V2 at 1,500: 68.8% strict / 74.2% accepted on the same rows.
- Result: v2 leads by 19.3 strict and 17.8 accepted percentage points without opening the locked test set.

## 2026-09-20 — Why the old student score is higher

- The old 70.1% student score is on the 840-row test mixture: 514 keyword-matched rows plus 326 unknown rows. Student strict accuracy was 82.8% on clean keyword rows but only 55.5% on unknown rows.
- The v2 development comparison is 461 shared rows, all marked `unknown` by the old keyword system; 36 newer dev rows have no student prediction.
- Result: student T5 scores 49.5% strict / 56.4% accepted on this hard unknown-only subset. This is not inconsistent with the old test result; it is the same weakness exposed without the easy keyword stratum.

## 2026-09-20 — 3,000-label run, seed 42

- Result: selected epoch 7 = 75.05% accuracy / 67.73% macro-F1 on development.
- Evidence: epoch 1 49.5% / 19.15%; epoch 3 73.44% / 66.23%; later epochs fluctuated, with epoch 7 best.
- Decision: retain this checkpoint as the current best; continue the planned 6,000-label run before deciding whether more data helps.

## 2026-09-20 — Category review of best 3,000 model

- Climate: 30/31 strict (96.8% recall); environment: 30/38 strict (78.9% recall); farmers: 1/2 strict.
- Climate+environment: 60/69 strict (87.0%); 64/69 (92.8%) when interchange is accepted.
- Farmers precision was 25%: 1 true farmer prediction among 4 farmer predictions; three false positives involved pig owners/slaughter, resource protest, and pig-farm expansion.
- Interpretation: errors follow actor/object shortcuts instead of the underlying grievance; the farmers dev support (2) is too small for a reliable estimate.

## 2026-09-20 — 6,000-label run, seed 42 (in progress)

- Epoch 1: 63.98% accuracy / 43.98% macro-F1. Epoch 2: 74.65% / 62.07%.
- At epoch 2, climate: 30/31; environment: 25/38; farmers: 1/2. Climate+environment: 55/69 strict, 58/69 (84.1%) with interchange.
- Decision: do not compare final performance yet; the 6,000 run has not reached the 3,000 model's best epoch.

## 2026-09-20 — Training-data audit for farm boundary cases

- The training releases contain relevant counterexamples: animal welfare (pig slaughter), environment (pigsty/industrial-farming opposition), and many genuine farmer-interest protests.
- Interpretation: the distinction exists but is sparse and imbalanced; generic farmer protests substantially outnumber anti-farm environmental/animal-welfare cases. This explains persistent shortcut errors.
- Decision: add fresh, human-reviewed contrastive examples labelled by stated grievance, while keeping the current dev/test rows untouched.

## 2026-09-20 — Possible targeted extension

- Synthetic data may supplement rare boundaries (roughly 50–150 varied, human-reviewed examples), but should not replace real events or make labels explicit in wording.
- Preferred procedure: integrate approved examples into training and retrain from the pretrained base; use fine-tuning of the selected model only as an experiment, not the headline result.

## 2026-09-20 — 6,000-label epoch-4 lenient overview

- Applied the notebook rule: primary label, manual alternative, or an explicitly accepted taxonomy pair.
- Result: strict 382/497 (76.9%); lenient 401/497 (80.7%); 96 rows remain wrong leniently (19 strict errors are rescued).
- Strong classes: climate 31/31 lenient, labor rights 88/93, immigration 13/15, women rights 9/9.
- Weak/low-support classes: discrimination 6/9 lenient, culture 2/4, farmers 1/2, LGBTQ 1/2; Palestine–Israel and Ukraine–Russia have no dev rows.
- Main remaining error groups: policies & politics→other (12), policies & politics→public services (11), public services→policies & politics (7), and unjust law enforcement↔policies & politics (10 combined).
- Decision: treat broad institutional-category overlap as the main remaining problem; do not interpret tiny-class percentages as stable estimates.

## 2026-09-20 — Locked test evaluation, 6,000 seed 42

- Result on the untouched 840-row test set: 722/840 strict (85.95%), 742/840 accepted (88.33%), macro-F1 81.74%.
- Comparison: previous student T5 70.1% strict and previous release 70.2% strict; v2 is about 16 points higher.
- Strong test classes included farmers (97.9% F1), climate (96.8%), environment (89.5%), labor rights (93.8%), and women rights (94.1%). Weakest were other (35.1%) and discrimination (37.5%).
- Decision: preserve this result as the seed-42 final comparison; do not use the locked test for further model selection. Seeds 17/83 remain useful for stability, not checkpoint selection.

## 2026-09-20 — Full-corpus prediction memory fix

- Diagnosis: prediction was chunked already, but default GPU inference batches were 64 sequences × 512 tokens, causing excessive ROCm activation memory. The monitoring `pgrep` loop also matched its own command and could wait forever.
- Fix: `classifier_v2/predict.py` now defaults to batch size 8 and 512-row chunks, uses inference mode, releases batch tensors/cache, reports progress, and overwrites derived output on restart to prevent duplicate appends.
- Validation: 16-row GPU smoke test passed; all 22 classifier tests pass. The previous full-corpus CSV was partial (not safe to use) and should be regenerated with the fixed script.
