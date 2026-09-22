# Trained model ledger

Checkpoints live under `models/`, which `.gitignore` keeps local. This file is
the tracked record of what was trained, on which labels, and what it scored.

Every run writes to `models/new_classifier/run-<release>/seed-<seed>/`, which is
derived from `--release` and `--seed` only. Two runs on different label sets
therefore collide on the same path and the second silently replaces the first's
`best/` and `metrics.json`. Pass `--models-dir` when training a release that has
already been trained, or copy the previous `best/` aside first.

## run-6000 / seed-42 — taxonomy v1 labels

- Weights: `models/new_classifier/run-6000/seed-42-v1-labels/` (copied aside on
  22 September 2026, before the v3 run could overwrite them; `model.safetensors`
  verified md5-identical to the original)
- Trained: 21 September 2026
- Labels: `data/annotations/` at taxonomy `2026-09-20-v1`, 21 classes, annotator
  `deepseek-flash` with 18 manually corrected dev rows. Archived as
  `data/annotation_runs/archives/annotations-v1-working-set.zip`.
- Config: ModernBERT-base, effective batch 32, length grouping, gradient
  checkpointing, BF16. 6,000 train rows, 497 dev rows, 5 truncated train notes.
- Result: **best epoch 5, dev macro-F1 0.6592, dev accuracy 0.7887.** Stopped at
  epoch 7 on patience 2.

| Epoch | macro-F1 | accuracy | loss |
| --- | --- | --- | --- |
| 1 | 0.4562 | 0.6439 | 1.239 |
| 2 | 0.5711 | 0.7606 | 0.799 |
| 3 | 0.6113 | 0.7545 | 0.825 |
| 4 | 0.6397 | 0.7807 | 0.809 |
| **5** | **0.6592** | **0.7887** | 0.914 |
| 6 | 0.6247 | 0.7807 | 1.103 |
| 7 | 0.6270 | 0.7847 | 1.093 |

## run-6000 / seed-42 — flash v3 labels

- Weights: `models/new_classifier/run-6000/seed-42/best/`
  (`model.safetensors` md5 `ba1d7b19bc0351fee06b440ce46f2dc1`)
- Trained: 22 September 2026, 5,547s (92 min). Log
  `logs/train_6000_flash_seed42.log`.
- Labels: `data/annotation_runs/deepseek-v4-1-flash/run-2026-09-21-flash-01/assembled/`,
  taxonomy `2026-09-21-v3`, 23 classes, annotator
  `deepseek-v4.1-flash-run-2026-09-21-flash-01`. Passed explicitly via
  `--train-csv` and `--dev-csv`, so `data/annotations/` was not modified.
- Config: defaults as committed in `4199615` — ModernBERT-base, 10 epochs,
  lr 2e-5, batch 4 x accum 8 (effective 32), max_length 256, patience 2, length
  grouping and gradient checkpointing on, BF16. 188 steps per epoch.
  6,000 train rows (5 truncated notes), 497 dev rows (0 truncated).
- Result: **best epoch 7, dev macro-F1 0.7274, dev accuracy 0.8652.**
  Early-stopped at epoch 9 on patience 2.

| Epoch | macro-F1 | accuracy | loss |
| --- | --- | --- | --- |
| 1 | 0.4214 | 0.6298 | 1.235 |
| 2 | 0.6669 | 0.8109 | 0.683 |
| 3 | 0.6958 | 0.8330 | 0.569 |
| 4 | 0.7062 | 0.8410 | 0.693 |
| 5 | 0.7077 | 0.8551 | 0.703 |
| 6 | 0.7062 | 0.8551 | 0.761 |
| **7** | **0.7274** | **0.8652** | 0.792 |
| 8 | 0.7224 | 0.8672 | 0.796 |
| 9 | 0.7159 | 0.8592 | 0.815 |

Loss bottomed at epoch 3 and rose thereafter while macro-F1 kept improving to
epoch 7: the model grew overconfident on common classes while still learning
rare ones, which macro-F1 weights equally. Accuracy peaked at epoch 8 (0.8672)
but selection is on macro-F1, so epoch 7 was saved.

### Comparison with the v1-label run

Same codebase, architecture, schedule and manifests; only the labels differ.

| Epoch | v1 macro-F1 | flash macro-F1 |
| --- | --- | --- |
| 1 | 0.4562 | 0.4214 |
| 2 | 0.5711 | 0.6669 |
| 3 | 0.6113 | 0.6958 |
| 4 | 0.6397 | 0.7062 |
| 5 | **0.6592** | 0.7077 |
| 6 | 0.6247 | 0.7062 |
| 7 | 0.6270 | **0.7274** |

Flash trails at epoch 1 and leads from epoch 2 onward, finishing **+6.8
macro-F1 points** and **+7.7 accuracy points** above the v1 best, on a harder
problem: 23 classes rather than 21.

### Scoring caveats

`palestine-israel conflict` (252 train rows) and `ukraine-russia war` (136) have
**zero rows in the dev split**, which covers 21 of 23 classes. Both contribute a
hard 0.000 to a macro average taken over all `CLASS_NAMES`, costing roughly 8.7
points. Over the 21 classes present, epoch 5 measured 0.7751 rather than 0.7077.
Report the 23-class figure as the headline and the 21-class one beside it.

Lenient scoring (`accepted()`: prediction matches the primary or the recorded
secondary) on the epoch-5 checkpoint gave accuracy 0.8773 and macro-F1 0.7301,
gains of +2.2 points each. Only 48 of 497 dev rows carry a secondary and 11 were
rescued by it, so the remaining 37 errors are genuine rather than ranking
differences.

Both runs are scored against their own annotator's dev labels, so each measures
agreement with that annotator's judgement rather than shared ground truth.
