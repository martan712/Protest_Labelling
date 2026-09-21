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

## run-6000 / seed-42 — flash v3 labels (in progress)

- Weights: `models/new_classifier/run-6000/seed-42/`
- Started: 22 September 2026, log `logs/train_6000_flash_seed42.log`
- Labels: `data/annotation_runs/deepseek-v4-1-flash/run-2026-09-21-flash-01/assembled/`,
  taxonomy `2026-09-21-v3`, 23 classes, annotator
  `deepseek-v4.1-flash-run-2026-09-21-flash-01`. Passed explicitly via
  `--train-csv` and `--dev-csv`, so `data/annotations/` was not modified.
- Config: defaults as committed in `4199615` — ModernBERT-base, 10 epochs,
  lr 2e-5, batch 4 x accum 8, max_length 256, patience 2, length grouping and
  gradient checkpointing on. 188 optimizer steps per epoch, ~2.3 s/step.

| Epoch | macro-F1 | accuracy | loss |
| --- | --- | --- | --- |
| 1 | 0.4214 | 0.6298 | 1.235 |

Epoch 1 sits 3.5 macro-F1 points below the v1 run's epoch 1 on a harder problem:
23 classes rather than 21, including the two classes v3 split out, which the two
annotators disagreed about most. Macro-F1 averages over class count, so the
extra sparse classes cost accuracy at this stage by construction.

The two runs are not strictly comparable. They use different taxonomies and
different dev labels, so each is scored against its own annotator's judgement
rather than against shared ground truth.
