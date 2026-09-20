# classifier_v2 — standalone ModernBERT protest classifier

One softmax classifier that reads an ACLED protest `notes` string and predicts
exactly one of the 21 classes in [`evaluation/taxonomy.py`](../evaluation/taxonomy.py).

There are **no keyword labels, no keyword features, no rule pretraining, no LLM
fallback, no metadata (country/year/actor) and no `clean_notes`**. The only model
input is the original note text.

## Files

| File | Purpose |
| --- | --- |
| `common.py` | Shared contract: label maps, note normalization, note hashing, validation, leakage checks, split loaders |
| `prepare_training_data.py` | Build the nested training releases and `dev_497.csv` |
| `sample_more_events.py` | Sample unused events from `data/filtered_events.csv` into a blind annotation queue |
| `train.py` | Train ModernBERT with the Hugging Face `Trainer` |
| `evaluate.py` | Score development (`--split dev`) or the locked test (`--split test`) |
| `predict.py` | Label the full event corpus |
| `learning_curve.py` | Turn the development report into the learning-curve JSON + PNG |

Generated artefacts (git-ignored because `data/` and `models/` are):

```text
data/classifier_v2/train_0800.csv … train_6000.csv
data/classifier_v2/dev_497.csv
models/classifier_v2/run-<size>/seed-<seed>/best/
evaluation/results/classifier_v2_dev.json
evaluation/results/classifier_v2_test.json
evaluation/results/classifier_v2_learning_curve.{json,png}
```

## Data splits

| Split | Rows | Source |
| --- | ---: | --- |
| Train | 800 → 1,500 → 3,000 → 6,000 (nested) | labelled semantic annotation chunks |
| Development | 497 | `data/run/dev_gold.csv` minus `DEU24281`, `FRA33651`, `ITA21511` |
| Test | 840 | `evaluation/manifests/test_manifest_840.csv` + `data/review/test_gold_v1/test_gold.csv` |

Every larger release strictly contains the smaller one. `prepare_training_data.py`
refuses to write anything if a training ID or note hash appears in development or
the frozen test manifest, if a label is outside the taxonomy, if `other` lacks a
reason, or if a note is empty.

## Workflow

```bash
# 0. Build data (reads the locked dev/test manifests for leakage checks only)
.venv/bin/python classifier_v2/prepare_training_data.py

# 1. Add annotations when a larger release is wanted.  The 1,500 release reuses
#    the existing expanded_v1 queue; larger releases need new sampling.
.venv/bin/python classifier_v2/sample_more_events.py --target 6000

# 2. Train (all requested seeds on one release)
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python classifier_v2/train.py --release 6000 --seeds 17 42 83

# 3. Development evaluation + learning curve
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python classifier_v2/evaluate.py --split dev --run run-0800 run-1500 run-3000 run-6000
.venv/bin/python classifier_v2/learning_curve.py

# 4. Final test evaluation, spent once
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python classifier_v2/evaluate.py --split test --run run-6000

# 5. Label the whole corpus
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python classifier_v2/predict.py --model models/classifier_v2/run-6000/seed-42/best
```

`train.py` reads only its training release and the development file, never the test
gold — the only place that reads `data/review/test_gold_v1/test_gold.csv` is
`evaluate.py --split test` (and the leakage tests).
`tests/test_classifier_v2_leakage.py` enforces this statically.

### GPU note (this machine)

The ROCm build reports `HIP error: invalid device function` for gfx1153. Setting
`HSA_OVERRIDE_GFX_VERSION=11.0.0` makes the Radeon 840M usable. Training falls back
to CPU automatically if CUDA/HIP is unavailable, but far more slowly.

## Training recipe

| Setting | Value |
| --- | --- |
| Base model | `answerdotai/ModernBERT-base` (21-label head) |
| Loss | cross-entropy (no class weights) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Effective batch size | 32 (8 × 4 grad-accum) |
| Max epochs | 10, early-stopping patience 2 |
| Warm-up | 10 % of steps |
| Max tokens | 512, dynamic per-batch padding |
| Selection metric | development macro-F1 over all 21 classes |
| Seeds | 17, 42, 83 |

Preprocessing only replaces invalid control characters and normalizes whitespace;
wording, punctuation and case are preserved. All released notes tokenize below 512
tokens (0 truncated in every run).

## Results

Development strict accuracy / macro-F1 (seed 42, 497 rows):

| Train rows | Strict acc. | Macro-F1 |
| ---: | ---: | ---: |
| 800 | 0.610 | 0.365 |
| 1,500 | 0.686 | 0.567 |
| 3,000 | 0.751 | 0.612 |
| 6,000 | 0.769 | 0.658 |

The three-seed 6,000 model on the 840-row test set is the final comparison:

| Model | Strict accuracy (unweighted) |
| --- | ---: |
| Students' T5 | 70.1 % |
| release21 | 70.2 % |
| **ModernBERT (mean of seeds 17/42/83)** | **85.6 %** |

Test macro-F1 81.5 %, accepted-pair accuracy 88.2 % (secondary, never used for
selection). `other` recall is limited (0.32–0.55 across seeds) and
`discrimination` remains weak; see `evaluation/results/classifier_v2_test.json`.

## Tests

```bash
.venv/bin/python -m unittest tests.test_classifier_v2_data tests.test_classifier_v2_leakage
```
