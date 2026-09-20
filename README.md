# Protest event theme classifier

A single-label semantic classifier for European protest-event notes. It assigns
exactly one of 21 themes using only the original `notes` text. There are no
keyword features, metadata features, rule pretraining, or fallback models.

The implementation is intentionally split into:

- `src/protest_classifier/`: reusable, path-independent library code;
- `scripts/`: thin runnable workflows with project paths and CLI arguments;
- `configs/annotation_prompt.md`: the exact annotation contract;
- `data/manifests/`: frozen, nested train selections plus separate development
  and locked-test selections;
- `notebooks/evaluate_classifier.ipynb`: evaluation tables, learning curves,
  per-class scores, confusion matrix, and examples.

See `docs/architecture.md` for module responsibilities and
`docs/data_provenance.md` for the input-data history. The original assignment is
kept at `Tweedejaarsproject-1.pdf`. Previous data and model artifacts are under
`archive/legacy_2026-09-20/`; superseded code remains available through Git.

## Setup

```bash
uv sync
```

On this machine, prefix GPU training and inference commands with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` for the Radeon 840M ROCm workaround.

## Workflow

```bash
# 1. Recreate the preserved, label-free split manifests.
.venv/bin/python scripts/build_manifests.py

# 2. Create blank annotation batches for the 6,000-row training pool.
.venv/bin/python scripts/create_annotation_chunks.py \
  --manifest data/manifests/train_6000.csv \
  --output data/annotations/train_chunks

# 3. After annotation, assemble all complete nested training releases.
.venv/bin/python scripts/prepare_training_releases.py

# Assemble development labels separately. Do the same for the locked test only
# after its labels have been collected independently.
.venv/bin/python scripts/assemble_annotations.py \
  --manifest data/manifests/dev.csv \
  --labels dev_labels.csv \
  --output data/annotations/dev.csv

# 4. Train a release. Development macro-F1 selects the best epoch.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python scripts/train_classifier.py \
  --release 6000 --seeds 17 42 83

# 5. Evaluate development results and plot the learning curve.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python scripts/evaluate_classifier.py \
  --split dev --run run-0800 run-1500 run-3000 run-6000
.venv/bin/python scripts/plot_learning_curve.py

# 6. Evaluate the locked test once, after model selection is finished.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python scripts/evaluate_classifier.py \
  --split test --run run-6000

# 7. Label the full event corpus in bounded-memory chunks.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python scripts/predict_events.py \
  --model models/new_classifier/run-6000/seed-42/best
```

Training code can read training and development annotations but has no route to
the locked test labels. `evaluation/test_data.py` is the sole library loader for
test gold.

## Verification

```bash
.venv/bin/python -m unittest discover -v
```
