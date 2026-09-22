# Protest event theme classifier

A single-label semantic classifier for European protest-event notes. It assigns
exactly one of 21 themes using only the original `notes` text. There are no
keyword features, metadata features, rule pretraining, or fallback models.

The implementation is intentionally split into:

- `src/protest_classifier/`: reusable, path-independent library code;
- `src/protest_classifier/cli/`: thin runnable workflows with project paths
  and CLI arguments, installed as console commands (see Setup);
- `docs/prompts/annotation_prompt.md`: the exact annotation contract;
- `data/manifests/`: frozen, nested train selections plus separate development
  and locked-test selections;
- `notebooks/evaluate_classifier.ipynb`: evaluation tables, learning curves,
  per-class scores, confusion matrix, and examples.

See `docs/architecture.md` for module responsibilities and
`docs/data_provenance.md` for the input-data history. The original assignment is
kept at `Tweedejaarsproject-1.pdf`. Previous data and model artifacts are under
`data/archive/legacy_2026-09-20/`; superseded code remains available through Git.

## Setup

```bash
uv sync
```

On this machine, prefix GPU training and inference commands with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` for the Radeon 840M ROCm workaround.

## Workflow

```bash
# 1. Recreate the preserved, label-free split manifests.
.venv/bin/build-manifests

# 2. Create blank annotation batches for the 6,000-row training pool.
.venv/bin/create-annotation-chunks \
  --manifest data/manifests/train_6000.csv \
  --output data/annotations/train_chunks

# 3. After annotation, assemble all complete nested training releases.
.venv/bin/prepare-training-releases

# Assemble development labels separately. Do the same for the locked test only
# after its labels have been collected independently.
.venv/bin/assemble-annotations \
  --manifest data/manifests/dev.csv \
  --labels dev_labels.csv \
  --output data/annotations/dev.csv

# 4. Train a release. Development macro-F1 selects the best epoch.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/train-classifier \
  --release 6000 --seeds 17 42 83

# 5. Evaluate development results and plot the learning curve.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/evaluate-classifier \
  --split dev --run run-0800 run-1500 run-3000 run-6000
.venv/bin/plot-learning-curve

# 6. Evaluate the locked test once, after model selection is finished.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/evaluate-classifier \
  --split test --run run-6000

# 7. Label the full event corpus in bounded-memory chunks.
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/predict-events \
  --model artifacts/models/new_classifier/run-6000/seed-42/best
```

The defaults are tuned for this corpus and GPU: 256-token inputs, length-grouped
training batches, activation checkpointing, a physical training batch of 4
(effective batch 32 through accumulation), BF16 inference, batches of 16, and
bounded 1,024-row inference chunks. Only 0.08% of the current notes exceed 256
tokens. Use `--no-length-grouping`, `--no-gradient-checkpointing`, `--no-bf16`,
or the corresponding batch/length arguments when comparing configurations.

Training code can read training and development annotations but has no route to
the locked test labels. `evaluation/test_data.py` is the sole library loader for
test gold.

## Verification

```bash
.venv/bin/python -m unittest discover -v
```

## Review and correct annotations

Run the local generic annotation editor when labels need manual verification:

```bash
.venv/bin/annotation-review-server
```

Open `http://127.0.0.1:8765`. It discovers assembled CSV files under
`data/annotations` (not intermediate chunks), displays one item per row, and
atomically writes label/evidence edits back to the selected file. See
`docs/annotation_review.md` for using another annotation directory or port.
