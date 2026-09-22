# Architecture

Reusable, path-independent code lives under `src/protest_classifier`. Project
paths, argument parsing, terminal output, and artifact writing belong to the
thin runners under `scripts`.

```text
scripts
  └─ data / modeling / evaluation
       └─ taxonomy / text / data contracts
```

## Library

- `taxonomy.py`: canonical annotation contract.
- `text.py`: note normalization and hashing.
- `data/contracts.py`: schemas, nesting, and leakage checks.
- `data/manifests.py`: select and validate event manifests.
- `data/annotations.py`: annotation queues and manifest-label assembly.
- `data/training_sets.py`: training/development loading; never test gold.
- `modeling/datasets.py`: tokenizer-to-PyTorch adapters.
- `modeling/training.py`: train one seed and save its checkpoint.
- `modeling/checkpoints.py`: shared label mapping and checkpoint validation.
- `modeling/inference.py`: batch and memory-bounded streaming inference.
- `evaluation/test_data.py`: the only library module that opens test gold.
- `evaluation/metrics.py`: pure scoring functions.
- `evaluation/learning_curve.py`: pure aggregation across releases/seeds.

## Runners

Scripts compose the library into the concrete project workflow. They may know
the repository layout; library functions do not. Training runners cannot import
`evaluation.test_data`, which keeps the locked test inaccessible during model
selection.
