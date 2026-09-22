# Active data layout

`data/raw/events.csv` is the source event table. Only its `notes` text is a
model input; the other columns are retained for sampling and auditability.

- `data/manifests/`: frozen event selections. They preserve classifier-v2's
  nested train/dev/test split so results remain comparable, but contain no old
  labels. The 497-row development set and 840-row test set remain separate.
- `data/annotations/`: human/LLM-reviewed labels using the active taxonomy.
- `artifacts/models/new_classifier/`: checkpoints (ignored by Git).
- `artifacts/reports/`: evaluation JSON, plots, and error reviews.

Old labels and derived predictions are in the dated archive and must not be
joined into the active annotations.

Runnable project workflows live in `src/protest_classifier/cli/` and are
installed as console commands; reusable implementation lives alongside them in
the `src/protest_classifier/` package.

All generated output — checkpoints, evaluation reports, and training logs —
lands under `artifacts/`, which `.gitignore` keeps local in full.
