# DeepSeek annotation handoff

Annotate every event in the frozen manifests using the exact contract in
`configs/annotation_prompt.md`.

Required sets:

- `data/manifests/train_6000.csv`: 6,000 training events
- `data/manifests/dev.csv`: 497 development events
- `data/manifests/test_locked.csv`: 840 locked-test events

Use only each row's `notes`. Do not inspect or reuse anything under `archive/`,
old labels, predictions, keywords, model outputs, country, or date. Fill these
fields for every row: `primary_label`, `alternative_labels`, `other_reason`,
`evidence`, and `annotator`. Use exact label strings. Supply at most one genuine
alternative; leave it empty otherwise. `other_reason` is required only for
`other`. Set `annotator` consistently to identify the model/run.

Create and complete the files chunk by chunk:

```bash
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/train_6000.csv --output data/annotations/train_chunks
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/dev.csv --output data/annotations/dev_chunks
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/test_locked.csv --output data/annotations/test_chunks
```

Do not train a model or inspect predictions. After all chunks are complete, run:

```bash
.venv/bin/python scripts/prepare_training_releases.py
.venv/bin/python scripts/assemble_annotations.py --manifest data/manifests/dev.csv --chunks data/annotations/dev_chunks --output data/annotations/dev.csv
.venv/bin/python scripts/assemble_annotations.py --manifest data/manifests/test_locked.csv --chunks data/annotations/test_chunks --output data/annotations/test_locked.csv
.venv/bin/python -m unittest discover -v
```

Completion means the assembled training releases contain exactly 800, 1,500,
3,000, and 6,000 rows; development contains 497; locked test contains 840; and
all validation/tests pass. Report only progress, validation failures, and the
final row counts. Do not modify taxonomy, manifests, library code, or scripts.
