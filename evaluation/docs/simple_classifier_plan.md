# Simple semantic protest classifier — implementation plan

This is the implementation plan to follow. It supersedes the modeling design in
`from_scratch_approach.md`.

## Goal

Train one standalone classifier that reads an ACLED protest note and predicts
exactly one of the existing 21 classes in `evaluation/taxonomy.py`.

No keyword labels, keyword features, rule pretraining, LLM fallback, trend
analysis, evidence extractor, or multi-label model.

## Data split

Use three strictly separate sets:

| Set | Rows | Use |
| --- | ---: | --- |
| Training annotations | Start with 800; expand as needed | Fit model weights |
| Development | 497 | Choose epoch and training-set size |
| Test | 840 | Final comparison, once |

### Training data

The 800 completed semantic training annotations are in:

- `data/review/unknown_starter_v1/chunks/chunk_01.csv` through `chunk_05.csv`;
- `data/review/expanded_v1/chunks/chunk_01.csv` through `chunk_03.csv`.

All 800 have labels assigned from the note and taxonomy guide. Combine them,
validate them, and deduplicate by `event_id_cnty`.

Create additional training annotations by sampling unused events uniformly at
random from `data/filtered_events.csv`. Exclude all development IDs, test IDs,
existing training IDs, and duplicate note hashes. Give the sampled original
`notes` and `evaluation/taxonomy_guide.md` to the LLM annotator. Do not expose
old classes, keywords, `clean_notes`, or model predictions.

Create nested training releases of:

```text
800 → 1,500 → 3,000 → 6,000
```

Every larger release must contain all rows from the smaller release. Do not
assume 6,000 is necessary: the development learning curve decides when to stop.

Required annotation columns:

```text
event_id_cnty, notes, note_hash, primary_label, other_reason, reviewer
```

`primary_label` must be one of the 21 classes. `other_reason` is required only
for `other`.

### Development data

Start from `data/run/dev_gold.csv`. Remove these three IDs because they also
occur in the test set:

```text
DEU24281
FRA33651
ITA21511
```

This leaves 497 development rows. Use them for epoch selection and for comparing
the 800, 1,500, 3,000, and 6,000 training runs.

### Test data

Keep these files completely outside training and model selection:

```text
evaluation/manifests/test_manifest_840.csv
data/review/test_gold_v1/test_gold.csv
```

Do not read test labels from training or development code. After the complete
recipe is selected, score the final model once and compare it with the existing
students and release21 results.

## Model

Use:

```text
answerdotai/ModernBERT-base
```

Load it with `AutoModelForSequenceClassification`:

```python
AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=21,
    id2label=id2label,
    label2id=label2id,
)
```

This is a normal single-label softmax classifier. Every event receives exactly
one label, including `other` where appropriate.

## Input and preprocessing

The model input is only the original `notes` string.

Do not include:

- country, year, actor, or other metadata;
- `clean_notes`;
- old labels or predictions;
- keyword matches.

Preprocessing:

1. reject empty notes;
2. replace invalid control characters and normalize whitespace;
3. otherwise preserve the original wording, punctuation, and case;
4. tokenize with the ModernBERT tokenizer;
5. truncate at 512 tokens;
6. use dynamic padding per batch.

More than 99% of the current notes are below 115 words, so a 512-token limit is
sufficient for this experiment. Record how many notes are truncated.

## Training

Use Hugging Face `Trainer` with ordinary cross-entropy loss.

Initial settings:

| Setting | Value |
| --- | --- |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Effective batch size | `32` |
| Maximum epochs | `10` |
| Early-stopping patience | `2` |
| Warm-up | `10%` of steps |
| Maximum tokens | `512` |
| Selection metric | development macro-F1 |
| Seeds | `17`, `42`, `83` |

Save the best development checkpoint for each seed. Report mean and range over
the three seeds. Do not add class weighting unless the unweighted 1,500-row run
shows clear majority-class collapse on development data.

Train the same configuration on each nested release:

```text
run-0800
run-1500
run-3000
run-6000
```

Stop adding annotations when either:

- development strict accuracy reaches the required target; or
- doubling the training set improves mean development strict accuracy by less
  than one percentage point.

## Evaluation

For development and final test, report:

- strict accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- three-seed results;
- number of `other` predictions and `other` recall.

Accepted-pair accuracy may be reported as a secondary legacy comparison, but it
must not select the model.

The final test report must compare the new model with the existing results on
the same 840 IDs:

| Model | Existing strict result |
| --- | ---: |
| Students' T5 | 70.1% unweighted |
| release21 | 70.2% unweighted |
| New ModernBERT | fill after final evaluation |

## Files to implement

Keep the new classifier separate from the legacy pipeline:

```text
classifier_v2/
├── prepare_training_data.py
├── sample_more_events.py
├── train.py
├── evaluate.py
├── predict.py
└── README.md
tests/
├── test_classifier_v2_data.py
└── test_classifier_v2_leakage.py
```

Expected generated files:

```text
data/classifier_v2/train_0800.csv
data/classifier_v2/train_1500.csv
data/classifier_v2/train_3000.csv
data/classifier_v2/train_6000.csv
data/classifier_v2/dev_497.csv
models/classifier_v2/<run-name>/
evaluation/results/classifier_v2_dev.json
evaluation/results/classifier_v2_test.json
```

## Required safeguards

Automated checks must fail when:

- a training ID or note hash occurs in development or test;
- a development ID or note hash occurs in test;
- a label is outside `evaluation.taxonomy.CLASS_NAMES`;
- a training release is not a strict superset of the previous release;
- a note is empty;
- the label mapping differs between training and inference.

Training code must not accept the test-gold path as an argument. Only
`evaluate.py --split test` may read the locked test gold.

## Implementation order

1. Build and validate `train_0800.csv` and `dev_497.csv`.
2. Add leakage and label-mapping tests.
3. Implement ModernBERT training and development evaluation.
4. Train the 800-row baseline with seed 42 as a smoke test.
5. Run the full three-seed 800-row experiment.
6. Generate annotation queues for 1,500, 3,000, and 6,000 rows as needed.
7. Train each completed nested release and plot the development learning curve.
8. Freeze the winning recipe.
9. Evaluate the three final-seed checkpoints on the 840-row test once.
10. Write the head-to-head comparison report.

## Definition of done

The project is complete when:

- one command trains the classifier from a selected training release;
- one command labels all 218,108 events;
- the training/dev/test leakage tests pass;
- the learning curve states how many annotations were actually necessary;
- the final 840-row result is reproducible and directly comparable with both
  existing models;
- no keyword-derived value is used anywhere in the new model's training or
  inference path.
