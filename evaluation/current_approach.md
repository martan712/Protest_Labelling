# Current training approach

This is the current post-redesign workflow as of commit `1d87b2c`.

## 1. Rebuild the candidate labels

`pipeline/rebuild_labels.py` starts from the full event snapshot, not the old balanced CSV.
It applies the versioned rules in `pipeline/finetuner/triggers.py` to every note and records:

- every matched rule and matched class;
- an unambiguous `rule_label`, or an unresolved label when rules conflict or match nothing;
- the original label, date, year, country, note hash, and duplicate-group ID;
- the source and status of the final label.

The corrected rules include the known labor, environment, nursery, spelling-variant, and
participant-versus-grievance cases. Rule conflicts are not silently resolved by dictionary order.

## 2. Add blind reviewed examples

The first reviewed training set contains 500 events selected separately from the held-out data.
The selection was approximately:

- 300 representative random events;
- 200 targeted events covering labor, energy, police-as-employees, student protests,
  war/peace lookalikes, supported-topic lookalikes, and thin-motive notes.

Luna annotated these events using only the note, date/year, country, and taxonomy guide. The
annotations include a primary label, independently supported alternatives, evidence, an
`other_reason` where needed, and taxonomy metadata.

These are currently marked `independent_review` and `independent_llm_review`. They are useful
experimental supervision, but are not being represented as human-adjudicated ground truth.

The annotations are stored locally in the ignored review artifacts and are reapplied by the label
rebuild through the reviewed-label ledger. Evaluation labels are kept out of this training path.

## 3. Select the training release

`pipeline/select_year_quota.py` removes the held-out event IDs and duplicate groups, then selects
up to 3,000 examples per class with explicit year coverage. It samples across countries within
each class-year cell and writes coverage reports.

The current selected release contains:

- 46,849 training rows;
- all 21 classes, including `other`;
- 500 independently reviewed rows;
- 46,349 weak rule-labelled rows;
- 48 `other` rows.

Thus the reviewed examples currently make up about 1.1% of the full training release. This is
the central remaining limitation: the model still receives most of its supervision from rules.

## 4. Train the model

`pipeline/train_release.py` trains a 21-class T5 sequence classifier. Each input is structured as:

```text
Event year: <year>
Country: <country>
Note: <original note>
```

The model uses the same 21-class mapping for every training mode and stores the mapping in its
checkpoint configuration.

Three modes are available:

- `full`: train on the complete selected release;
- `reviewed`: train only on independently reviewed examples;
- `mixed`: construct a fixed-size dataset with a configurable reviewed fraction, oversampling
  reviewed rows when necessary.

The full-data run uses the rule-labelled release as a warm-up. Reviewed-only and mixed runs start
from that checkpoint and use a lower learning rate for fine-tuning.

The default maximum is 20 epochs with patience 3. After every epoch, the checkpoint is evaluated
on the separate reviewed development set. The saved model is the epoch with the best accepted-pair
development accuracy, not the epoch with the best weak-label loss.

## 5. Evaluation

The 500-event development set is separate from the 500 reviewed training events. Its labels are
also blind Luna annotations and are used only for checkpoint selection and comparison.

The important metrics are:

- strict accuracy: exact primary-label match;
- accepted-pair accuracy: exact match or one of the explicitly allowed symmetric taxonomy pairs.

The weak internal validation/test split is retained only as a diagnostic. It shares the rule-label
generation process with training and therefore measures how well the model reproduces the weak
labeler, not how well it understands the grievance.

For the completed one-epoch T5-base full-data run, the results were:

| Evaluation | Strict | Accepted |
|---|---:|---:|
| Reviewed 500-event dev set | 64.0% | 68.8% |
| Weak internal validation split | 98.53% | not applicable |
| Weak internal test split | 98.12% | not applicable |

The 98% figures are not comparable to the reviewed-dev score: they measure reproduction of the
keyword/rule labels, while the reviewed dev score measures semantic grievance classification.

## 6. Current experiments

The immediate follow-up experiments are:

1. fine-tune the one-epoch full-data checkpoint on the 500 reviewed training rows;
2. compare that with a 50/50 reviewed–weak mixture;
3. expand the reviewed training set by 1,000 blind examples and rerun the better recipe.

The additional 1,000-event queue is not part of the current 46,849-row release until its
annotations are complete and the candidate labels are rebuilt.

## Important interpretation

The redesign has fixed the main evaluation problem and made reviewed supervision possible, but it
has not yet solved training-data dominance by weak labels. A materially improved model must either
give reviewed rows much more influence through fine-tuning/mixed sampling or expand the reviewed
training set. More epochs on the current full release alone would mainly improve reproduction of
the weak keyword labels.
