# evaluation

The frozen evaluation boundary is in `manifests/`: the legacy 200 reviewed events are dev
data, while `test_manifest.csv` contains 1,000 fresh events from the fixed snapshot. The
test annotation queue is intentionally separate from all training candidates.

Two notebooks that check predicted topic labels against the 200 manually labeled random
`unknown` events in `data/manual_labelled_data/random_unknown_labeled.csv`. Each one scores
the labels, reviews every error by hand, and explains the real errors with checks on all
unknown events.

- `verify_students.ipynb`: the students' labels
  (`data/filtered_events_class_with_predicted_students.csv`).
- `verify_ours.ipynb`: our T5 labels (`data/filtered_events_class_with_predicted.csv`, from
  `models/attempt7-t5-leave-keywords-in-epoch-1`). It also re-runs the model with words removed
  to confirm causes, so it needs that checkpoint.
- `students_error_review.csv`, `ours_error_review.csv`: the hand-review verdict per error
  (wrong, defensible, or no fitting class) with a one-line reason. Each notebook asserts its
  file matches the errors.
- `improvement_plan.md`: a step-by-step plan to improve the model, based on both notebooks.

Run from this folder with the project venv:

```
../.venv/bin/python -m nbconvert --to notebook --execute --inplace verify_students.ipynb
../.venv/bin/python -m nbconvert --to notebook --execute --inplace verify_ours.ipynb
```
