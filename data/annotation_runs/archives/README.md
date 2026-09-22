# Annotation archives

Zipped snapshots of every annotation set produced for this project. Unpack with
`unzip <archive> -d <destination>`.

## Current runs (taxonomy `2026-09-21-v3`)

Two independent runs over the frozen manifests, using the byte-identical prompt
`docs/prompts/annotation_prompt.md` (SHA-256
`19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250`) and
differing only in the annotating model.

| Archive | Model | Annotator string |
| --- | --- | --- |
| `run-2026-09-21-flash-01.zip` | DeepSeek-V4.1-Flash (`deepseek-flash`) | `deepseek-v4.1-flash-run-2026-09-21-flash-01` |
| `run-2026-09-21-pro-01.zip` | DeepSeek-V4-Pro-0813 (`deepseek-v4-pro`) | `deepseek-v4-pro-run-2026-09-21-pro-01` |

Each holds `assembled/` (train_0800, train_1500, train_3000, train_6000, dev,
test_locked), the prompt snapshot and hash, and provenance and review notes.
Intermediate `*_chunks/` files are excluded: the assembled CSVs are those same
rows joined to the manifests.

Agreement between them, from `src/protest_classifier/cli/compare_annotations.py`:

| Split | Strict | Lenient | Cohen's kappa |
| --- | --- | --- | --- |
| dev | 90.74% | 94.37% | 0.8968 |
| test_locked | 91.31% | 93.21% | 0.9052 |
| train_6000 | 93.17% | 95.25% | 0.9256 |

Flash is the working set. The two disagree mainly at the `other` boundary, in
opposite directions: flash under-uses the group/institution fallback, pro
over-uses it. The ~545 strict disagreements on train_6000 are not adjudicated.
See `notebooks/compare_annotators.ipynb`.

## Superseded

`annotations-v1-working-set.zip` — the full contents of `data/annotations/` as
of 21 September 2026: assembled splits plus `dev_chunks/`, `test_chunks/` and
`train_chunks/`. Labels are taxonomy `2026-09-20-v1`, which v3 supersedes: it
renamed `blm` to `racism`, removed `anti-government and anti-establishment`, and
added three classes. **`dev.csv` contains 18 rows with `annotator=manual`** —
hand-corrected labels that exist nowhere else. Archived in full for that reason.

`run-2026-09-21-v3-flash.zip` — the first v3 flash run. Its labels are complete
and schema-valid, but per-row write provenance cannot be attributed to a single
writer: an uncoordinated second process wrote into the same run directory
mid-run, and two subagents observed their assigned chunks being rewritten. See
`progress.md` and `integrity_snapshot.json` inside the archive. Kept as a record
of why the current handoffs give each model its own output root; not for use as
training data.

Older material lives outside this directory: `data/archive/legacy_2026-09-20/`
(classifier-v2 era, kept local by `.gitignore` with only its metadata tracked)
and `data/archive/jun17/research/`, whose two CSVs are tracked directly.

The June 17 full-corpus predictions are tracked as
`data/archive/jun17/data/filtered_events_class_with_predicted.zip` (36 MB,
holding a 173 MB CSV). The unpacked CSV exceeds GitHub's 100 MB file limit, so
`.gitignore` keeps `data/archive/jun17/data/` local except for `*.zip`. Unpack
with `unzip <archive> -d <destination>` as above.
