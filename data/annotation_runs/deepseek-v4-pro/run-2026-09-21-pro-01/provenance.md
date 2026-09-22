# Run provenance — run-2026-09-21-pro-01

## Run identity
- run_id: `run-2026-09-21-pro-01`
- annotator (set on every row): `deepseek-v4-pro-run-2026-09-21-pro-01`
- output_root: `data/annotation_runs/deepseek-v4-pro/run-2026-09-21-pro-01/`

## Model
- requested_model_string: `deepseek-v4-pro`
- requested_model_name: DeepSeek-V4-Pro-0813
- reported_model_identity: `deepseek-v4-pro` (model ID reported by the runtime: `deepseek/deepseek-v4-pro`)
- NOT used (would duplicate the first run): `deepseek-flash`, `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`

## Prompt
- prompt_source: `docs/prompts/annotation_prompt.md`
- prompt_snapshot: `prompt_snapshot.md` (byte-identical copy of source)
- prompt_sha256: `19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250`
- taxonomy_version: `2026-09-21-v3`

## Generation settings
- instructions: unchanged `docs/prompts/annotation_prompt.md` sent verbatim as the annotation instruction for every batch
- batch_input: `event_id_cnty` + `notes` only
- batch_size: 25-50 events (chunk-size 50)
- temperature: fixed low (provider default)
- thinking: not applicable to this runtime

## Scope
- train_6000.csv: 6000 events
- dev.csv: 497 events
- test_locked.csv: 840 events
- total: 7337 events

## Ownership
Exclusive ownership of this root is a hard requirement. Only this run's
coordinator and its own subagents (all using `deepseek-v4-pro`) may write here.
Any file found here not written by this run's processes must be reported
immediately. No foreign files were observed during this run.

## Completion — run-2026-09-21-pro-01

- pilot: 100 training rows (80 random seed=42 + 20 boundary cases), validated,
  reused in the training set. See pilot_review_notes.md for unresolved issues.
- workers: 8 subagents (all `deepseek-v4-pro`), exclusive chunk files, each
  following read-annotate-write-validate per batch.
- all 147 chunks validate PASS (validate_chunks.py): train 120, dev 10, test 17.
- assembled under `assembled/`:
  - train_0800.csv = 800 rows
  - train_1500.csv = 1,500 rows
  - train_3000.csv = 3,000 rows
  - train_6000.csv = 6,000 rows
  - dev.csv = 497 rows
  - test_locked.csv = 840 rows
  - total = 7,337 rows; annotator set to `deepseek-v4-pro-run-2026-09-21-pro-01` on every row.
- `python -m unittest discover -v`: 11 tests OK.
- prompt_sha256 (prompt_snapshot.md) = `19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250` (matches frozen prompt).
- No writes to data/annotations/, data/manifests/, deepseek-v4-1-flash/, or run-2026-09-21-v3-flash/.
