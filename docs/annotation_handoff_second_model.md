# Independent second-model annotation handoff — DeepSeek V4-Pro

This handoff covers exactly **one** annotator model: **DeepSeek-V4-Pro-0813**,
API model string `deepseek-v4-pro`. Do not use any other model for this run.

It produces an independent annotation set to compare against the first
annotator, DeepSeek-V4.1-Flash, which runs under
`docs/annotation_handoff.md`. That run is not your job and not your directory.
Read this file and `configs/annotation_prompt.md` completely before starting.
Follow the exact same taxonomy and classification instructions as the first run
so the two annotation sets can be compared.

## Model — fixed, not a choice

- Model: DeepSeek-V4-Pro-0813
- API model string: `deepseek-v4-pro`
- This is a genuinely different model from the first run's `deepseek-flash`
  (DeepSeek-V4.1-Flash). Do not use `deepseek-flash`, and do not use the retired
  aliases `deepseek-v4-flash` / `deepseek-v4-flash-vision-exp` — those route to
  V4.1-Flash and would make this run a duplicate of the first, not a comparison.
- V4-Pro is roughly four times the price of Flash per token. Size batches and
  concurrency with that in mind; measure on the pilot before scaling up.

Verify the provider's actual model identity at launch and record what the API
reports; aliases and routing can change. If `deepseek-v4-pro` is unavailable,
stop and report that the different-model requirement is unmet. Do not substitute
another model and do not misrepresent which model produced the labels.

## Output directory — exclusive to this run

Your output root is exclusively:

`data/annotation_runs/deepseek-v4-pro/RUN_ID/`

Replace `RUN_ID` with a unique run identifier (suggested form
`run-YYYY-MM-DD-pro-NN`). All chunks, assembled CSVs, pilot files, progress
records, review notes and the prompt snapshot for this run live under that root.

Do not write to `data/annotations/`, to the frozen manifests, to
`data/annotation_runs/deepseek-v4-1-flash/` (the first model's root), or to any
other annotation run. `data/annotation_runs/run-2026-09-21-v3-flash/` is a
superseded earlier run with known write-provenance contamination: do not read
it, resume it, or write into it. Do not publish over the first annotation set.

If any process other than your own subagents writes into your output root, stop,
record it, and report it. Exclusive ownership of this directory is a hard
requirement — a previous run failed on exactly this point.

## Provenance to record

Set `annotator` on every row to `deepseek-v4-pro-RUN_ID`, consistently, with
`RUN_ID` substituted. Record the actual model/provider/version as the API
reports it, generation settings, run ID, taxonomy version, the exact prompt
snapshot and its SHA-256 hash.

Use the same frozen prompt as the first run: `configs/annotation_prompt.md`,
taxonomy `2026-09-21-v3`, SHA-256
`19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250`. If the hash
differs, report the mismatch before annotating: different instructions would
confound a comparison of models. Check the first run's provenance only; do not
read its labels or evidence.

## Scope and independence

Annotate every event in:

- `data/manifests/train_6000.csv`: 6,000 training events
- `data/manifests/dev.csv`: 497 development events
- `data/manifests/test_locked.csv`: 840 locked-test events

Use only each event's notes for classification. Event IDs identify records.
Do not consult previous annotations, manual labels, predictions, error reports,
or archives. Do not use another annotator's decisions as hints. Do not train a
model, change the taxonomy, edit source notes, or alter frozen manifests.

For each model request, send the unchanged annotation prompt as instructions,
plus the run identifier and a single batch containing only `event_id_cnty` and
`notes`. Return the CSV schema specified by the prompt: `primary_label`,
`alternative_labels`, `other_reason`, `evidence`, `annotator`. Use exact label
strings. Supply at most one genuine alternative; leave it empty otherwise.
`other_reason` is required only for `other`. Do not use keyword rules to
generate labels.

## Pilot and strict subagent workflow

Start with a 100-row training pilot, including a reproducible random sample and
boundary cases selected from the notes without consulting existing labels.
Validate format and semantic consistency before continuing. Record unresolved
interpretation issues separately; do not invent or modify rules. Then process
the remaining rows in bounded batches of 25–50 events, validating each
immediately and retrying only missing or invalid rows. Do not mix records across
splits.

You may use subagents. Every subagent uses `deepseek-v4-pro` — the same single
model as the coordinator — and the same frozen prompt, taxonomy version and run
identifier. Assign exclusive batch files to each worker. Every subagent must
read the full annotation prompt and then strictly follow this cycle:

1. Read ONE assigned batch of at most 50 events.
2. Annotate every event in that batch.
3. Write the completed batch to its assigned file under this run's output root.
4. Validate the saved batch and repair any errors.
5. ONLY THEN read the next batch.

Do not preload multiple batches, read the whole dataset into a worker's context,
or hold annotations for a later combined write. Parallelism is across workers;
each worker completes its read-annotate-write-validate cycle sequentially.
Return file paths and counts to the coordinator, not entire annotation batches.

The coordinator tracks ownership and validated completion. Retry only failed or
incomplete batches. On resume, reuse only validated files from this exact run
with the same model and prompt. Never recreate queues over completed files.
Reuse validated pilot annotations when completing the training set.

## Queue creation

Replace `RUN_ID` in every command. Run queue creation only for a fresh run,
before annotations have been written.

```bash
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/train_6000.csv --output data/annotation_runs/deepseek-v4-pro/RUN_ID/train_chunks --chunk-size 50
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/dev.csv --output data/annotation_runs/deepseek-v4-pro/RUN_ID/dev_chunks --chunk-size 50
.venv/bin/python scripts/create_annotation_chunks.py --manifest data/manifests/test_locked.csv --output data/annotation_runs/deepseek-v4-pro/RUN_ID/test_chunks --chunk-size 50
```

Validate each batch for exact expected IDs, no missing/duplicate rows, allowed
labels, valid other reasons, and nonempty evidence and annotator. Recheck each
other prediction against the prompt's fallback and outside-taxonomy rules.
Record review counts and unresolved cases separately from the annotation CSV.

## Assembly and completion

After all batches are saved and validated:

```bash
.venv/bin/python scripts/prepare_training_releases.py --chunks data/annotation_runs/deepseek-v4-pro/RUN_ID/train_chunks --output data/annotation_runs/deepseek-v4-pro/RUN_ID/assembled
.venv/bin/python scripts/assemble_annotations.py --manifest data/manifests/dev.csv --chunks data/annotation_runs/deepseek-v4-pro/RUN_ID/dev_chunks --output data/annotation_runs/deepseek-v4-pro/RUN_ID/assembled/dev.csv
.venv/bin/python scripts/assemble_annotations.py --manifest data/manifests/test_locked.csv --chunks data/annotation_runs/deepseek-v4-pro/RUN_ID/test_chunks --output data/annotation_runs/deepseek-v4-pro/RUN_ID/assembled/test_locked.csv
.venv/bin/python -m unittest discover -v
```

Do not treat a zero exit status as completion. `prepare_training_releases.py`
skips any release whose labels are incomplete, prints `not yet complete: [...]`,
and still exits 0, and blank-label rows are silently dropped when chunks are
read. Before reporting completion, assert the counts explicitly:

```bash
.venv/bin/python - <<'CHECK'
import pandas as pd, sys
root = "data/annotation_runs/deepseek-v4-pro/RUN_ID"
expected = {"train_0800.csv": 800, "train_1500.csv": 1500, "train_3000.csv": 3000,
            "train_6000.csv": 6000, "dev.csv": 497, "test_locked.csv": 840}
bad = []
for name, n in expected.items():
    try:
        rows = len(pd.read_csv(f"{root}/assembled/{name}", dtype=str))
    except FileNotFoundError:
        bad.append(f"{name}: MISSING"); continue
    if rows != n:
        bad.append(f"{name}: {rows} rows, expected {n}")
print("INCOMPLETE:\n" + "\n".join(bad) if bad else "All six outputs complete.")
sys.exit(1 if bad else 0)
CHECK
```

Report the row counts this prints. If any line is short or missing, the run is
not finished: find the unfilled chunks, complete them, and reassemble. Every one
of the 6,000 train, 497 dev and 840 test events must carry a label; partial
coverage is a failed run, not a smaller run.

Completion requires validated nested training releases containing exactly 800,
1,500, 3,000 and 6,000 rows, development containing 497, and test containing 840,
with all validation/tests passing. Preserve event IDs and source text for later
comparison between annotators. Report progress periodically. Finish with the
actual model identity, prompt hash, output paths, row counts, validation results
and unresolved review cases. Do not modify taxonomy, manifests, library code, or
scripts. Do not compare labels against the first annotator during this run.
