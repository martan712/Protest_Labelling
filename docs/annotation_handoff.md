# Annotation handoff — DeepSeek V4.1-Flash

This handoff covers exactly **one** annotator model: **DeepSeek-V4.1-Flash**,
API model string `deepseek-flash`. Do not use any other model for this run. A
second, independent annotation set is produced by a different model under
`docs/annotation_handoff_second_model.md`; that is not your job and not your
directory.

## Paste this task into the annotation agent

Read `docs/prompts/annotation_prompt.md` completely and follow its ordered procedure
for every event. It is the instruction prompt, not optional reference material.
Read this handoff completely before taking action. Use taxonomy `2026-09-21-v3`.
Do not begin training. Do not change the taxonomy to accommodate difficult rows.

## Model — fixed, not a choice

- Model: DeepSeek-V4.1-Flash
- API model string: `deepseek-flash`
- Legacy aliases `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` are
  retired names that route to this same model. Call `deepseek-flash` explicitly
  so the recorded identity is unambiguous.

Verify the provider's actual model identity at launch and record what the API
reports; aliases and routing can change. If `deepseek-flash` is unavailable,
stop and report it — do not substitute another model, and do not run this
handoff with the V4-Pro model used by the second-model handoff.

## Output directory — exclusive to this run

Your output root is exclusively:

`data/annotation_runs/deepseek-v4-1-flash/RUN_ID/`

Replace `RUN_ID` with a unique run identifier (suggested form
`run-YYYY-MM-DD-flash-NN`). All chunks, assembled CSVs, pilot files, progress
records, review notes and the prompt snapshot for this run live under that root.

Do not write to `data/annotations/`, to the frozen manifests, to
`data/annotation_runs/deepseek-v4-pro/` (the second model's root), or to any
other annotation run. `data/annotation_runs/run-2026-09-21-v3-flash/` is a
superseded earlier run with known write-provenance contamination: do not read
it, resume it, or write into it. Do not silently publish this run over
`data/annotations/`.

If any process other than your own subagents writes into your output root, stop,
record it, and report it. Exclusive ownership of this directory is a hard
requirement — the previous run failed on exactly this point.

## Provenance to record

Set `annotator` on every row to `deepseek-v4.1-flash-RUN_ID`, consistently, with
`RUN_ID` substituted. Save in the run directory: the exact instruction prompt,
its SHA-256 hash, the taxonomy version, the model identifier as reported by the
provider, generation settings and progress counts.

At run start the frozen prompt is `docs/prompts/annotation_prompt.md`, taxonomy
`2026-09-21-v3`, SHA-256
`19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250`. If the
prompt's hash differs, report the mismatch before annotating: the second model
must run the byte-identical prompt or the two sets cannot be compared.

## Pilot, then bounded batches

First annotate a 100-row training pilot, including both a reproducible random
sample and examples of ambiguous boundaries. Check schema validity and
consistency of repeated subjects. Flag unresolved interpretation issues; do not
invent rules. Continue with the frozen prompt if the rules resolve them. Process
the remaining rows in bounded batches of 25–50 events. Validate each batch
immediately and retry only missing/invalid rows. Do not mix records across
splits. Review every other prediction against the two reason definitions and the
permitted group fallback. Record review counts and any unresolved cases
separately from the annotation CSV. Reuse validated pilot annotations when
completing the training set.

For each model request, send the unchanged `docs/prompts/annotation_prompt.md` as the
system instruction and a user message containing the run identifier and only
`event_id_cnty` and `notes` for that batch. JSON input is suitable; output is CSV
under the prompt's exact schema. Do not send old gold labels, classifier errors,
or predictions. Do not use keyword rules to generate labels.

## Subagent workflow: read one batch, write it, then continue

Use annotation subagents for independent batches. Every subagent uses
`deepseek-flash` — the same single model as the coordinator — and must read
`docs/prompts/annotation_prompt.md` completely before annotating, with the same
frozen prompt, taxonomy version, and run identifier.

Assign exclusive batch files to each subagent; never let two agents write the
same file. Each subagent must follow this sequence, including during the pilot:

1. Read exactly one assigned batch of at most 50 event notes.
2. Annotate every event in that batch according to the prompt.
3. Write that completed batch to its assigned file immediately.
4. Validate the saved CSV: exact expected IDs, no duplicates or missing rows,
   valid labels/reasons, and nonempty evidence and annotator. Repair that batch
   if validation fails and report completion only after validation succeeds.
5. Only then read the next assigned batch.

Do not preload multiple batches, hold completed batches in memory for a later
combined write, or read the whole dataset into a subagent's context. Parallelism
is across subagents; the read-annotate-write-validate cycle is sequential within
each subagent. Reuse the read-only prompt, not earlier event batches, as context.
Return file paths and counts to the coordinator rather than all annotation text.

The coordinator tracks batch ownership and validated completion, retries only
incomplete or failed batches, and assembles outputs after all batches are saved.
On resume, reuse only completed, validated chunks from this same run, with the
same model and prompt, and verify them before skipping. Never recreate queues
over completed chunk files. Do not overwrite validated batches merely to restart
a worker.

## Throughput

Start with small batches and short evidence, using non-thinking mode if supported
and a fixed low temperature if supported. Measure pilot throughput and semantic
quality before increasing concurrency. Keep the static prompt identical to benefit
from provider prompt caching where available. Do not promise an ETA from model
marketing. Escalate ambiguous rows for review rather than repeatedly regenerating
the entire dataset. Low temperature does not guarantee determinism.

## Files and assembly

Annotate every event in the frozen manifests using the exact contract in
`docs/prompts/annotation_prompt.md`.

Required sets:

- `data/manifests/train_6000.csv`: 6,000 training events
- `data/manifests/dev.csv`: 497 development events
- `data/manifests/test_locked.csv`: 840 locked-test events

Use only each row's `notes`. Do not inspect or reuse anything under `data/archive/`,
old labels, predictions, keywords, model outputs, country, or date. Fill these
fields for every row: `primary_label`, `alternative_labels`, `other_reason`,
`evidence`, and `annotator`. Use exact label strings. Supply at most one genuine
alternative; leave it empty otherwise. `other_reason` is required only for
`other`.

Create and complete files chunk by chunk. Replace `RUN_ID` in every command. Run
queue creation only for a fresh run, before annotations have been written; do not
execute it again when resuming.

```bash
uv run scripts/create_annotation_chunks.py --manifest data/manifests/train_6000.csv --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/train_chunks --chunk-size 50
uv run scripts/create_annotation_chunks.py --manifest data/manifests/dev.csv --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/dev_chunks --chunk-size 50
uv run scripts/create_annotation_chunks.py --manifest data/manifests/test_locked.csv --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/test_chunks --chunk-size 50
```

Do not train a model or inspect predictions. After all chunks are complete, run:

```bash
uv run scripts/prepare_training_releases.py --chunks data/annotation_runs/deepseek-v4-1-flash/RUN_ID/train_chunks --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/assembled
uv run scripts/assemble_annotations.py --manifest data/manifests/dev.csv --chunks data/annotation_runs/deepseek-v4-1-flash/RUN_ID/dev_chunks --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/assembled/dev.csv
uv run scripts/assemble_annotations.py --manifest data/manifests/test_locked.csv --chunks data/annotation_runs/deepseek-v4-1-flash/RUN_ID/test_chunks --output data/annotation_runs/deepseek-v4-1-flash/RUN_ID/assembled/test_locked.csv
.venv/bin/python -m unittest discover -v
```

Do not treat a zero exit status as completion. `prepare_training_releases.py`
skips any release whose labels are incomplete, prints `not yet complete: [...]`,
and still exits 0, and blank-label rows are silently dropped when chunks are
read. Before reporting completion, assert the counts explicitly:

```bash
.venv/bin/python - <<'CHECK'
import pandas as pd, sys
root = "data/annotation_runs/deepseek-v4-1-flash/RUN_ID"
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

Completion means the assembled training releases contain exactly 800, 1,500,
3,000, and 6,000 rows; development contains 497; locked test contains 840; and
all validation/tests pass. Preserve event IDs and source text for later
comparison against the second model's set. Finish with the actual model identity,
prompt hash, output paths, row counts, validation results and unresolved review
cases. Report only progress, validation failures, and the final row counts. Do
not modify taxonomy, manifests, library code, or scripts. Do not compare labels
against any other annotator during this run.
