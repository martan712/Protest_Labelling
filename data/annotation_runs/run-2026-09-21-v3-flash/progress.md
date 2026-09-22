# Run progress — run-2026-09-21-v3-flash

- Taxonomy version: `2026-09-21-v3` (23 classes)
- Model / annotator identifier: `deepseek-v4-flash-run-2026-09-21-v3-flash`
- Frozen prompt: `annotation_prompt.md` (copy in this dir),
  SHA-256 `19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d92756167418af18ea8250`
  (verified byte-identical to `docs/prompts/annotation_prompt.md` at run start).
- Output location: this run directory only. `data/annotations/` was left
  untouched (it holds superseded `2026-09-20-v1` labels).

## Completion state

| Split | Files | Rows | Filled | Validated |
| --- | --- | --- | --- | --- |
| train_chunks | 120 | 6,000 | 6,000 | schema PASS |
| dev_chunks | 10 | 497 | 497 | schema PASS |
| test_chunks | 17 | 840 | 840 | schema PASS |

Assembled outputs (`assembled/`):

| File | Rows |
| --- | --- |
| train_0800.csv | 800 |
| train_1500.csv | 1,500 |
| train_3000.csv | 3,000 |
| train_6000.csv | 6,000 |
| dev.csv | 497 |
| test_locked.csv | 840 |

`python -m unittest discover -v` → 11 tests, all OK.

## Pilot

100-row training pilot (`pilot/`), randomly sampled with boundary cases.
Both halves validated; see `pilot/pilot_review.md`.

## Batch workflow

Batches were bounded at 50 events (chunk size 50). Subagents were assigned
exclusive chunk ranges and instructed to follow the read -> annotate -> write ->
validate -> next cycle strictly. Chunks were written in place and validated
immediately; assembly reads only rows with a nonempty primary label.

## Review counts (assembled)

- `other` rows (primary): train_6000 = 519 (499 outside_taxonomy,
  20 insufficient_information); dev = 79 (75/4); test_locked = 69 (61/8).
  Every one carries a valid `other_reason` and no non-`other` row carries one.
- Secondary label present: train_6000 = 747, dev = 38, test_locked = 72;
  never more than one and never equal to the primary.
- Evidence and annotator nonempty on every row; no duplicate IDs.

## Provenance limitation (UNRESOLVED — review)

A separate, uncoordinated process wrote annotations into the SAME run
directory while this session was running:

- A distinct `claude` OS process was active on this repository, and chunk files
  outside the ranges assigned by this coordinator (e.g. `chunk_031..060`,
  `chunk_071`) became populated.
- Two annotation subagents reported their assigned files being rewritten
  mid-run by an external process (`chunk_013..018`, and `ITA20467`/`ITA22206`
  in `chunk_093`), and re-applied their own labels.

Consequence: the final files are schema-valid and internally consistent, and
every row carries the same annotator string, but per-row write provenance inside
this run cannot be attributed to a single exclusive writer. This violates the
handoff's exclusive-batch-ownership requirement and is recorded here as an
unresolved integrity/review case. The frozen prompt, taxonomy and manifests were
not modified.
