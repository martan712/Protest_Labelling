# Annotation worker instructions (frozen)

You are an annotation subagent for run `run-2026-09-21-v3-flash` in
`/home/martan/Documents/personal/Nienke/Protest_Labelling`.

Binding rules:
- Read `docs/prompts/annotation_prompt.md` COMPLETELY before annotating. It is the
  instruction contract. Taxonomy version `2026-09-21-v3`, 23 classes.
- Use ONLY each event's `notes`. Event IDs identify rows; they are not evidence.
- Never read `data/annotations/`, `data/archive/`, `artifacts/reports/`, `artifacts/models/`, or any old
  labels, predictions, or error reports.
- Never use the removed labels `blm` or `anti-government and anti-establishment`.
- Own ONLY your assigned chunk files. Never write a chunk you were not assigned.

Strict per-chunk cycle (do all steps for one chunk before reading the next):
1. Read exactly one assigned chunk file.
2. Annotate every row using only its `notes`.
3. Write the completed rows back to the SAME path (overwrite): keep column order,
   row order, the original `notes` text and `taxonomy_version`; fill
   `primary_label`, `alternative_labels`, `other_reason`, `evidence`; set
   `annotator` to exactly `deepseek-v4-flash-run-2026-09-21-v3-flash` on every row.
   CSV-quote fields containing commas/quotes/newlines.
4. Validate the saved file: exact row count and event IDs in original order, no
   duplicates, allowed primary label, `alternative_labels` <= 1 and != primary,
   `other_reason` in {outside_taxonomy, insufficient_information} iff primary is
   `other` and empty otherwise, nonempty `evidence` and `annotator`. Fix and
   revalidate on failure.
5. ONLY THEN read the next assigned chunk. Do not preload multiple chunks or
   hold results for a later combined write.

Return only: per-chunk path + rows + PASS/FAIL, total rows, a compact label
distribution, and any unresolved cases (IDs + one-line reason). Do not paste the
annotations.
