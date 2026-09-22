# Review log — run-2026-09-21-v3-flash

## What was reviewed
- Schema/integrity of every chunk (147/147) and every assembled output: exact
  expected IDs once each, input order, `notes` byte-identical to the manifest,
  labels from taxonomy `2026-09-21-v3`, `alternative_labels` empty or one valid
  different label, `other_reason` only for `other`, nonempty evidence/annotator.
- Re-annotation consistency: the 100-row pilot was drawn from the same corpus and
  re-annotated inside its chunks; 54 rows overlapped (pilot ∩ first 3,000 train
  chunks) with **94.4%** primary-label agreement. The 3 disagreements were the
  flagged boundary cases (AUT463, BEL2706, ESP8238).
- The `other` decision was reviewed against the two reason definitions
  (`outside_taxonomy`, `insufficient_information`) and the permitted
  group/institution fallback: a **systematic 50% sample** (every other row) of the
  `other` class — **334 of 667** rows — across train/dev/test.

## Review counts
| quantity | value |
|---|---|
| `other` rows (train/dev/test) | 519 / 79 / 69 = 667 |
| `outside_taxonomy` / `insufficient_information` | 635 / 32 |
| `other` rows in the review sample | 334 (50.1%) |
| invalid `other_reason` found | 0 |
| chunks with integrity defects | 0 after repair |
| repeated-subject label conflicts (exact duplicate notes) | 0 |

Notes repaired during QA (notes had been altered by a worker; labels kept):
`train_chunks/chunk_098.csv` (ITA3958) and `dev_chunks/chunk_002.csv` (ESP6459).

## Unresolved / judgment-call cases
Recorded separately in `review/unresolved_cases.csv`. These are defensible
`other` calls where an alternative covered label is also arguable; they are left
as annotated rather than changed ad hoc. Main clusters:
- Group-fallback tension: BEL4981 (social workers vs welfare-sector cuts),
  GBR1161 (residents vs proposed housing development).
- International-scope tension: ITA28447 (children's rights + peace for
  Ukraine/Palestine), ESP15259 (opposition to an amnesty pact).
- Far-right rally/act vs `racism`/`anti-fascism and extremism`/`crime, violence
  and victim justice`: FRA17999, ITA13367, SWE2610.
- Energy/infrastructure siting opposed with no environmental reason stated
  (`other` vs `climate and environment`): DEU3205, NOR1936, DNK3065, BEL627,
  FRA8398 (5G), FRA3644/BEL3563 (local plants).
- Other: DEU2651 (slaughterhouse preservation), MNE83 (hunters vs conditions),
  SRB1131 (veterans' social security / health insurance).

No taxonomy rule or keyword rule was invented to resolve these.
