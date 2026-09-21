# Pilot review — run-2026-09-21-v3-flash

Pilot size: 100 training events (reproducible random sample + boundary cases),
split into `pilot_batch_001.csv` / `pilot_batch_002.csv`; completed as
`pilot_001_done.csv` / `pilot_002_done.csv`.

## Validation counts

- Rows: 100 (50 + 50), unique event IDs, no duplicates.
- Input order and note text preserved in both halves.
- All primary labels are among the 23 taxonomy v3 classes. No removed labels
  (`blm`, `anti-government and anti-establishment`) used.
- `alternative_labels`: populated on 20 rows; never equal to the primary;
  never more than one label.
- `other`: 13 rows, each with an `other_reason` in
  {`outside_taxonomy`, `insufficient_information`}; `other_reason` empty on all
  non-`other` rows.
- `evidence` and `annotator` nonempty on every row.
- `annotator` consistent: `deepseek-v4-flash-run-2026-09-21-v3-flash`.
- Library validator `validate_labels` passes for each half and for the combined
  pilot.

## Repeated-subject / consistency spot check

Repeated subjects (e.g. national "1 of 5 Million" anti-government mobilizations,
pandemic-vs-labour work disputes) were labelled consistently with the ordered
decision procedure; the `other` fallback was used where the note supplied no
covered domain. No rule changes were required.

## Unresolved interpretation issues (forwarded, not resolved by inventing rules)

1. Public-employment recruitment-exam cancellations (e.g. `ESP8238`):
   labour vs education vs `other`. Recorded `other` (outside_taxonomy); flag for
   review.
2. Local administrative/permit-rule protests with no covered domain
   (e.g. `NLD3884`): recorded `other` (outside_taxonomy).
3. Anti-government mobilization motivated by an attack on an opposition leader
   (`SRB789`): recorded `crime, violence and victim justice`; democratic
   institutions is a defensible alternative.
4. Same movement with no issue stated (`SRB1149`): recorded
   `other`/`outside_taxonomy`.
5. Compulsory livestock culling (`FRA43792`): `animal welfare` primary with
   `farmers` secondary, per the "owners opposing compulsory slaughter" boundary
   rule; split remains judgement-heavy.
6. Pandemic measures vs work consequences (`NLD868`, `FRA2826`): primary chosen
   as pandemic vs labour respectively, each with the other as secondary; could be
   swapped under emphasis.
7. Local-administration inaction on crime (`ITA2800`): recorded
   `crime, violence and victim justice`; `other` is defensible.

These are recorded as review items only. The frozen prompt was not modified.
