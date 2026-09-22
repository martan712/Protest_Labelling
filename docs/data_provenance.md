# Data provenance

## Active event table

`data/raw/events.csv` is the unchanged file formerly named
`data/filtered_events.csv`. It was moved on 20 September 2026; it was not
rewritten during the overhaul.

Observed properties:

- 218,108 unique events dated 2 January 2018 through 1 June 2026;
- 43 selected European countries;
- 213,199 protests and 4,909 riots;
- original ACLED fields plus the previously derived `clean_notes` and
  `country_code` columns;
- SHA-256: `49dfb282bbfd7ef37700296fd6412ad4e4ee74386e272249fc9838dc1decbec0`.

This is the source used by classifier-v2 and remains the active classification
population. The model reads only `notes`; `clean_notes` is retained solely
because it was already present in the source file.

## Larger ACLED export

`data/raw/acled_2026-06-10.csv` is a larger European ACLED export whose filename
indicates 10 June 2026. No download receipt or API query was retained, so its
origin must not be stated more precisely than that.

Observed properties:

- 616,973 unique events dated 1 January 2018 through 1 June 2026;
- 53 European countries and all six event types;
- every event ID in `events.csv` is present in this export;
- SHA-256: `2e69e4e6fafeb1b79eda2eb19d093524a8eddd46e8ab2ac03dae4576558185e7`.

The two files are not related by a presently reproducible single filter: the
larger export contains 70 additional events whose current disorder type includes
`Demonstrations`. This likely reflects filtering or source revisions, but the
repository does not contain enough provenance to distinguish those explanations.
For reproducibility, use `events.csv` directly rather than regenerating it.

## Split selections

The active manifests retain classifier-v2's exact event selections while
discarding its old labels. Consequently, new-taxonomy results remain comparable
across the same 800/1,500/3,000/6,000 training sizes, 497-row development set,
and 840-row locked test set.
