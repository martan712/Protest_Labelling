# Active annotation files

These files must be created from the manifests using the active taxonomy. Do
not copy labels from `archive/legacy_2026-09-20/data`; they use an incompatible
taxonomy. Each annotation file should contain `event_id_cnty`, `notes`,
`primary_label`, `alternative_labels`, `other_reason`, `evidence`, and
`annotator`.

Required assembled splits are `dev.csv`, `test_locked.csv`, and nested training releases
`train_0800.csv`, `train_1500.csv`, `train_3000.csv`, and `train_6000.csv`.
Training labels may first be entered in `train_chunks/chunk_*.csv`; the assembly
command joins them to the frozen manifests and performs leakage/schema checks.
