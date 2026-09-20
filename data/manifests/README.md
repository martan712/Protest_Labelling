# Selection manifests

Manifests contain event identity, notes, and normalized-note hashes—never
labels. Generate them with
`.venv/bin/python scripts/build_manifests.py`.

The committed files preserve the exact classifier-v2 event selections after
stripping their labels. By default the script refreshes their source fields from
`data/raw/events.csv` while retaining those IDs. Use `--fresh-random` only if
comparability with prior learning curves is intentionally being abandoned.

The development and locked-test manifests are disjoint from every training
release. Training manifests are nested: 800 is contained in 1500, which is
contained in 3000, which is contained in 6000.
