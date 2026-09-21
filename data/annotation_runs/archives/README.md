# Annotation run archives

Zipped snapshots of the two independent annotation runs over the frozen
manifests. Both used the byte-identical prompt `configs/annotation_prompt.md`
(taxonomy `2026-09-21-v3`, SHA-256 `19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250`)
and differ only in the annotating model.

| Archive | Model | Annotator string |
| --- | --- | --- |
| `run-2026-09-21-flash-01.zip` | DeepSeek-V4.1-Flash (`deepseek-flash`) | `deepseek-v4.1-flash-run-2026-09-21-flash-01` |
| `run-2026-09-21-pro-01.zip` | DeepSeek-V4-Pro-0813 (`deepseek-v4-pro`) | `deepseek-v4-pro-run-2026-09-21-pro-01` |

Each archive holds `assembled/` (train_0800, train_1500, train_3000, train_6000,
dev, test_locked), the prompt snapshot and its hash, and the run's provenance
and review notes. Intermediate `*_chunks/` files are excluded: the assembled
CSVs are those same rows joined to the manifests.

Agreement between the two, from `scripts/compare_annotations.py`:

| Split | Strict | Lenient | Cohen's kappa |
| --- | --- | --- | --- |
| dev | 90.74% | 94.37% | 0.8968 |
| test_locked | 91.31% | 93.21% | 0.9052 |
| train_6000 | 93.17% | 95.25% | 0.9256 |

Flash is the working set for now. The two disagree mainly at the `other`
boundary, in opposite directions: flash under-uses the group/institution
fallback, pro over-uses it. The ~545 strict disagreements on train_6000 are not
adjudicated. See `notebooks/compare_annotators.ipynb`.

Unpack with `unzip <archive> -d <destination>`.
