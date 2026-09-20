"""Build the nested ModernBERT training releases and the development set.

The classifier is trained on the *semantic* annotations only.  This script:

1. loads every labelled annotation chunk (``unknown_starter_v1`` + ``expanded_v1``
   + any later ``sampled_v1`` chunks) in source order;
2. validates the taxonomy contract (labels, ``other_reason``, non-empty notes);
3. deduplicates by ``event_id_cnty``;
4. writes nested releases ``train_0800.csv`` → ``train_1500.csv`` → … where each
   release strictly contains the smaller one;
5. writes ``dev_497.csv`` (``dev_gold.csv`` minus the three test-overlapping IDs).

Training data is checked against development and the locked test *manifest*
(IDs and hashes only — never the test labels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier_v2.common import (
    ANNOTATION_SOURCES,
    CLASSIFIER_DATA_DIR,
    DEV_GOLD,
    DEV_REMOVE_IDS,
    RELEASE_COLUMNS,
    TEST_MANIFEST_840,
    TRAINING_RELEASE_SIZES,
    assert_disjoint,
    assert_nested,
    load_annotations,
    load_dev_497,
)


def prepare(
    sizes: tuple[int, ...] = TRAINING_RELEASE_SIZES,
    out_dir: Path = CLASSIFIER_DATA_DIR,
    sources: tuple[tuple[str, str], ...] = ANNOTATION_SOURCES,
    test_manifest: Path = TEST_MANIFEST_840,
    dev_gold: Path = DEV_GOLD,
) -> dict[int, pd.DataFrame]:
    """Return (and, when ``out_dir`` is given, write) the nested releases."""
    annotations = load_annotations(sources)
    available = len(annotations)
    print(f"Labelled annotations available: {available}")

    dev = load_dev_497(dev_gold)
    if len(dev) != 497:
        raise ValueError(f"Expected 497 development rows, got {len(dev)}")

    manifest = pd.read_csv(test_manifest, low_memory=False)
    test_ids = manifest[["event_id_cnty", "note_hash"]].copy()

    # Development must be clean against training and test.
    assert_disjoint(annotations, dev, left_name="training", right_name="development")
    assert_disjoint(dev, test_ids, left_name="development", right_name="test")
    assert_disjoint(annotations, test_ids, left_name="training", right_name="test")

    # The three removed IDs really are in the test manifest.
    present = set(DEV_REMOVE_IDS) & set(test_ids["event_id_cnty"])
    if present != set(DEV_REMOVE_IDS):
        raise ValueError(f"Expected all of {DEV_REMOVE_IDS} in the test manifest, found {sorted(present)}")

    requested = sorted(sizes)
    releases: dict[int, pd.DataFrame] = {}
    for size in requested:
        if size > available:
            print(f"  skip train_{size:04d}.csv: only {available} labelled rows available")
            continue
        releases[size] = annotations.iloc[:size][list(RELEASE_COLUMNS)].copy()

    if not releases:
        raise ValueError(
            f"Not enough labelled annotations for the smallest release "
            f"({requested[0]}); have {available}."
        )

    ordered_sizes = sorted(releases)
    for smaller, larger in zip(ordered_sizes, ordered_sizes[1:]):
        assert_nested(
            releases[smaller], releases[larger],
            smaller_name=f"train_{smaller:04d}", larger_name=f"train_{larger:04d}",
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for size, frame in releases.items():
            path = out_dir / f"train_{size:04d}.csv"
            frame.to_csv(path, index=False)
            print(f"  wrote {path} ({len(frame)} rows)")
        dev_path = out_dir / "dev_497.csv"
        dev.to_csv(dev_path, index=False)
        print(f"  wrote {dev_path} ({len(dev)} rows)")

    return releases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=CLASSIFIER_DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(TRAINING_RELEASE_SIZES))
    parser.add_argument(
        "--dev-gold", type=Path, default=DEV_GOLD,
        help="Path to data/run/dev_gold.csv (default). Three IDs are always removed.",
    )
    parser.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST_840,
                        help="Frozen test manifest (IDs/hashes only).")
    args = parser.parse_args()

    releases = prepare(
        sizes=tuple(args.sizes), out_dir=args.out_dir,
        test_manifest=args.test_manifest, dev_gold=args.dev_gold,
    )
    for size, frame in sorted(releases.items()):
        counts = frame["primary_label"].value_counts().sort_index()
        print(f"\ntrain_{size:04d} class distribution (top 5):")
        print(counts.head().to_string())


if __name__ == "__main__":
    main()
