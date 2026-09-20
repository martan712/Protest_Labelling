"""Data-contract tests for the classifier_v2 training releases."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier_v2.common import (  # noqa: E402
    CLASS_NAMES,
    CLASSIFIER_DATA_DIR,
    DEV_REMOVE_IDS,
    OTHER_REASONS,
    RELEASE_COLUMNS,
    load_annotations,
    load_dev_497,
)
from classifier_v2.prepare_training_data import prepare  # noqa: E402


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.annotations = load_annotations()
        cls.dev = load_dev_497()

    def test_base_release_has_800_rows(self) -> None:
        # The labelled pool grows over time; the frozen floor release is 800.
        self.assertGreaterEqual(len(self.annotations), 800)
        self.assertEqual(self.annotations["event_id_cnty"].nunique(), len(self.annotations))
        base_path = CLASSIFIER_DATA_DIR / "train_0800.csv"
        if not base_path.exists():
            self.skipTest("generated train_0800.csv not present")
        base = pd.read_csv(base_path, low_memory=False)
        self.assertEqual(len(base), 800)
        self.assertTrue(set(base["event_id_cnty"]).issubset(set(self.annotations["event_id_cnty"])))

    def test_release_columns(self) -> None:
        self.assertEqual(list(self.annotations.columns), list(RELEASE_COLUMNS))

    def test_labels_within_taxonomy(self) -> None:
        self.assertTrue(set(self.annotations["primary_label"]).issubset(set(CLASS_NAMES)))

    def test_other_reason_contract(self) -> None:
        other = self.annotations[self.annotations["primary_label"] == "other"]
        self.assertTrue(other["other_reason"].isin(OTHER_REASONS).all())
        non_other = self.annotations[self.annotations["primary_label"] != "other"]
        self.assertTrue(non_other["other_reason"].isin(["", "nan"]).all())

    def test_notes_are_non_empty(self) -> None:
        self.assertTrue((self.annotations["notes"].str.strip() != "").all())

    def test_development_has_497_rows_with_three_ids_removed(self) -> None:
        self.assertEqual(len(self.dev), 497)
        self.assertFalse(set(DEV_REMOVE_IDS) & set(self.dev["event_id_cnty"]))

    def test_development_labels_within_taxonomy(self) -> None:
        self.assertTrue(set(self.dev["primary_label"]).issubset(set(CLASS_NAMES)))

    def test_nested_releases_are_strict_supersets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            releases = prepare(sizes=(800, 1500, 3000, 6000), out_dir=Path(directory))
        sizes = sorted(releases)
        self.assertIn(800, sizes)
        for smaller, larger in zip(sizes, sizes[1:]):
            self.assertTrue(
                set(releases[smaller]["event_id_cnty"]).issubset(set(releases[larger]["event_id_cnty"]))
            )
            self.assertLess(len(releases[smaller]), len(releases[larger]))

    def test_written_release_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepare(sizes=(800,), out_dir=Path(directory))
            written = pd.read_csv(Path(directory) / "train_0800.csv")
            dev = pd.read_csv(Path(directory) / "dev_497.csv")
        self.assertEqual(len(written), 800)
        self.assertEqual(len(dev), 497)
        self.assertEqual(list(written.columns), list(RELEASE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
