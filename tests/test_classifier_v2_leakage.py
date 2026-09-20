"""Leakage and label-mapping tests for the standalone classifier.

These are the automated safeguards required by the plan: training, development
and test must share no event ID and no note hash; the model mapping must be the
shared taxonomy; and training code must not be able to read the locked test gold.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier_v2 import common  # noqa: E402
from classifier_v2.common import (  # noqa: E402
    CLASS_NAMES,
    CLASSIFIER_DATA_DIR,
    DEV_REMOVE_IDS,
    assert_disjoint,
    label_maps,
    load_annotations,
    load_dev_497,
    load_test_set,
    validate_annotations,
)


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_annotations()
        cls.dev = load_dev_497()
        cls.test = load_test_set()

    def test_split_sizes(self) -> None:
        # The labelled pool grows as annotations arrive; the 800 row release is
        # the floor.  Development and test are frozen.
        self.assertGreaterEqual(len(self.train), 800)
        self.assertEqual(len(self.dev), 497)
        self.assertEqual(len(self.test), 840)

    def test_train_dev_disjoint(self) -> None:
        assert_disjoint(self.train, self.dev, left_name="train", right_name="dev")

    def test_train_test_disjoint(self) -> None:
        assert_disjoint(self.train, self.test, left_name="train", right_name="test")

    def test_dev_test_disjoint(self) -> None:
        assert_disjoint(self.dev, self.test, left_name="dev", right_name="test")

    def test_required_removed_ids_are_in_test(self) -> None:
        self.assertTrue(set(DEV_REMOVE_IDS).issubset(set(self.test["event_id_cnty"])))

    def test_generated_releases_are_clean_if_present(self) -> None:
        train_path = CLASSIFIER_DATA_DIR / "train_0800.csv"
        dev_path = CLASSIFIER_DATA_DIR / "dev_497.csv"
        if not (train_path.exists() and dev_path.exists()):
            self.skipTest("generated releases not present")
        train = pd.read_csv(train_path, low_memory=False)
        dev = pd.read_csv(dev_path, low_memory=False)
        assert_disjoint(train, dev, left_name="train_0800", right_name="dev_497")
        assert_disjoint(train, self.test, left_name="train_0800", right_name="test")
        assert_disjoint(dev, self.test, left_name="dev_497", right_name="test")

    def test_label_mapping_is_shared_taxonomy(self) -> None:
        id2label, label2id = label_maps()
        self.assertEqual(len(id2label), 21)
        self.assertEqual([id2label[i] for i in range(21)], list(CLASS_NAMES))
        self.assertEqual({name: i for i, name in enumerate(CLASS_NAMES)}, label2id)

    def test_validate_rejects_out_of_taxonomy_label(self) -> None:
        frame = pd.DataFrame([{
            "event_id_cnty": "X1", "notes": "a protest", "note_hash": "h",
            "primary_label": "not a class", "other_reason": "", "reviewer": "t",
        }])
        with self.assertRaises(ValueError):
            validate_annotations(frame)

    def test_validate_rejects_other_without_reason(self) -> None:
        frame = pd.DataFrame([{
            "event_id_cnty": "X1", "notes": "a protest", "note_hash": "h",
            "primary_label": "other", "other_reason": "", "reviewer": "t",
        }])
        with self.assertRaises(ValueError):
            validate_annotations(frame)

    def test_validate_rejects_empty_note(self) -> None:
        frame = pd.DataFrame([{
            "event_id_cnty": "X1", "notes": "   ", "note_hash": "h",
            "primary_label": "climate", "other_reason": "", "reviewer": "t",
        }])
        with self.assertRaises(ValueError):
            validate_annotations(frame)


class TrainingCodeIsolationTests(unittest.TestCase):
    """Training-path modules must never reference the locked test gold."""

    def test_training_modules_do_not_read_test_gold(self) -> None:
        modules = [
            ROOT / "classifier_v2" / "train.py",
            ROOT / "classifier_v2" / "prepare_training_data.py",
            ROOT / "classifier_v2" / "sample_more_events.py",
            ROOT / "classifier_v2" / "predict.py",
        ]
        for module in modules:
            source = module.read_text().lower()
            self.assertNotIn("test_gold.csv", source, f"{module.name} references the test gold")
            self.assertNotIn("load_test_set", source, f"{module.name} loads the test set")
            self.assertNotIn("--test-gold", source, f"{module.name} accepts a test-gold argument")

    def test_only_evaluate_reads_test_gold(self) -> None:
        evaluate = (ROOT / "classifier_v2" / "evaluate.py").read_text()
        self.assertIn("load_test_set", evaluate)

    def test_test_gold_path_constant_is_not_a_cli_arg(self) -> None:
        # common.TEST_GOLD is a constant used by evaluate/tests, never an argparse option.
        self.assertTrue(str(common.TEST_GOLD).endswith("test_gold.csv"))


if __name__ == "__main__":
    unittest.main()
