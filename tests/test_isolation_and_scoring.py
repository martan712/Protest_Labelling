from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from protest_classifier.evaluation.metrics import score

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "src" / "protest_classifier" / "cli"


class ScoringTests(unittest.TestCase):
    def test_strict_and_independent_alternative_scores(self) -> None:
        frame = pd.DataFrame([
            {"event_id_cnty": "1", "notes": "n1", "primary_label": "farmers", "alternative_labels": ""},
            {"event_id_cnty": "2", "notes": "n2", "primary_label": "culture", "alternative_labels": "education"},
        ])
        result = score(frame, ["farmers", "education"])
        self.assertEqual(result["strict_accuracy"], 0.5)
        self.assertEqual(result["accepted_accuracy"], 1.0)


class IsolationTests(unittest.TestCase):
    def test_training_path_does_not_reference_test_gold(self) -> None:
        for name in ("train_classifier.py", "predict_events.py"):
            source = (CLI / name).read_text().lower()
            self.assertNotIn("data/annotations/test_locked.csv", source)
            self.assertNotIn("load_locked_test", source)

    def test_evaluation_is_the_test_gold_entry_point(self) -> None:
        source = (CLI / "evaluate_classifier.py").read_text()
        self.assertIn("load_locked_test", source)


if __name__ == "__main__":
    unittest.main()
