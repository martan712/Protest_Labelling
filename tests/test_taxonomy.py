from __future__ import annotations

import unittest
from pathlib import Path

from protest_classifier.taxonomy import BOUNDARY_RULES, CLASS_NAMES, DESCRIPTIONS, validate_label


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_has_23_unique_defined_labels(self) -> None:
        self.assertEqual(len(CLASS_NAMES), 23)
        self.assertEqual(len(set(CLASS_NAMES)), 23)
        self.assertEqual(set(CLASS_NAMES), set(DESCRIPTIONS))

    def test_other_reason_contract(self) -> None:
        validate_label("other", "outside_taxonomy")
        validate_label("other", "insufficient_information")
        with self.assertRaises(ValueError):
            validate_label("other")
        with self.assertRaises(ValueError):
            validate_label("farmers", "outside_taxonomy")

    def test_annotator_prompt_contains_canonical_wording(self) -> None:
        prompt = (Path(__file__).resolve().parent.parent / "configs" / "annotation_prompt.md").read_text()
        normalized = " ".join(prompt.replace("`", "").replace("→", "->").split())
        for label, description in DESCRIPTIONS.items():
            self.assertIn(f"{label}: {' '.join(description.split())}", normalized)
        for rule in BOUNDARY_RULES:
            self.assertIn(" ".join(rule.split()), normalized)


if __name__ == "__main__":
    unittest.main()
