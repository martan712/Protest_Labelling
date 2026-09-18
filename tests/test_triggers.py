"""Regression tests for known weak-label trigger failures."""

from __future__ import annotations

import unittest

from pipeline.finetuner.triggers import match_rules, rule_label


class TriggerRegressionTests(unittest.TestCase):
    def assert_only(self, expected: str, note: str) -> None:
        self.assertEqual(rule_label(match_rules(note)), expected)

    def test_pension_does_not_match_suspension(self) -> None:
        self.assertEqual(match_rules("Residents opposed suspension of the bridge works."), {})
        self.assert_only("labor rights", "Retired workers demanded restoration of their pensions.")

    def test_labor_conditions_and_workplace_cases(self) -> None:
        for note in [
            "Staff protested poor labour conditions.",
            "Employees opposed the factory closure.",
            "Nurses protested excessive workload.",
            "Police officers demanded a pay rise.",
            "The union opposed suspension of workers.",
        ]:
            with self.subTest(note=note):
                self.assert_only("labor rights", note)

    def test_hunger_strike_is_not_labor_by_participant_action(self) -> None:
        self.assertEqual(match_rules("Prisoners began a hunger strike over isolation."), {})

    def test_environment_and_nursery_regressions(self) -> None:
        self.assert_only("environment", "Residents protested deforestation and biodiversity loss.")
        self.assert_only("education", "Parents opposed the nursery closure.")

    def test_morphological_variants(self) -> None:
        cases = {
            "Russian troops were the subject of an anti-war rally.": "ukraine-russia war",
            "Solidarity with Ukrainian civilians motivated the march.": "ukraine-russia war",
            "Campaigners defended migrant rights.": "immigration",
            "The rally supported asylum seekers.": "immigration",
            "Activists marched for LGBT rights.": "lgbtq",
        }
        for note, expected in cases.items():
            with self.subTest(note=note):
                self.assert_only(expected, note)


if __name__ == "__main__":
    unittest.main()
