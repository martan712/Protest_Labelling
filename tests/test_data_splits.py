"""Tests that duplicate provenance groups cannot cross model splits."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.finetuner.data import load_data


class GroupSplitTests(unittest.TestCase):
    def test_duplicate_group_is_atomic(self) -> None:
        rows = []
        for group in range(10):
            for member in range(3 if group == 0 else 1):
                rows.append({
                    "event_id_cnty": f"E{group}-{member}", "notes": f"note {group} {member}",
                    "clean_notes": f"note {group} {member}", "year": 2024, "country": "NL",
                    "final_label": "labor rights" if group % 2 else "other",
                    "duplicate_group_id": f"group-{group}",
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            splits = load_data(path, train_split=0.6, val_split=0.2, random_state=42)
        memberships: dict[str, set[str]] = {}
        for name, frame in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
            for group in frame.duplicate_group_id:
                memberships.setdefault(group, set()).add(name)
        self.assertTrue(memberships)
        self.assertTrue(all(len(names) == 1 for names in memberships.values()))


if __name__ == "__main__":
    unittest.main()
