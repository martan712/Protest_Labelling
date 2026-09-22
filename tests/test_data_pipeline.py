from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from protest_classifier.data.annotations import assemble, write_annotation_chunks
from protest_classifier.data.contracts import assert_disjoint, assert_nested
from protest_classifier.data.manifests import load_event_pool, select_random, write_manifests
from protest_classifier.text import note_hash


class DataPipelineTests(unittest.TestCase):
    def test_random_manifests_are_nested_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(40):
                rows.append({
                    "event_id_cnty": f"E{index}", "event_date": "2025-01-01",
                    "year": 2025, "country": "X", "notes": f"note {index}",
                })
            source = root / "events.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            out = root / "manifests"
            splits = select_random(
                load_event_pool(source), dev_size=5, test_size=5, train_sizes=(10, 20)
            )
            write_manifests(splits, out)
            small = pd.read_csv(out / "train_0010.csv")
            large = pd.read_csv(out / "train_0020.csv")
            dev = pd.read_csv(out / "dev.csv")
            test = pd.read_csv(out / "test_locked.csv")
            assert_nested(small, large, smaller_name="small", larger_name="large")
            assert_disjoint(large, dev, left_name="train", right_name="dev")
            assert_disjoint(large, test, left_name="train", right_name="test")
            assert_disjoint(dev, test, left_name="dev", right_name="test")

    def test_annotation_assembly_preserves_manifest_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = "Farm workers demand higher wages."
            manifest = pd.DataFrame([{
                "event_id_cnty": "E1", "event_date": "2025-01-01", "year": "2025",
                "country": "X", "notes": notes, "note_hash": note_hash(notes),
            }])
            labels = pd.DataFrame([{
                "event_id_cnty": "E1", "primary_label": "labor rights and wages",
                "alternative_labels": "", "other_reason": "",
                "evidence": "higher wages", "annotator": "test",
            }])
            result = assemble(manifest, labels)
            self.assertEqual(result.loc[0, "notes"], notes)
            self.assertEqual(result.loc[0, "primary_label"], "labor rights and wages")

            paths = write_annotation_chunks(manifest, root / "chunks")
            queue = pd.read_csv(paths[0], keep_default_na=False)
            self.assertEqual(queue.loc[0, "notes"], notes)


if __name__ == "__main__":
    unittest.main()
