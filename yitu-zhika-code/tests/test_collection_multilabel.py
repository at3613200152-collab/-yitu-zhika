import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import app.collection as collection
from app.collection import normalise_corrected_categories


class CorrectedCategoriesTest(unittest.TestCase):
    def test_accepts_and_deduplicates_multi_labels(self):
        values = normalise_corrected_categories({
            "corrected_categories": ["vegetable", "meat", "vegetable"]
        })
        self.assertEqual(values, ["vegetable", "meat"])

    def test_converts_legacy_single_label(self):
        values = normalise_corrected_categories({"corrected_category": "fruit"})
        self.assertEqual(values, ["fruit"])

    def test_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "不是合法类别"):
            normalise_corrected_categories({"corrected_categories": ["vegetable", "hotpot"]})

    def test_save_record_keeps_multi_labels_separate_from_legacy_single_label(self):
        original_db = collection.DB
        with tempfile.TemporaryDirectory() as tmp:
            collection.DB = Path(tmp) / "collection.sqlite3"
            try:
                collection.init()
                record_id, source, consent = collection.save_record({
                    "dish_uuid": "dish-1",
                    "quality": "confirm_only",
                    "corrected_categories": ["vegetable", "meat"],
                    "category_label_schema": "v2_fruit_12class",
                    "training_consent": False,
                }, "127.0.0.1")
                db = sqlite3.connect(str(collection.DB))
                try:
                    row = db.execute(
                        "SELECT label_value, label_source FROM records_v1 WHERE record_id = ?",
                        (record_id,)
                    ).fetchone()
                finally:
                    db.close()
                value = json.loads(row[0])
                self.assertEqual(value["corrected_categories"], ["vegetable", "meat"])
                self.assertIsNone(value["corrected_category"])
                self.assertEqual(row[1], "user_estimate")
                self.assertEqual(source, "user_estimate")
                self.assertFalse(consent)
            finally:
                collection.DB = original_db


if __name__ == "__main__":
    unittest.main()
