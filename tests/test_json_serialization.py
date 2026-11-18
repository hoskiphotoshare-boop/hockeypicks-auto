"""Regression tests for JSON report serialization."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.report_builder import write_json_report


class TestJsonSerialization(unittest.TestCase):
    def test_nan_raises_and_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with self.assertRaises(ValueError):
                write_json_report({"x": float("nan")}, path)
            self.assertFalse(os.path.exists(path))

    def test_clean_none_roundtrips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            write_json_report({"injury_status": None}, path)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn('"injury_status": null', text)
            self.assertNotIn("NaN", text)
            loaded = json.loads(text)
            self.assertIsNone(loaded["injury_status"])
            json.dumps(loaded, allow_nan=False)

    def test_allow_nan_false_on_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            write_json_report({"a": 1, "b": [2, 3]}, path)
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            json.dumps(loaded, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
