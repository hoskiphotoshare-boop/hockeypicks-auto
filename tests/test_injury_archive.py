"""Tests for the daily injury archive (docs/injury_history/YYYY-MM-DD.json)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.availability import write_injury_history


def _rec(player="Sebastian Aho", team="CAR", position="C", status="Out", is_out=True, is_dtd=False, note=""):
    return {
        "player": player,
        "team": team,
        "position": position,
        "status": status,
        "is_out": is_out,
        "is_dtd": is_dtd,
        "note": note,
    }


class TestInjuryArchive(unittest.TestCase):
    def test_writes_daily_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_injury_history(
                [_rec()],
                tmp,
                "2026-11-24",
                captured_utc="2026-11-24T16:30:00+00:00",
            )
            self.assertIsNotNone(path)
            self.assertTrue(path.endswith(os.path.join("injury_history", "2026-11-24.json")))
            data = json.load(open(path, encoding="utf-8"))
            self.assertEqual(data["date"], "2026-11-24")
            self.assertEqual(data["captured_utc"], "2026-11-24T16:30:00+00:00")
            self.assertEqual(data["source"], "espn-cbs")
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["records"][0]["player"], "Sebastian Aho")
            self.assertTrue(data["records"][0]["is_out"])

    def test_same_day_rewrite_updates_without_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_injury_history([_rec()], tmp, "2026-11-24", captured_utc="2026-11-24T11:00:00+00:00")
            path = write_injury_history(
                [_rec(player="A"), _rec(player="B", is_out=False, is_dtd=True, status="DTD")],
                tmp,
                "2026-11-24",
                captured_utc="2026-11-24T16:30:00+00:00",
            )
            data = json.load(open(path, encoding="utf-8"))
            self.assertEqual(data["count"], 2)
            self.assertEqual(len(data["records"]), 2)
            self.assertEqual(data["captured_utc"], "2026-11-24T16:30:00+00:00")
            dtd = [r for r in data["records"] if r["is_dtd"]]
            self.assertEqual(dtd[0]["status"], "DTD")

    def test_empty_records_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(write_injury_history([], tmp, "2026-11-24"))
            self.assertFalse(os.path.exists(os.path.join(tmp, "injury_history")))

    def test_accepts_injury_history_directory_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "injury_history")
            path = write_injury_history([_rec()], target, "2026-11-25")
            self.assertEqual(path, os.path.join(target, "2026-11-25.json"))
            self.assertTrue(os.path.exists(path))

    def test_normalizes_alternate_field_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = {"name": "Nazem Kadri", "team": "CGY", "position": "C", "status": "IR", "is_out": True, "comment": "upper body"}
            path = write_injury_history([rec], tmp, "2026-11-24")
            data = json.load(open(path, encoding="utf-8"))
            self.assertEqual(data["records"][0]["player"], "Nazem Kadri")
            self.assertEqual(data["records"][0]["note"], "upper body")


if __name__ == "__main__":
    unittest.main()
