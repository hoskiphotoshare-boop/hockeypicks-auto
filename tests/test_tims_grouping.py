"""Regression tests for pure-Python Tims candidate grouping."""

from __future__ import annotations

import json
import unittest

from src.tims_grouping import group_tims_candidates


class TestTimsGrouping(unittest.TestCase):
    def test_mixed_injury_status_survives(self):
        items = [
            {"group": 1, "player": "A", "injury_status": None, "injury_out": False, "final_score": 0.5},
            {"group": 1, "player": "B", "injury_status": "DTD", "injury_out": False, "final_score": 0.4},
            {"group": 1, "player": "C", "injury_status": "IR", "injury_out": True, "final_score": 0.9},
        ]
        groups = group_tims_candidates(items)
        statuses = {p["player"]: p["injury_status"] for p in groups["1"]}
        self.assertIsNone(statuses["A"])
        self.assertEqual(statuses["B"], "DTD")
        self.assertEqual(statuses["C"], "IR")
        json.dumps(groups, allow_nan=False)

    def test_keys_ascending_strings(self):
        items = [
            {"group": 2, "player": "X", "injury_out": False, "final_score": 0.1},
            {"group": 1, "player": "Y", "injury_out": False, "final_score": 0.2},
        ]
        groups = group_tims_candidates(items)
        self.assertEqual(list(groups.keys()), ["1", "2"])

    def test_sort_order_within_group(self):
        items = [
            {"group": 1, "player": "low_healthy", "injury_out": False, "final_score": 0.1},
            {"group": 1, "player": "high_healthy", "injury_out": False, "final_score": 0.9},
            {"group": 1, "player": "high_injured", "injury_out": True, "final_score": 0.99},
            {"group": 1, "player": "low_injured", "injury_out": True, "final_score": 0.2},
        ]
        groups = group_tims_candidates(items)
        names = [p["player"] for p in groups["1"]]
        self.assertEqual(names, ["high_healthy", "low_healthy", "high_injured", "low_injured"])

    def test_json_serializable(self):
        items = [{"group": 1, "player": "A", "injury_status": None, "injury_out": False, "final_score": 0.5}]
        groups = group_tims_candidates(items)
        json.dumps(groups, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
