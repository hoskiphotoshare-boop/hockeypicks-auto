"""Day-one roster membership diff (stats attribution vs current roster)."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from src.data_sync import build_roster_membership_moves, clean_name


class TestRosterMembership(unittest.TestCase):
    def _player(self, pid, name, team):
        return {"id": pid, "name": name, "team": team}

    def test_larkin_style_move_and_injury_filter(self):
        cols = ["name", "team", "situation", "games_played", "I_F_points", "I_F_xGoals"]
        rows = [
            ["Cole Caufield", "MTL", "all", 20, 19, 10],
            ["Nick Suzuki", "MTL", "all", 14, 17, 9],
            ["Lane Hutson", "MTL", "all", 20, 18, 8],
            ["Mike Matheson", "MTL", "all", 10, 3, 2],
            ["Lucas Raymond", "DET", "all", 21, 21, 12],
            ["Dylan Larkin", "DET", "all", 10, 6, 4],
            ["Simon Edvinsson", "DET", "all", 15, 4, 2],
        ]
        df = pd.DataFrame(rows, columns=cols)
        rosters = {
            "MTL": [
                self._player(1, "Dylan Larkin", "MTL"),
                self._player(2, "Cole Caufield", "MTL"),
                self._player(3, "Nick Suzuki", "MTL"),
                self._player(4, "Lane Hutson", "MTL"),
                self._player(5, "Mike Matheson", "MTL"),
                self._player(6, "Rookie Callup", "MTL"),
            ],
            "DET": [
                self._player(10, "Lucas Raymond", "DET"),
                self._player(11, "Depth Rookie", "DET"),
            ],
        }
        values = {
            clean_name("Dylan Larkin"): {
                "name": "Dylan Larkin",
                "current": {"gp": 10.0, "points_pg": 0.6},
                "prior": {"gp": 82.0, "points_pg": 1.0},
            },
        }
        injuries = [
            {"player": "Simon Edvinsson", "team": "DET", "position": "D", "is_out": True, "is_dtd": False},
            {"player": "Mike Matheson", "team": "MTL", "position": "D", "is_out": True, "is_dtd": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(os.path.join(tmp, "2026_regular_skaters.csv"), index=False)
            moves = build_roster_membership_moves(
                rosters, tmp, 2026, player_values=values, injury_records=injuries, min_value=0.15
            )

        self.assertIn("Dylan Larkin", moves["MTL"]["in"])
        self.assertNotIn("Rookie Callup", moves["MTL"]["in"])  # no value -> filtered
        self.assertIn("Dylan Larkin", moves["DET"]["out"])
        self.assertNotIn("Simon Edvinsson", moves["DET"]["out"])  # injured, not a trade
        self.assertNotIn("Mike Matheson", moves["MTL"]["out"])  # still on the roster
        self.assertEqual(moves["MTL"]["out"], [])


if __name__ == "__main__":
    unittest.main()
