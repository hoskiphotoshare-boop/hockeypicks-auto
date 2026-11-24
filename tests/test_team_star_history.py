"""Per-game Top-3 definition and depth in the historical star table."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from src.team_game_model import (
    build_team_star_goalie_history,
    context_spans_multiple_seasons,
    extract_latest_team_form,
)


class TestStarHistoryDefinition(unittest.TestCase):
    def test_top3_and_depth_use_per_game_rates(self):
        teams = pd.DataFrame(
            [{"team": "TOR", "situation": "all", "games_played": 20}]
        )
        skaters = pd.DataFrame(
            [
                {"name": "A", "team": "TOR", "situation": "all", "games_played": 20, "I_F_points": 30, "I_F_xGoals": 15},
                {"name": "B", "team": "TOR", "situation": "all", "games_played": 20, "I_F_points": 20, "I_F_xGoals": 10},
                {"name": "C", "team": "TOR", "situation": "all", "games_played": 10, "I_F_points": 10, "I_F_xGoals": 5},
                {"name": "D", "team": "TOR", "situation": "all", "games_played": 20, "I_F_points": 4, "I_F_xGoals": 2},
                {"name": "E", "team": "TOR", "situation": "all", "games_played": 20, "I_F_points": 2, "I_F_xGoals": 1},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            teams.to_csv(os.path.join(tmp, "2026_regular_teams.csv"), index=False)
            skaters.to_csv(os.path.join(tmp, "2026_regular_skaters.csv"), index=False)
            _, df_stars, _ = build_team_star_goalie_history(tmp, [2026])
        row = df_stars[df_stars["team"] == "TOR"].iloc[0]
        # top-3 by per-game: A 1.5, B 1.0, C 1.0 (not the 60 total / 20 = 3.0)
        self.assertAlmostEqual(row["Top3_Points_per_game"], 3.5, places=6)
        self.assertAlmostEqual(row["Top3_Points"], 60.0, places=6)
        # depth = remaining skaters' per-game rates: D 0.2 + E 0.1
        self.assertAlmostEqual(row["Depth_Points_per_game"], 0.3, places=6)
        self.assertAlmostEqual(row["Depth_xGoals_per_game"], 0.15, places=6)


class TestEloContextGuard(unittest.TestCase):
    """`Causal_Elo` overrides are only valid on a multi-season frame."""

    @staticmethod
    def _logs(seasons):
        rows = []
        for i, season in enumerate(seasons):
            for j in range(4):
                home, away = ("TOR", "MTL") if j % 2 == 0 else ("MTL", "TOR")
                tor_won = 1 if (home == "TOR") == (i == 0) else 0
                rows.append(
                    {
                        "Date": f"{season}-11-{10 + j:02d}",
                        "Season": str(season),
                        "Team": home,
                        "Opponent": away,
                        "Is_Home": 1,
                        "Won": 1 if home == "TOR" and tor_won == 1 else 0,
                        "Team_Score": 5 if home == "TOR" and tor_won == 1 else 2,
                        "Opp_Score": 2 if home == "TOR" and tor_won == 1 else 4,
                        "Current_Streak": 0,
                    }
                )
        return pd.DataFrame(rows)

    def test_helper_detects_season_span(self):
        self.assertFalse(context_spans_multiple_seasons(None))
        self.assertFalse(context_spans_multiple_seasons(pd.DataFrame()))
        self.assertFalse(context_spans_multiple_seasons(self._logs([2026])))
        self.assertFalse(context_spans_multiple_seasons(pd.DataFrame([{"Date": "x", "Team": "TOR"}])))
        self.assertTrue(context_spans_multiple_seasons(self._logs([2025, 2026])))

    def test_season_only_elo_differs_from_full_history(self):
        full = extract_latest_team_form(self._logs([2025, 2026]))["TOR"]["Causal_Elo"]
        current = extract_latest_team_form(self._logs([2026]))["TOR"]["Causal_Elo"]
        self.assertGreater(abs(full - current), 3.0)


if __name__ == "__main__":
    unittest.main()
