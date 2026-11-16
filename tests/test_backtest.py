"""Unit tests for walk-forward Tim Hortons backtest logic."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from backtest_tims import (
    evaluate_walk_forward_backtest,
    extract_boxscore_skaters,
    parse_toi_seconds,
)
from src.skater_model import HAS_SKATER_ML
from src.team_game_model import HAS_GAME_ML


class TestBacktestHelpers(unittest.TestCase):
    def test_parse_toi_seconds(self):
        self.assertEqual(parse_toi_seconds("18:42"), 1122)
        self.assertEqual(parse_toi_seconds("00:00"), 0)
        self.assertEqual(parse_toi_seconds(None), 0)
        self.assertEqual(parse_toi_seconds("bad"), 0)

    def test_extract_boxscore_skaters_filters_zero_toi(self):
        payload = {
            "playerByGameStats": {
                "awayTeam": {
                    "forwards": [
                        {
                            "playerId": 97,
                            "name": {"default": "Connor McDavid"},
                            "position": "C",
                            "goals": 2,
                            "sog": 5,
                            "toi": "21:15",
                        },
                        {
                            "playerId": 99,
                            "name": {"default": "Healthy Scratch"},
                            "position": "L",
                            "goals": 0,
                            "sog": 0,
                            "toi": "00:00",
                        },
                    ],
                    "defense": [],
                },
                "homeTeam": {
                    "forwards": [
                        {
                            "playerId": 34,
                            "name": {"default": "Auston Matthews"},
                            "position": "C",
                            "goals": 1,
                            "shots": 6,
                            "toi": "20:05",
                        }
                    ],
                    "defense": [],
                },
            }
        }
        meta = {
            "season": "20252026",
            "game_type": 2,
            "game_id": 1001,
            "game_date": "2025-11-10",
            "away_team": "EDM",
            "home_team": "TOR",
            "away_score": 4,
            "home_score": 2,
        }
        rows = extract_boxscore_skaters(payload, meta)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["player_id"], 97)
        self.assertEqual(rows[0]["goals"], 2)
        self.assertEqual(rows[0]["team_goals_against"], 2)
        self.assertEqual(rows[1]["player_id"], 34)
        self.assertEqual(rows[1]["team_goals_against"], 4)

    @unittest.skipUnless(HAS_SKATER_ML and HAS_GAME_ML, "xgboost/sklearn not installed in current Python runtime")
    def test_walk_forward_backtest_synthetic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_dir = os.path.join(tmpdir, "moneypuck")
            os.makedirs(mp_dir, exist_ok=True)
            rng = np.random.default_rng(7)
            for y in (2022, 2023, 2024):
                sk = pd.DataFrame(
                    {
                        "playerId": np.arange(1, 45),
                        "season": y,
                        "name": [f"Skater {i}" for i in range(1, 45)],
                        "team": ["EDM" if i <= 22 else "TOR" for i in range(1, 45)],
                        "position": "C",
                        "situation": "all",
                        "games_played": 40,
                        "I_F_goals": np.linspace(2, 28, 44).astype(int),
                        "I_F_points": np.linspace(8, 68, 44).astype(int),
                        "I_F_xGoals": np.linspace(3.0, 26.0, 44),
                        "I_F_shotsOnGoal": np.linspace(30, 180, 44).astype(int),
                        "onIce_xGoalsPercentage": rng.uniform(0.45, 0.56, 44),
                    }
                )
                sk.to_csv(os.path.join(mp_dir, f"{y}_regular_skaters.csv"), index=False)
                tm = pd.DataFrame(
                    {
                        "team": ["EDM", "TOR"],
                        "season": y,
                        "situation": "all",
                        "games_played": [40, 40],
                        "goalsAgainst": [95, 115],
                        "xGoalsAgainst": [98.0, 112.0],
                        "highDangerxGoalsAgainst": [30.0, 38.0],
                        "xGoalsFor": [125.0, 105.0],
                    }
                )
                tm.to_csv(os.path.join(mp_dir, f"{y}_regular_teams.csv"), index=False)
                go = pd.DataFrame(
                    {
                        "team": ["EDM", "TOR"],
                        "situation": "all",
                        "games_played": [30, 30],
                        "goalsSavedAboveExpected": [6.0, -3.0],
                        "savePercentage": [0.914, 0.895],
                    }
                )
                go.to_csv(os.path.join(mp_dir, f"{y}_regular_goalies.csv"), index=False)

            logs = []
            for i in range(60):
                d = f"2024-{(i // 28) + 10:02d}-{(i % 28) + 1:02d}"
                w = 1 if i % 2 == 0 else 0
                logs.append({"Season": "20242025", "Team": "EDM", "Date": d, "GameType": 2, "Opponent": "TOR", "Is_Home": 1, "Team_Score": 4, "Opp_Score": 2, "Won": w, "Last_Game_Won": 1, "Current_Streak": 1})
                logs.append({"Season": "20242025", "Team": "TOR", "Date": d, "GameType": 2, "Opponent": "EDM", "Is_Home": 0, "Team_Score": 2, "Opp_Score": 4, "Won": 1 - w, "Last_Game_Won": 0, "Current_Streak": -1})
            df_logs = pd.DataFrame(logs)

            # Build 4 nights of 3 games each with 42 dressed skaters per night
            teams = [("EDM", "TOR"), ("COL", "SJS"), ("FLA", "BUF")]
            box_rows = []
            for day_idx in range(4):
                gd = f"2025-11-{10 + day_idx:02d}"
                for g_off, (away, home) in enumerate(teams):
                    gid = 2000 + day_idx * 10 + g_off
                    for p_offset in range(14):
                        pid = (g_off * 14) + p_offset + 1
                        is_h = 1 if p_offset >= 7 else 0
                        tm_code = home if is_h else away
                        opp_code = away if is_h else home
                        scored = 1 if pid % 11 == day_idx else 0
                        box_rows.append(
                            {
                                "season": "20252026",
                                "game_type": 2,
                                "game_id": gid,
                                "game_date": gd,
                                "team": tm_code,
                                "opponent": opp_code,
                                "is_home": is_h,
                                "team_goals_against": 3,
                                "player_id": pid,
                                "name": f"Skater {pid}",
                                "name_clean": f"skater {pid}",
                                "position": "C",
                                "goals": scored,
                                "shots": 3,
                                "toi_sec": 1000,
                            }
                        )
            df_box = pd.DataFrame(box_rows)
            res = evaluate_walk_forward_backtest(
                label="Synthetic",
                df_box=df_box,
                archive_dir=mp_dir,
                df_master_logs=df_logs,
                skater_mode="regular",
                skater_target_years=[2023, 2024],
                baseline_files=["2024_regular_skaters.csv"],
                pairwise_years=[2022, 2023, 2024],
                profile_year=2024,
                max_log_season="20242025",
            )
            self.assertEqual(res["tims_3tier_slates"], 4)
            self.assertIn("ai_full", res["tiers"]["group_1"])


if __name__ == "__main__":
    unittest.main()
