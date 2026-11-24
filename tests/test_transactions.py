"""Tests for roster/trade ingestion, roster-aware top-3, and churn shrink."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.data_sync import build_player_value_table, build_roster_aware_star_table, clean_name
from src.team_game_model import (
    _shrink_prob_for_churn,
    apply_roster_stars_to_profiles,
)
from src.transactions import (
    compute_team_churn,
    diff_rosters,
    fetch_espn_transactions,
    load_latest_roster_history,
    load_roster_snapshot,
    write_roster_history,
)


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _player(pid, name, team):
    return {"id": pid, "name": name, "position": "C", "team": team}


class TestTransactions(unittest.TestCase):
    def test_espn_aliases_and_trade_flag(self):
        payload = {
            "pageCount": 1,
            "transactions": [
                {
                    "date": "2026-11-22T08:00Z",
                    "description": "Acquired F Dylan Larkin in a trade with Detroit.",
                    "team": {"abbreviation": "SJ"},
                }
            ],
        }
        with mock.patch("src.transactions._fetch_with_retries", return_value=_FakeResponse(payload)):
            txs = fetch_espn_transactions(session=object())
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0]["team"], "SJS")
        self.assertTrue(txs[0]["is_trade"])

    def test_diff_rosters_detects_both_directions(self):
        prev = {
            "teams": {
                "MTL": [_player(1, "Dylan Larkin", "MTL"), _player(3, "Nick Suzuki", "MTL")],
                "DET": [_player(2, "Kirby Dach", "DET")],
            }
        }
        curr = {
            "teams": {
                "MTL": [_player(2, "Kirby Dach", "MTL"), _player(3, "Nick Suzuki", "MTL")],
                "DET": [_player(1, "Dylan Larkin", "DET")],
            }
        }
        moves = diff_rosters(prev, curr)
        self.assertEqual(moves["moves"][1]["from"], "MTL")
        self.assertEqual(moves["moves"][1]["to"], "DET")
        self.assertEqual(moves["moves"][2]["from"], "DET")
        self.assertEqual(moves["moves"][2]["to"], "MTL")
        self.assertIn("Dylan Larkin", moves["teams"]["MTL"]["out"])
        self.assertIn("Dylan Larkin", moves["teams"]["DET"]["in"])
        # unchanged player ignored
        self.assertNotIn(3, moves["moves"])

    def test_roster_history_roundtrip(self):
        rosters = {"MTL": [_player(1, "Dylan Larkin", "MTL")]}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_roster_history(rosters, tmp, "2026-11-24", captured_utc="2026-11-24T12:00:00+00:00")
            self.assertTrue(path.endswith(os.path.join("roster_history", "2026-11-24.json")))
            snap = load_latest_roster_history(tmp, before_date="2026-11-25")
            self.assertIsNotNone(snap)
            self.assertEqual(snap["teams"]["MTL"][0]["name"], "Dylan Larkin")
            # strictly before the same day -> nothing
            self.assertIsNone(load_latest_roster_history(tmp, before_date="2026-11-24"))
            # same-day overwrite
            write_roster_history({"MTL": []}, tmp, "2026-11-24")
            snap2 = load_latest_roster_history(tmp, before_date="2026-11-25")
            self.assertEqual(snap2["teams"]["MTL"], [])

    def test_roster_snapshot_freshness(self):
        from datetime import datetime, timedelta, timezone

        with tempfile.TemporaryDirectory() as tmp:
            fresh = datetime.now(timezone.utc).isoformat()
            write_roster_history({"MTL": []}, tmp, "2026-11-24", captured_utc=fresh)
            self.assertIsNotNone(load_roster_snapshot(tmp, "2026-11-24", max_age_minutes=120))
            old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
            write_roster_history({"MTL": []}, tmp, "2026-11-24", captured_utc=old)
            self.assertIsNone(load_roster_snapshot(tmp, "2026-11-24", max_age_minutes=120))

    def test_compute_team_churn_blended_and_trade_filter(self):
        values = {
            clean_name("Dylan Larkin"): {
                "name": "Dylan Larkin",
                "current": {"gp": 10.0, "points_pg": 0.6},
                "prior": {"gp": 82.0, "points_pg": 1.0},
            },
            clean_name("Kirby Dach"): {
                "name": "Kirby Dach",
                "current": {"gp": 20.0, "points_pg": 0.4},
                "prior": {"gp": 60.0, "points_pg": 0.5},
            },
            clean_name("Depth Guy"): {
                "name": "Depth Guy",
                "current": {"gp": 10.0, "points_pg": 0.1},
                "prior": {"gp": 40.0, "points_pg": 0.1},
            },
        }
        moves = {
            "moves": {
                1: {"name": "Dylan Larkin", "from": "DET", "to": "MTL"},
                2: {"name": "Kirby Dach", "from": "MTL", "to": "DET"},
                3: {"name": "Depth Guy", "from": "MTL", "to": "DET"},
            }
        }
        trades = [
            {
                "is_trade": True,
                "description": "Acquired F Dylan Larkin ... in exchange for: C Kirby Dach, ...",
            }
        ]
        churn = compute_team_churn(moves, values, recent_trades=trades)
        # Larkin blended = (1-1/3)*1.0 + (1/3)*0.6 = 0.8667; Dach = 0.45
        self.assertAlmostEqual(churn["MTL"], 0.8667 - 0.45, places=3)
        self.assertAlmostEqual(churn["DET"], 0.8667 - 0.45, places=3)
        # Depth Guy is not in any trade description -> ignored with trades present
        churn_all = compute_team_churn(moves, values)
        # The unmentioned Depth Guy move reduces the net churn magnitude.
        self.assertLess(churn_all["MTL"], churn["MTL"])

    def test_player_value_table_aggregates_team_rows(self):
        cols = ["name", "team", "situation", "games_played", "I_F_points", "I_F_xGoals", "icetime"]
        cur_rows = [
            {"name": "Dylan Larkin", "team": "DET", "situation": "all", "games_played": 10, "I_F_points": 6, "I_F_xGoals": 5, "icetime": 6000},
            {"name": "Dylan Larkin", "team": "MTL", "situation": "all", "games_played": 2, "I_F_points": 2, "I_F_xGoals": 1, "icetime": 1200},
            {"name": "Dylan Larkin", "team": "MTL", "situation": "5on5", "games_played": 2, "I_F_points": 1, "I_F_xGoals": 1, "icetime": 900},
        ]
        prior_rows = [
            {"name": "Dylan Larkin", "team": "DET", "situation": "all", "games_played": 82, "I_F_points": 80, "I_F_xGoals": 30, "icetime": 80000},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(cur_rows, columns=cols).to_csv(os.path.join(tmp, "2026_regular_skaters.csv"), index=False)
            pd.DataFrame(prior_rows, columns=cols).to_csv(os.path.join(tmp, "2025_regular_skaters.csv"), index=False)
            table = build_player_value_table(tmp, [2025, 2026])
        entry = table[clean_name("Dylan Larkin")]
        self.assertEqual(entry["current"]["gp"], 12.0)
        self.assertAlmostEqual(entry["current"]["points"], 8.0)
        self.assertAlmostEqual(entry["current"]["points_pg"], 8.0 / 12.0)
        self.assertAlmostEqual(entry["prior"]["points_pg"], 80.0 / 82.0)

    def test_roster_aware_star_table_moves_star(self):
        keys = {
            "Dylan Larkin": {"cur_gp": 10, "cur_pg": 0.6, "pr_gp": 82, "pr_pg": 1.0},
            "Cole Caufield": {"cur_gp": 20, "cur_pg": 0.95, "pr_gp": 82, "pr_pg": 0.8},
            "Nick Suzuki": {"cur_gp": 14, "cur_pg": 1.214, "pr_gp": 82, "pr_pg": 0.9},
            "Juraj Slafkovsky": {"cur_gp": 20, "cur_pg": 0.65, "pr_gp": 82, "pr_pg": 0.5},
            "Lucas Raymond": {"cur_gp": 21, "cur_pg": 1.0, "pr_gp": 82, "pr_pg": 0.9},
            "Alex DeBrincat": {"cur_gp": 21, "cur_pg": 0.762, "pr_gp": 82, "pr_pg": 0.8},
            "Kirby Dach": {"cur_gp": 20, "cur_pg": 0.4, "pr_gp": 60, "pr_pg": 0.5},
            "Depth Guy": {"cur_gp": 20, "cur_pg": 0.2, "pr_gp": 82, "pr_pg": 0.2},
        }
        values = {}
        for name, v in keys.items():
            values[clean_name(name)] = {
                "name": name,
                "current": {"gp": float(v["cur_gp"]), "points_pg": v["cur_pg"]},
                "prior": {"gp": float(v["pr_gp"]), "points_pg": v["pr_pg"]},
            }
        rosters = {
            "MTL": [_player(i, n, "MTL") for i, n in enumerate(["Dylan Larkin", "Cole Caufield", "Nick Suzuki", "Juraj Slafkovsky"])],
            "DET": [_player(i + 10, n, "DET") for i, n in enumerate(["Lucas Raymond", "Alex DeBrincat", "Kirby Dach", "Depth Guy"])],
        }
        table = build_roster_aware_star_table(rosters, values, prior_weight_gp=20.0)
        # MTL top-3 = Larkin + Caufield + Suzuki
        larkin = (1 - 10 / 30) * 1.0 + (10 / 30) * 0.6
        caufield = (1 - 20 / 40) * 0.8 + (20 / 40) * 0.95
        suzuki = (1 - 14 / 34) * 0.9 + (14 / 34) * 1.214
        self.assertAlmostEqual(table["MTL"]["Top3_Points_per_game"], larkin + caufield + suzuki, places=3)
        # DET top-3 = Raymond + DeBrincat + Dach (Larkin not on the roster)
        raymond = (1 - 21 / 41) * 0.9 + (21 / 41) * 1.0
        debrincat = (1 - 21 / 41) * 0.8 + (21 / 41) * 0.762
        dach = (1 - 20 / 40) * 0.5 + (20 / 40) * 0.4
        self.assertAlmostEqual(table["DET"]["Top3_Points_per_game"], raymond + debrincat + dach, places=3)
        # moving Larkin to DET would raise DET and lower MTL
        rosters2 = {
            "MTL": [_player(2, "Cole Caufield", "MTL"), _player(3, "Nick Suzuki", "MTL"), _player(4, "Juraj Slafkovsky", "MTL")],
            "DET": [_player(1, "Dylan Larkin", "DET"), _player(10, "Lucas Raymond", "DET"), _player(11, "Alex DeBrincat", "DET"), _player(12, "Kirby Dach", "DET")],
        }
        table2 = build_roster_aware_star_table(rosters2, values, prior_weight_gp=20.0)
        self.assertLess(table2["MTL"]["Top3_Points_per_game"], table["MTL"]["Top3_Points_per_game"])
        self.assertGreater(table2["DET"]["Top3_Points_per_game"], table["DET"]["Top3_Points_per_game"])
        self.assertGreater(table["MTL"]["Depth_Points_per_game"], 0.0)
        self.assertAlmostEqual(table["DET"]["Depth_Points_per_game"], 0.2, places=3)

    def test_apply_roster_stars_scales_totals(self):
        profiles = {
            "TOR": {
                "games_played": 20,
                "Top3_Points_per_game": 2.5,
                "Top3_xGoals_per_game": 1.1,
                "Top3_Points": 50.0,
                "Top3_xGoals": 22.0,
            }
        }
        deltas = apply_roster_stars_to_profiles(
            profiles, {"TOR": {"Top3_Points_per_game": 3.0, "Top3_xGoals_per_game": 1.2}}
        )
        self.assertAlmostEqual(deltas["TOR"], 0.5, places=6)
        self.assertEqual(profiles["TOR"]["Top3_Points_per_game"], 3.0)
        self.assertEqual(profiles["TOR"]["Top3_xGoals_per_game"], 1.2)
        self.assertEqual(profiles["TOR"]["Top3_Points"], 60.0)
        self.assertEqual(profiles["TOR"]["Top3_xGoals"], 24.0)
        # missing team / None -> no crash
        apply_roster_stars_to_profiles(profiles, {"MTL": {"Top3_Points_per_game": 9.0}})
        self.assertEqual(apply_roster_stars_to_profiles(profiles, None), {})
        self.assertEqual(profiles["TOR"]["Top3_Points"], 60.0)

    def test_apply_roster_stars_reports_negative_delta(self):
        profiles = {"TOR": {"games_played": 20, "Top3_Points_per_game": 3.1, "Top3_xGoals_per_game": 1.2}}
        deltas = apply_roster_stars_to_profiles(profiles, {"TOR": {"Top3_Points_per_game": 2.8}})
        self.assertAlmostEqual(deltas["TOR"], -0.3, places=6)

    def test_shrink_prob_for_churn(self):
        self.assertAlmostEqual(_shrink_prob_for_churn(0.70, 0.0, 0.0), 0.70)
        self.assertAlmostEqual(_shrink_prob_for_churn(0.70, 1.0, 0.0), 0.65)
        self.assertAlmostEqual(_shrink_prob_for_churn(0.70, 0.0, 1.0), 0.65)
        self.assertAlmostEqual(_shrink_prob_for_churn(0.30, 1.0, 0.0), 0.35)
        # clamps churn to [0, 1]
        self.assertAlmostEqual(_shrink_prob_for_churn(0.70, 5.0, 0.0), 0.65)


if __name__ == "__main__":
    unittest.main()
