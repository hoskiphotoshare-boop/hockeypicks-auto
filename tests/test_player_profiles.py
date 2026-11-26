import csv
import json
import os
import tempfile
import unittest

from src.player_profiles import (
    build_players_payload,
    fetch_player_game_logs,
    parse_game_log_payload,
    read_season_skater_stats,
)


class TestPlayerProfiles(unittest.TestCase):
    def test_goal_prob_math(self):
        scorers = [{"player": "A", "team": "BOS", "position": "C", "opponent": "MTL", "base_pred": 0.2, "live_gp": 5, "live_goals": 2, "live_shots": 10, "live_g_gp": 0.4, "eb_live_weight_pct": 50, "def_multiplier": 1.0, "final_score": 0.3, "mode": "Regular Season", "injury_status": None, "injury_out": False, "injury_dtd": False, "injury_note": ""}]
        payload = build_players_payload(slate_scorers=scorers, updated_utc="2026-01-01T00:00:00Z")
        self.assertEqual(len(payload["players"]), 1)
        p = payload["players"][0]
        import math
        self.assertAlmostEqual(p["tonight"]["goal_prob"], 1 - math.exp(-0.3))

    def test_filter_gp_and_sort_cap(self):
        scorers = []
        for i in range(5):
            scorers.append({"player": f"P{i}", "team": "BOS", "position": "C", "opponent": "MTL", "base_pred": 0.1, "live_gp": 2 if i == 0 else 5, "live_goals": 1, "live_shots": 5, "live_g_gp": 0.2, "eb_live_weight_pct": 50, "def_multiplier": 1.0, "final_score": 0.5 - i * 0.1, "mode": "Regular", "injury_status": None, "injury_out": False, "injury_dtd": False, "injury_note": ""})
        payload = build_players_payload(slate_scorers=scorers)
        # first filtered out (gp<3)
        self.assertEqual(len(payload["players"]), 4)
        # sorted by goal_rate desc
        rates = [p["tonight"]["goal_rate"] for p in payload["players"]]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_last6_ordering_limit(self):
        logs = {1: [{"date": "2026-01-01", "opp": "MTL", "g": 1, "a": 0, "p": 1, "sog": 3, "toi": "18:00"}, {"date": "2026-01-05", "opp": "TOR", "g": 0, "a": 1, "p": 1, "sog": 2, "toi": "17:00"}, {"date": "2026-01-03", "opp": "OTT", "g": 2, "a": 0, "p": 2, "sog": 4, "toi": "19:00"}]}
        scorers = [{"player": "A", "team": "BOS", "position": "C", "opponent": "MTL", "base_pred": 0.1, "live_gp": 5, "live_goals": 1, "live_shots": 5, "live_g_gp": 0.2, "eb_live_weight_pct": 50, "def_multiplier": 1.0, "final_score": 0.4, "mode": "Regular", "injury_status": None, "injury_out": False, "injury_dtd": False, "injury_note": "", "playerId": 1}]
        payload = build_players_payload(slate_scorers=scorers, game_logs=logs)
        p = payload["players"][0]
        self.assertEqual(len(p["last6"]), 3)
        # newest-last by date
        self.assertEqual([r["date"] for r in p["last6"]], ["2026-01-01", "2026-01-03", "2026-01-05"])
        # cap 6
        many = [{"date": f"2026-01-{i:02d}", "opp": "X", "g": 0, "a": 0, "p": 0, "sog": 0, "toi": ""} for i in range(1, 10)]
        logs2 = {1: many}
        payload2 = build_players_payload(slate_scorers=scorers, game_logs=logs2)
        self.assertEqual(len(payload2["players"][0]["last6"]), 6)

    def test_read_season_skater_stats(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "2025_regular_skaters.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "situation", "games_played", "I_F_goals", "I_F_assists", "I_F_points", "I_F_shotsOnGoal", "icetime"])
            w.writerow(["Player A", "all", 10, 5, 3, 8, 30, 12000])
            w.writerow(["Player A", "pp", 10, 2, 1, 3, 0, 0])
            w.writerow(["Player B", "all", 4, 1, 1, 2, 8, 2400])
        stats = read_season_skater_stats(d, 2025)
        self.assertIn("player a", stats)
        s = stats["player a"]
        self.assertEqual(s["gp"], 10)
        self.assertEqual(s["goals"], 5)
        self.assertEqual(s["assists"], 3)
        self.assertEqual(s["points"], 8)
        self.assertEqual(s["sog"], 30)
        self.assertAlmostEqual(s["toi_pg"], 12000 / 10 / 60)
        self.assertEqual(s["pp_points"], 3)
        self.assertEqual(stats["player b"]["pp_points"], 0)

    def test_builder_maps_fields_and_missing(self):
        scorers = [{"player": "P", "team": "TOR", "position": "LW", "opponent": "MTL", "base_pred": 0.1, "live_gp": 5, "live_goals": 1, "live_shots": 5, "live_g_gp": 0.2, "eb_live_weight_pct": 50, "def_multiplier": 1.1, "final_score": 0.35, "mode": "Regular", "injury_status": "DTD", "injury_out": False, "injury_dtd": True, "injury_note": "day to day"}]
        schedule = [{"away": "TOR", "home": "MTL", "game_state": "FUT", "start_time_utc": "2026-01-01T00:00:00Z"}]
        starting_goalies = {"MTL": "Montembeault"}
        tims_groups = {"P": 2}
        season_stats = {"p": {"gp": 20, "goals": 10, "assists": 5, "points": 15, "sog": 60, "toi_pg": 18.5, "pp_points": 4}}
        payload = build_players_payload(
            slate_scorers=scorers,
            season_stats=season_stats,
            game_logs={},
            tims_groups=tims_groups,
            schedule_games=schedule,
            starting_goalies=starting_goalies,
            updated_utc="2026-01-01T00:00:00Z",
        )
        p = payload["players"][0]
        self.assertEqual(p["name"], "P")
        self.assertEqual(p["team"], "TOR")
        self.assertEqual(p["gp"], 20)
        self.assertEqual(p["tims_group"], 2)
        self.assertEqual(p["tonight"]["opp"], "MTL")
        self.assertFalse(p["tonight"]["home"])
        self.assertEqual(p["tonight"]["state"], "FUT")
        self.assertEqual(p["tonight"]["goalie"], "Montembeault")
        self.assertEqual(p["injury"]["status"], "DTD")
        self.assertTrue(p["injury"]["dtd"])
        # missing stats handled
        payload2 = build_players_payload(slate_scorers=[{**scorers[0], "player": "Q", "live_gp": 5}], season_stats={}, game_logs={}, schedule_games=[], starting_goalies={}, updated_utc="x")
        q = payload2["players"][0]
        self.assertEqual(q["gp"], 0)
        self.assertEqual(q["last6"], [])

    def test_game_log_parser_fake_payload(self):
        payload = {"gameLog": [
            {"gameDate": "2026-01-02", "opponentAbbrev": "MTL", "goals": 1, "assists": 1, "points": 2, "shots": 4, "toi": "18:30"},
            {"gameDate": "2026-01-01", "opponentAbbrev": "TOR", "goals": 0, "assists": 0, "points": 0, "shots": 2, "toi": "17:00"},
        ]}
        rows = parse_game_log_payload(payload)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["opp"], "TOR")
        self.assertEqual(rows[1]["g"], 1)
        self.assertEqual(rows[1]["toi"], "18:30")

    def test_fetch_player_game_logs_signature_no_network(self):
        # should not raise with no session
        res = fetch_player_game_logs([1, 2], "20252026", 2, None, os.path.join(tempfile.mkdtemp(), "cache.json"))
        self.assertIsInstance(res, dict)


if __name__ == "__main__":
    unittest.main()
