import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.calls_ledger import (
    grade_calls,
    load_calls,
    save_calls,
    snapshot_calls,
    summarize,
    summarize_admin,
    update_calls_ledger,
    update_closing,
)

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
FUTURE = "2026-11-25T23:00:00Z"
PAST = "2026-11-24T23:00:00Z"


def make_game(**kw):
    base = {
        "id": 1,
        "date": "2026-11-25",
        "start_time_utc": FUTURE,
        "game_state": "FUT",
        "away": "BOS",
        "home": "MTL",
        "away_prob": 55.0,
        "home_prob": 45.0,
        "away_score": None,
        "home_score": None,
        "away_edge": {"odds": 2.1, "edge_pct": 5.0},
        "home_edge": {"odds": 1.9, "edge_pct": 1.0},
        "bookie": "Pinnacle",
        "bookie_key": "pinnacle",
    }
    base.update(kw)
    return base


class TestCallsLedger(unittest.TestCase):
    def test_load_save_missing_corrupt(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "x.json")
        self.assertEqual(load_calls(path), [])
        save_calls(path, [{"a": 1}])
        self.assertEqual(load_calls(path), [{"a": 1}])
        with open(path, "w") as f:
            f.write("{bad")
        self.assertEqual(load_calls(path), [])

    def test_snapshot_pick_higher_prob_and_ignores_started(self):
        games = [make_game(id=1), make_game(id=2, start_time_utc=PAST, game_state="FINAL", away_score=3, home_score=2)]
        existing = []
        n = snapshot_calls(games, existing, NOW)
        self.assertEqual(n, 1)
        self.assertEqual(len(existing), 1)
        c = existing[0]
        self.assertEqual(c["pick"], "BOS")
        self.assertAlmostEqual(c["pick_prob"], 0.55)
        self.assertEqual(c["status"], "open")
        self.assertIsNone(c["closing_prob"])
        self.assertIsNone(c["result"])
        # no sensitive keys
        for k in ("price", "bookmaker", "profit"):
            self.assertNotIn(k, c)

    def test_admin_snapshot(self):
        games = [make_game(id=1)]
        existing = []
        n = snapshot_calls(games, existing, NOW, admin=True)
        self.assertEqual(n, 1)
        c = existing[0]
        self.assertEqual(c["market"], "ml")
        self.assertEqual(c["side"], "away")
        self.assertEqual(c["team"], "BOS")
        self.assertEqual(c["price"], 2.1)
        self.assertEqual(c["bookmaker"], "Pinnacle")
        self.assertAlmostEqual(c["edge"], 0.05)
        self.assertAlmostEqual(c["model_prob"], 0.55)

    def test_no_duplicate_snapshot(self):
        games = [make_game(id=1)]
        existing = []
        snapshot_calls(games, existing, NOW)
        n2 = snapshot_calls(games, existing, NOW)
        self.assertEqual(n2, 0)
        self.assertEqual(len(existing), 1)

    def test_no_lookahead_started_not_snapshotted(self):
        games = [make_game(id=1, start_time_utc=PAST, game_state="LIVE")]
        existing = []
        n = snapshot_calls(games, existing, NOW)
        self.assertEqual(n, 0)

    def test_closing_updates_only_prestart(self):
        games = [make_game(id=1, away_prob=60.0, home_prob=40.0, away_odds=2.2, home_odds=1.8)]
        existing_pub = []
        existing_adm = []
        snapshot_calls(games, existing_pub, NOW)
        snapshot_calls(games, existing_adm, NOW, admin=True)
        # make a started version
        games2 = [make_game(id=1, start_time_utc=PAST, game_state="LIVE", away_prob=70.0, home_prob=30.0, away_odds=2.5, home_odds=1.5)]
        updated = update_closing(games2, existing_pub, existing_adm, NOW)
        self.assertEqual(updated, 0)
        self.assertIsNone(existing_pub[0]["closing_prob"])

        games3 = [make_game(id=1, away_prob=60.0, home_prob=40.0, away_odds=2.2, home_odds=1.8)]
        updated2 = update_closing(games3, existing_pub, existing_adm, NOW)
        self.assertGreater(updated2, 0)
        self.assertAlmostEqual(existing_pub[0]["closing_prob"], 0.60)

    def test_grading_idempotent_and_profit(self):
        games = [make_game(id=1, game_state="FINAL", away_score=3, home_score=2, start_time_utc=PAST)]
        existing_pub = []
        existing_adm = []
        snapshot_calls([make_game(id=1)], existing_pub, NOW)
        snapshot_calls([make_game(id=1)], existing_adm, NOW, admin=True)
        n = grade_calls(games, existing_pub, existing_adm, NOW)
        self.assertEqual(n, 2)
        self.assertEqual(existing_pub[0]["result"], "win")
        self.assertEqual(existing_pub[0]["final_score"], "3-2")
        self.assertEqual(existing_adm[0]["result"], "win")
        self.assertAlmostEqual(existing_adm[0]["profit"], 1.1)
        # idempotent
        n2 = grade_calls(games, existing_pub, existing_adm, NOW)
        self.assertEqual(n2, 0)

    def test_ot_so_counts(self):
        games = [make_game(id=1, game_state="OFF", away_score=2, home_score=3, start_time_utc=PAST)]
        existing_pub = []
        snapshot_calls([make_game(id=1)], existing_pub, NOW)
        # pick home? pick BOS (away) since away_prob 55; away loses -> loss
        grade_calls(games, existing_pub, [], NOW)
        self.assertEqual(existing_pub[0]["result"], "loss")
        self.assertEqual(existing_pub[0]["final_score"], "2-3")

    def test_fetch_scores_fallback(self):
        existing_pub = [{
            "id": 1,
            "call_id": 1,
            "date": "2026-11-24",
            "away": "BOS",
            "home": "MTL",
            "pick": "BOS",
            "pick_prob": 0.55,
            "away_prob": 0.55,
            "home_prob": 0.45,
            "kickoff": "2026-11-24T23:00:00Z",
            "snapshot_at": "2026-11-24T12:00:00+00:00",
            "status": "open",
            "closing_prob": None,
            "result": None,
            "final_score": None,
            "graded_at": None,
        }]
        def fetch_scores(date):
            return {1: {"state": "FINAL", "away_score": 4, "home_score": 1}}
        n = grade_calls([], existing_pub, [], NOW, fetch_scores=fetch_scores)
        self.assertEqual(n, 1)
        self.assertEqual(existing_pub[0]["result"], "win")
        self.assertEqual(existing_pub[0]["final_score"], "4-1")

    def test_summary_brier_streak_winrate(self):
        calls = []
        snapshot_calls([make_game(id=1)], calls, NOW)
        g = [make_game(id=1, game_state="FINAL", away_score=3, home_score=2, start_time_utc=PAST)]
        grade_calls(g, calls, [], NOW)
        s = summarize(calls)
        self.assertEqual(s["calls"], 1)
        self.assertEqual(s["graded"], 1)
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["losses"], 0)
        self.assertAlmostEqual(s["win_rate"], 1.0)
        self.assertAlmostEqual(s["brier"], (0.55 - 1) ** 2)
        self.assertEqual(s["current_streak"], {"type": "W", "n": 1})
        self.assertIn("recent", s)

    def test_admin_roi(self):
        calls = []
        snapshot_calls([make_game(id=1)], calls, NOW, admin=True)
        g = [make_game(id=1, game_state="FINAL", away_score=3, home_score=2, start_time_utc=PAST)]
        grade_calls(g, [], calls, NOW)
        s = summarize_admin(calls)
        self.assertEqual(s["graded"], 1)
        self.assertAlmostEqual(s["roi"], 1.1)
        self.assertAlmostEqual(s["profit"], 1.1)

    def test_public_no_sensitive_keys(self):
        calls = []
        snapshot_calls([make_game(id=1)], calls, NOW)
        for c in calls:
            for k in ("price", "bookmaker", "profit", "roi", "bookie"):
                self.assertNotIn(k, c)

    def test_update_calls_ledger(self):
        d = tempfile.mkdtemp()
        games = [make_game(id=1)]
        res = update_calls_ledger(d, games, NOW)
        self.assertIn("summary", res)
        self.assertIn("admin_summary", res)
        self.assertIn("recent", res)
        self.assertTrue(os.path.exists(os.path.join(d, "calls_history.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "calls_ledger.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "calls_scoreboard.json")))


if __name__ == "__main__":
    unittest.main()
