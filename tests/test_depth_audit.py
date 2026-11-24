"""Unit tests for depth audit helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import collect_depth_audit as cda


class TestDepthAudit(unittest.TestCase):
    def test_select_last_pregame(self):
        snapshots = [
            {
                "meta": {"updated_utc": "2026-01-01 10:00:00 UTC"},
                "games": [
                    {"id": 1, "game_state": "FUT", "start_time_utc": "2026-01-01T12:00:00Z",
                     "away_prob": 50, "home_prob": 50, "date": "2026-01-01",
                     "away": "A", "home": "H"},
                ],
            },
            {
                "meta": {"updated_utc": "2026-01-01 11:00:00 UTC"},
                "games": [
                    {"id": 1, "game_state": "FUT", "start_time_utc": "2026-01-01T12:00:00Z",
                     "away_prob": 55, "home_prob": 45, "date": "2026-01-01",
                     "away": "A", "home": "H"},
                ],
            },
        ]
        selected = cda.select_pregame_snapshots(snapshots)
        self.assertEqual(selected[1]["away_prob"], 55)

    def test_reject_past_start(self):
        snapshots = [
            {
                "meta": {"updated_utc": "2026-01-01 13:00:00 UTC"},
                "games": [
                    {"id": 2, "game_state": "FUT", "start_time_utc": "2026-01-01T12:00:00Z",
                     "away_prob": 50, "home_prob": 50},
                ],
            },
        ]
        selected = cda.select_pregame_snapshots(snapshots)
        self.assertEqual(len(selected), 0)

    def test_live_final_not_predictions(self):
        snapshots = [
            {
                "meta": {"updated_utc": "2026-01-01 10:00:00 UTC"},
                "games": [
                    {"id": 3, "game_state": "LIVE", "start_time_utc": "2026-01-01T12:00:00Z",
                     "away_prob": 50, "home_prob": 50},
                    {"id": 4, "game_state": "FINAL", "start_time_utc": "2026-01-01T12:00:00Z",
                     "away_prob": 50, "home_prob": 50},
                ],
            },
        ]
        selected = cda.select_pregame_snapshots(snapshots)
        self.assertEqual(len(selected), 0)

    def test_outcome_join(self):
        pred_snap = {
            "meta": {"updated_utc": "2026-01-01 10:00:00 UTC"},
            "games": [
                {"id": 5, "game_state": "FUT", "start_time_utc": "2026-01-01T12:00:00Z",
                 "away_prob": 40, "home_prob": 60, "date": "2026-01-01",
                 "away": "A", "home": "H",
                 "away_prob_unadjusted": 42, "home_prob_unadjusted": 58,
                 "away_depth_eff": 0.9, "home_depth_eff": 1.0},
            ],
        }
        final_snap = {
            "meta": {"updated_utc": "2026-01-01 15:00:00 UTC"},
            "games": [
                {"id": 5, "game_state": "FINAL", "start_time_utc": "2026-01-01T12:00:00Z",
                 "away_score": 2, "home_score": 3},
            ],
        }
        selected = cda.select_pregame_snapshots([pred_snap])
        rows = cda.join_outcomes(selected, [pred_snap, final_snap])
        self.assertEqual(rows[0]["away_score"], 2)
        self.assertEqual(rows[0]["home_score"], 3)
        self.assertEqual(rows[0]["home_win"], 1)

        selected2 = cda.select_pregame_snapshots([pred_snap])
        rows2 = cda.join_outcomes(selected2, [])
        self.assertEqual(rows2[0]["home_win"], "")
        self.assertEqual(rows2[0]["away_score"], "")

    def test_summary_skips_no_shadow(self):
        rows = [
            {"has_shadow": True, "home_win": 1, "home_prob": 60, "home_prob_unadjusted": 50,
             "away_depth_eff": 0.9, "home_depth_eff": 1.0},
            {"has_shadow": False, "home_win": 0, "home_prob": 40, "home_prob_unadjusted": "",
             "away_depth_eff": 1.0, "home_depth_eff": 1.0},
        ]
        summary = cda.summarize_rows(rows)
        self.assertEqual(summary["n_shadow"], 1)

    def test_depth_subset_and_brier(self):
        rows = [
            {"has_shadow": True, "home_win": 1, "home_prob": 80, "home_prob_unadjusted": 70,
             "away_depth_eff": 0.9, "home_depth_eff": 1.0},
            {"has_shadow": True, "home_win": 0, "home_prob": 30, "home_prob_unadjusted": 40,
             "away_depth_eff": 1.0, "home_depth_eff": 1.0},
        ]
        summary = cda.summarize_rows(rows)
        self.assertEqual(summary["n_depth_affected"], 1)
        self.assertAlmostEqual(summary["overall"]["brier_adjusted"], 0.065, places=6)

    def test_cli_parser(self):
        parser = cda.build_parser()
        args = parser.parse_args(["--collect", "--summarize", "--nhl-api", "--input", "foo.csv"])
        self.assertTrue(args.collect)
        self.assertTrue(args.summarize)
        self.assertTrue(args.nhl_api)
        self.assertEqual(args.input, "foo.csv")

    def test_importable(self):
        self.assertTrue(hasattr(cda, "select_pregame_snapshots"))
        self.assertTrue(hasattr(cda, "summarize_rows"))


if __name__ == "__main__":
    unittest.main()
