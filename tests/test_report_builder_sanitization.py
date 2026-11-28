"""Tests for public sanitization helpers."""

from __future__ import annotations

import unittest

from src.report_builder import (
    PUBLIC_DROP_GAME_KEYS,
    sanitize_public_payload,
    sanitize_public_cup_futures,
)


class SanitizationTests(unittest.TestCase):
    def test_bookie_keys_dropped(self):
        self.assertIn("bookie", PUBLIC_DROP_GAME_KEYS)
        self.assertIn("bookie_key", PUBLIC_DROP_GAME_KEYS)

    def test_sanitize_public_payload_drops_odds_usage(self):
        payload = {
            "meta": {"odds_usage": {"x": 1}, "keep": "v"},
            "games": [{"id": 1, "bookie": "X", "bookie_key": "y", "home_team": "A"}],
            "calls_admin": {"secret": 1},
        }
        out = sanitize_public_payload(payload)
        self.assertNotIn("calls_admin", out)
        self.assertNotIn("odds_usage", out["meta"])
        self.assertEqual(out["meta"]["keep"], "v")
        self.assertNotIn("bookie", out["games"][0])
        self.assertNotIn("bookie_key", out["games"][0])
        self.assertEqual(out["games"][0]["home_team"], "A")

    def test_sanitize_public_cup_futures(self):
        payload = {
            "meta": {
                "updated_et": "x",
                "n_sims": 10,
                "odds_enabled": True,
                "odds_api": {"secret": 1},
                "bankroll": 1000,
            },
            "summary": {
                "avg_gp": 1.0,
                "max_team_pct": 2.0,
                "months_to_cup": 3.0,
                "bankroll": 1000,
                "recommended_bets": [],
            },
            "teams": [{
                "rank": 1,
                "team": "TOR",
                "best_odds": 5.0,
                "best_bookie": "X",
                "market_cup_pct": 10.0,
                "blended_cup_pct": 11.0,
                "model_cup_pct": 12.0,
            }],
        }
        out = sanitize_public_cup_futures(payload)
        self.assertFalse(out["meta"]["odds_enabled"])
        self.assertNotIn("odds_api", out["meta"])
        self.assertNotIn("bankroll", out["meta"])
        self.assertEqual(set(out["summary"].keys()), {"avg_gp", "max_team_pct", "months_to_cup"})
        self.assertEqual(set(out["teams"][0].keys()), {"rank", "team", "model_cup_pct"})
        self.assertNotIn("best_odds", out["teams"][0])


if __name__ == "__main__":
    unittest.main()
