"""Tests for line (puck-line + total) models."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

import numpy as np
import pandas as pd

from src.line_models import (
    predict_spread,
    predict_total,
    evaluate_line_edge,
    build_line_training_frame,
)
from src.report_builder import PUBLIC_DROP_GAME_KEYS, sanitize_public_payload


class LineModelMathTests(unittest.TestCase):
    def _dummy_bundle(self):
        # Build a minimal bundle-like object using a real trained tiny model? Instead test predict with monkeypatch.
        return None

    def test_spread_push_zero_for_half_point(self):
        # Construct a tiny bundle manually
        from src.line_models import _ConstantProb

        bundle = {
            "feature_names": [],
            "spread": {"model": _ConstantProb(0.4), "features": [], "metrics": {}, "validated": True},
            "margin": {"model": None, "residuals": [0.0, 1.0, -1.0, 2.0, -2.0], "metrics": {}, "features": []},
        }
        # margin model needs predict; use a dummy
        class M:
            def predict(self, X):
                return np.array([0.0])

        bundle["margin"]["model"] = M()
        feats = {}
        res = predict_spread(bundle, feats, -1.5)
        self.assertAlmostEqual(res["home_cover_prob"] + res["away_cover_prob"] + res["push_prob"], 1.0, places=6)
        self.assertEqual(res["push_prob"], 0.0)

    def test_spread_integer_has_push(self):
        from src.line_models import _ConstantProb

        class M:
            def predict(self, X):
                return np.array([0.0])

        bundle = {
            "feature_names": [],
            "spread": {"model": _ConstantProb(0.4), "features": [], "metrics": {}, "validated": True},
            "margin": {"model": M(), "residuals": [-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5], "metrics": {}, "features": []},
        }
        feats = {}
        res = predict_spread(bundle, feats, -1.0)
        self.assertAlmostEqual(res["home_cover_prob"] + res["away_cover_prob"] + res["push_prob"], 1.0, places=6)
        self.assertGreaterEqual(res["push_prob"], 0.0)

    def test_total_half_point_push_zero(self):
        from src.line_models import _ConstantProb

        bundle = {
            "feature_names": [],
            "total": {
                "classifiers": {
                    5: {"model": _ConstantProb(0.7), "T": 1.0},
                    6: {"model": _ConstantProb(0.5), "T": 1.0},
                    7: {"model": _ConstantProb(0.3), "T": 1.0},
                    8: {"model": _ConstantProb(0.1), "T": 1.0},
                },
                "regressor": {"model": None, "residuals": []},
                "base_rates": {},
                "brier": {},
                "log_loss": {},
                "metrics": {},
                "validated": False,
                "experimental": True,
                "stake_enabled": False,
            },
        }

        class M:
            def predict(self, X):
                return np.array([6.0])

        bundle["total"]["regressor"] = {"model": M(), "residuals": []}

        class Feats(dict):
            pass

        feats = {}
        res = predict_total(bundle, feats, 5.5)
        self.assertEqual(res["push"], 0.0)
        self.assertAlmostEqual(res["over"] + res["under"] + res["push"], 1.0, places=6)
        self.assertTrue(res["experimental"])
        self.assertFalse(res["stake_enabled"])

    def test_total_integer_push(self):
        from src.line_models import _ConstantProb

        bundle = {
            "feature_names": [],
            "total": {
                "classifiers": {
                    5: {"model": _ConstantProb(0.8), "T": 1.0},
                    6: {"model": _ConstantProb(0.6), "T": 1.0},
                    7: {"model": _ConstantProb(0.35), "T": 1.0},
                    8: {"model": _ConstantProb(0.15), "T": 1.0},
                },
                "regressor": {"model": None, "residuals": []},
                "base_rates": {},
                "brier": {},
                "log_loss": {},
                "metrics": {},
                "validated": False,
                "experimental": True,
                "stake_enabled": False,
            },
        }

        class M:
            def predict(self, X):
                return np.array([6.0])

        bundle["total"]["regressor"] = {"model": M(), "residuals": []}

        feats = {}
        res = predict_total(bundle, feats, 6.0)
        self.assertAlmostEqual(res["over"] + res["under"] + res["push"], 1.0, places=6)
        self.assertGreaterEqual(res["push"], 0.0)

    def test_evaluate_line_edge_totals_no_stake(self):
        edge = evaluate_line_edge(0.55, 2.0, stake_enabled=False)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.get("rec_bet"), 0)
        self.assertFalse(edge.get("stake_enabled"))

    def test_experimental_flags_in_keys(self):
        self.assertIn("spread_model", PUBLIC_DROP_GAME_KEYS)
        self.assertIn("total_model", PUBLIC_DROP_GAME_KEYS)


class OddsParsingTests(unittest.TestCase):
    def test_cached_odds_preserves_spread_total(self):
        payload = {
            "meta": {"line_markets_as_of": "2026-01-01T00:00:00+00:00"},
            "games": [{
                "id": 1,
                "date": "2026-01-01",
                "away": "BOS",
                "home": "TOR",
                "home_odds": 2.0,
                "away_odds": 1.9,
                "spread_market": {"line": -1.5, "best_home": {"odds": 2.1, "book": "Pinnacle"}},
                "total_market": {"line": 5.5, "best_over": {"odds": 1.95, "book": "Pinnacle"}},
            }],
        }
        from src.team_game_model import extract_cached_odds_lookup

        cached = extract_cached_odds_lookup(payload)
        entry = cached[("2026-01-01", "BOS", "TOR")]
        self.assertIn("spread", entry)
        self.assertIn("total", entry)
        self.assertEqual(entry["spread"]["line"], -1.5)


class OddsParsingFixtureTests(unittest.TestCase):
    def test_fetch_parses_spreads_totals(self):
        from src.team_game_model import _fetch_nhl_odds_lookup

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"x-requests-last": "3", "x-requests-used": "10", "x-requests-remaining": "490"}
        resp.json.return_value = [{
            "home_team": "Toronto Maple Leafs",
            "away_team": "Boston Bruins",
            "bookmakers": [{
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Toronto Maple Leafs", "price": 1.9},
                        {"name": "Boston Bruins", "price": 2.0},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Toronto Maple Leafs", "point": -1.5, "price": 2.1},
                        {"name": "Boston Bruins", "point": 1.5, "price": 1.8},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 5.5, "price": 1.95},
                        {"name": "Under", "point": 5.5, "price": 1.9},
                    ]},
                ],
            }],
        }]
        session = MagicMock()
        session.get.return_value = resp
        meta = {}
        out = _fetch_nhl_odds_lookup("key", session, markets="h2h,spreads,totals", meta_out=meta)
        self.assertTrue(out)
        entry = list(out.values())[0]
        self.assertIn("spread", entry)
        self.assertIn("total", entry)
        self.assertEqual(entry["spread"]["line"], -1.5)
        self.assertEqual(entry["total"]["line"], 5.5)
        self.assertEqual(meta.get("markets"), "h2h,spreads,totals")

    def test_public_guard_detects_spread_total_model(self):
        # ensure keys are in drop set
        self.assertIn("spread_model", PUBLIC_DROP_GAME_KEYS)
        self.assertIn("total_model", PUBLIC_DROP_GAME_KEYS)


class SanitizerTests(unittest.TestCase):
    def test_sanitize_drops_new_keys(self):
        payload = {"games": [{"id": 1, "spread_model": {"x": 1}, "total_model": {"y": 2}, "spread_market": {}, "total_market": {}}]}
        out = sanitize_public_payload(payload)
        for k in ("spread_model", "total_model", "spread_market", "total_market"):
            self.assertNotIn(k, out["games"][0])


if __name__ == "__main__":
    unittest.main()
