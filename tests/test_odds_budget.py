"""Unit tests for The Odds API 1-credit bookmakers query, live-loop cap, and cached odds reuse."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import main_pipeline
import src.live_loop as live_loop
from src.team_game_model import (
    ODDS_API_BOOKMAKERS,
    _fetch_nhl_odds_lookup,
    extract_cached_odds_lookup,
    predict_upcoming_games,
)

ET_ZONE = ZoneInfo("America/New_York")


class _FakeOddsResponse:
    def __init__(self, payload, *, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeOddsSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, timeout: int = 15):
        self.urls.append(url)
        if not self._responses:
            return _FakeOddsResponse([], status_code=200)
        return self._responses.pop(0)


class _DummyGameClf:
    def predict_proba(self, X):
        p = np.full(len(X), 0.65, dtype=float)
        return np.column_stack([1.0 - p, p])


class _DummySkaterReg:
    def predict(self, X):
        return np.full(len(X), 0.35, dtype=float)


class _DummyImputer:
    def transform(self, X):
        return X


def _sample_profiles() -> dict:
    return {
        "EDM": {
            "Elo": 1540.0,
            "Causal_Elo": 1540.0,
            "Last_Game_Date": "2026-11-18",
            "games_played": 15,
            "goalsAgainst": 35.0,
            "xGoalsAgainst": 35.0,
            "highDangerxGoalsAgainst": 15.0,
        },
        "TOR": {
            "Elo": 1490.0,
            "Causal_Elo": 1490.0,
            "Last_Game_Date": "2026-11-18",
            "games_played": 15,
            "goalsAgainst": 42.0,
            "xGoalsAgainst": 42.0,
            "highDangerxGoalsAgainst": 18.0,
        },
    }


def _sample_trained_state(today_str: str = "2026-11-19") -> dict:
    now_et = datetime.strptime(f"{today_str} 12:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET_ZONE)
    return {
        "context": {
            "today_str": today_str,
            "now_et": now_et,
            "nhl_season": "20262027",
            "nhl_season_label": "2026-27",
            "mp_year": 2026,
        },
        "session": None,
        "sync_stats": {"active_mp_year": 2026},
        "log_stats": {"total_rows": 30},
        "latest_team_context": {},
        "pairwise_bundle": {
            "model": _DummyGameClf(),
            "feature_names": ["Elo_diff", "Is_B2B_Home", "Is_B2B_Away"],
            "optimal_T": 1.0,
            "metrics": {"oof_accuracy": 0.6},
        },
        "skater_bundles": {
            "regular": {
                "model": _DummySkaterReg(),
                "imputer": _DummyImputer(),
                "features": ["I_F_shotsOnGoal_per_game"],
                "n_half": 16.0,
                "mode": "regular",
                "metrics": {"cv_rmse": 0.25},
            }
        },
        "profiles_regular": _sample_profiles(),
        "profiles_playoffs": None,
        "active_goalies": {},
        "df_pool": pd.DataFrame(
            [
                {
                    "playerId": 97,
                    "name": "Connor McDavid",
                    "name_clean": "connor mcdavid",
                    "team": "EDM",
                    "position": "C",
                    "games_played": 15,
                    "I_F_shotsOnGoal_per_game": 4.2,
                    "base_pred_regular": 0.55,
                    "base_pred": 0.55,
                }
            ]
        ),
        "baseline_label": "Synthetic Test",
    }


class TestOddsBudget(unittest.TestCase):
    def test_fetch_nhl_odds_lookup_uses_single_credit_bookmakers_and_parses_headers(self):
        payload = [
            {
                "away_team": "Edmonton Oilers",
                "home_team": "Toronto Maple Leafs",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Toronto Maple Leafs", "price": 2.05},
                                    {"name": "Edmonton Oilers", "price": 1.80},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Toronto Maple Leafs", "price": 2.10},
                                    {"name": "Edmonton Oilers", "price": 1.85},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "away_team": "Boston Bruins",
                "home_team": "Montreal Canadiens",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Montreal Canadiens", "price": 2.25},
                                    {"name": "Boston Bruins", "price": 1.68},
                                ],
                            }
                        ],
                    }
                ],
            },
        ]
        session = _FakeOddsSession(
            [
                _FakeOddsResponse(
                    payload,
                    headers={
                        "X-Requests-Last": "1",
                        "X-Requests-Used": "120",
                        "X-Requests-Remaining": "380",
                    },
                )
            ]
        )
        meta: dict = {}
        odds = _fetch_nhl_odds_lookup("test-key", session, meta_out=meta)
        self.assertEqual(len(session.urls), 1)
        self.assertIn(f"bookmakers={ODDS_API_BOOKMAKERS}", session.urls[0])
        self.assertNotIn("regions=", session.urls[0])

        # Prefers Pinnacle when available, falls back to FanDuel when Pinnacle is missing.
        self.assertEqual(odds[("EDM", "TOR")]["bookie"], "Pinnacle")
        self.assertEqual(odds[("EDM", "TOR")]["TOR"], 2.10)
        self.assertEqual(odds[("EDM", "TOR")]["bookie_key"], "pinnacle")
        self.assertEqual(odds[("EDM", "TOR")]["books"]["pinnacle"]["title"], "Pinnacle")
        self.assertEqual(odds[("EDM", "TOR")]["books"]["pinnacle"]["home"], 2.10)
        self.assertEqual(odds[("EDM", "TOR")]["books"]["pinnacle"]["away"], 1.85)
        self.assertEqual(odds[("EDM", "TOR")]["books"]["fanduel"]["home"], 2.05)
        self.assertEqual(odds[("EDM", "TOR")]["books"]["fanduel"]["away"], 1.80)
        self.assertEqual(odds[("BOS", "MTL")]["bookie"], "FanDuel")
        self.assertEqual(odds[("BOS", "MTL")]["BOS"], 1.68)
        self.assertEqual(list(odds[("BOS", "MTL")]["books"]), ["fanduel"])
        self.assertEqual(odds[("BOS", "MTL")]["bookie_key"], "fanduel")

        self.assertTrue(meta["api_ok"])
        self.assertEqual(meta["requests_last"], 1)
        self.assertEqual(meta["requests_used"], 120)
        self.assertEqual(meta["requests_remaining"], 380)

    def test_extract_cached_odds_and_predict_reuse(self):
        # Supports both new explicit away_odds/home_odds (even with None edges) and legacy edge fallback
        legacy_and_new = {
            "games": [
                {
                    "date": "2026-11-19",
                    "away": "EDM",
                    "home": "TOR",
                    "bookie": "Pinnacle",
                    "away_odds": 1.75,
                    "home_odds": 2.15,
                    "book_odds": {"pinnacle": {"title": "Pinnacle", "home": 2.15, "away": 1.75}},
                    "bookie_key": "pinnacle",
                    "away_edge": None,
                    "home_edge": None,
                },
                {
                    "date": "2026-11-19",
                    "away": "BOS",
                    "home": "MTL",
                    "bookie": "FanDuel",
                    "away_edge": None,
                    "home_edge": {"odds": 2.40, "edge_pct": 5.0},
                },
            ]
        }
        cached = extract_cached_odds_lookup(legacy_and_new)
        self.assertEqual(cached[("2026-11-19", "EDM", "TOR")]["TOR"], 2.15)
        self.assertEqual(cached[("2026-11-19", "EDM", "TOR")]["EDM"], 1.75)
        self.assertEqual(cached[("2026-11-19", "EDM", "TOR")]["books"]["pinnacle"]["home"], 2.15)
        self.assertEqual(cached[("2026-11-19", "EDM", "TOR")]["bookie_key"], "pinnacle")
        self.assertEqual(cached[("2026-11-19", "BOS", "MTL")]["MTL"], 2.40)

        bundle = {"model": _DummyGameClf(), "feature_names": ["Elo_diff", "Is_B2B_Home", "Is_B2B_Away"], "optimal_T": 1.0}
        sched = [
            {
                "id": 1,
                "date": "2026-11-19",
                "is_today": True,
                "away": "EDM",
                "home": "TOR",
                "game_type": 2,
                "game_state": "FUT",
                "start_time_et": "10:00 PM ET",
            }
        ]
        preds, _ = predict_upcoming_games(
            sched,
            _sample_profiles(),
            bundle,
            odds_api_key="",
            cached_odds=cached,
        )
        self.assertEqual(preds[0]["bookie"], "Pinnacle")
        self.assertEqual(preds[0]["away_odds"], 1.75)
        self.assertEqual(preds[0]["home_odds"], 2.15)
        # DummyGameClf outputs home_prob=0.65, 0.65 * 2.15 - 1 = +39.75% edge
        self.assertIsNotNone(preds[0]["home_edge"])
        self.assertAlmostEqual(preds[0]["home_edge"]["edge_pct"], 39.75, places=2)

    def test_live_loop_cap_all_started_skip_and_date_reset(self):
        api_game_payload = [
            {
                "away_team": "Edmonton Oilers",
                "home_team": "Toronto Maple Leafs",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Toronto Maple Leafs", "price": 2.20},
                                    {"name": "Edmonton Oilers", "price": 1.72},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        schedule_state = {
            "date": "2026-11-19",
            "game_state": "FUT",
        }

        def _fake_schedule(ctx, session=None):
            return {
                "games": [
                    {
                        "id": 1001,
                        "date": schedule_state["date"],
                        "is_today": True,
                        "away": "EDM",
                        "home": "TOR",
                        "game_type": 2,
                        "game_state": schedule_state["game_state"],
                        "start_time_et": "10:00 PM ET",
                        "start_time_utc": f"{schedule_state['date']}T03:00:00Z",
                    }
                ],
                "today_games": [],
                "daily_matchups": {"TOR": "EDM", "EDM": "TOR"},
                "active_game_type": 2,
                "active_mode": "Regular Season",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            models_dir = os.path.join(tmpdir, "models")
            docs_dir = os.path.join(tmpdir, "docs")
            os.makedirs(docs_dir, exist_ok=True)
            fake_session = _FakeOddsSession(
                [
                    _FakeOddsResponse(
                        api_game_payload,
                        headers={"X-Requests-Last": "1", "X-Requests-Used": str(10 + i), "X-Requests-Remaining": str(490 - i)},
                    )
                    for i in range(6)
                ]
            )
            state = _sample_trained_state("2026-11-19")
            state["session"] = fake_session

            with (
                patch.object(main_pipeline, "fetch_nhl_schedule", side_effect=_fake_schedule),
                patch.object(main_pipeline, "fetch_tims_picks", return_value=(pd.DataFrame(), "ok")),
                patch.object(main_pipeline, "fetch_nhl_injuries", return_value=([], "ok")),
                patch.object(main_pipeline, "fetch_starting_goalies", return_value=({}, "ok")),
                patch.object(main_pipeline, "fetch_live_club_skaters", return_value={}),
                patch.object(main_pipeline, "settle_pending_tims_days", return_value=0),
            ):
                # Waves 1, 2, 3 in live_loop consume 1 call each (up to cap=3)
                for expected_count in (1, 2, 3):
                    p = main_pipeline.run_inference(
                        data_dir=data_dir,
                        models_dir=models_dir,
                        docs_dir=docs_dir,
                        odds_api_key="key",
                        odds_policy="live_loop",
                        live_odds_cap=3,
                        trained_state=state,
                    )
                    usage = p["meta"]["odds_usage"]
                    self.assertEqual(usage["live_loop_fetches"], expected_count)
                    self.assertEqual(usage["last_source"], "api")
                self.assertEqual(len(fake_session.urls), 3)

                # Wave 4 hits cap -> skips HTTP call and reuses cached Pinnacle odds
                p4 = main_pipeline.run_inference(
                    data_dir=data_dir,
                    models_dir=models_dir,
                    docs_dir=docs_dir,
                    odds_api_key="key",
                    odds_policy="live_loop",
                    live_odds_cap=3,
                    trained_state=state,
                )
                self.assertEqual(len(fake_session.urls), 3)
                self.assertEqual(p4["meta"]["odds_usage"]["last_source"], "cached_cap_reached")
                self.assertEqual(p4["meta"]["odds_usage"]["live_loop_fetches"], 3)
                self.assertTrue(p4["meta"]["odds_enabled"])
                self.assertEqual(p4["games"][0]["bookie"], "Pinnacle")
                self.assertEqual(p4["games"][0]["home_odds"], 2.20)
                self.assertEqual(p4["games"][0]["away_odds"], 1.72)

                # Scheduled always run (e.g. infer_2200) still fetches fresh odds WITHOUT bumping live_loop_fetches
                p_always = main_pipeline.run_inference(
                    data_dir=data_dir,
                    models_dir=models_dir,
                    docs_dir=docs_dir,
                    odds_api_key="key",
                    odds_policy="always",
                    live_odds_cap=3,
                    trained_state=state,
                )
                self.assertEqual(len(fake_session.urls), 4)
                self.assertEqual(p_always["meta"]["odds_usage"]["last_source"], "api")
                self.assertEqual(p_always["meta"]["odds_usage"]["live_loop_fetches"], 3)

                # New day resets counter to 0; when all games are LIVE, live_loop skips API
                state_day2 = _sample_trained_state("2026-11-20")
                state_day2["session"] = fake_session
                schedule_state["date"] = "2026-11-20"
                schedule_state["game_state"] = "LIVE"
                p_started = main_pipeline.run_inference(
                    data_dir=data_dir,
                    models_dir=models_dir,
                    docs_dir=docs_dir,
                    odds_api_key="key",
                    odds_policy="live_loop",
                    live_odds_cap=3,
                    trained_state=state_day2,
                )
                self.assertEqual(len(fake_session.urls), 4)
                self.assertEqual(p_started["meta"]["odds_usage"]["live_loop_fetches"], 0)
                self.assertEqual(p_started["meta"]["odds_usage"]["last_source"], "cached_all_started")

                # Once an unstarted game exists on Day 2, wave 1 succeeds with counter=1
                schedule_state["game_state"] = "FUT"
                p_day2_w1 = main_pipeline.run_inference(
                    data_dir=data_dir,
                    models_dir=models_dir,
                    docs_dir=docs_dir,
                    odds_api_key="key",
                    odds_policy="live_loop",
                    live_odds_cap=3,
                    trained_state=state_day2,
                )
                self.assertEqual(len(fake_session.urls), 5)
                self.assertEqual(p_day2_w1["meta"]["odds_usage"]["live_loop_fetches"], 1)
                self.assertEqual(p_day2_w1["meta"]["odds_usage"]["last_source"], "api")

    def test_live_loop_default_inference_passes_cap_flags(self):
        with patch.object(live_loop.subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            rc = live_loop._default_inference(live_odds_cap=3)
            self.assertEqual(rc, 0)
            cmd = mock_run.call_args[0][0]
            self.assertIn("--odds-policy", cmd)
            self.assertIn("live_loop", cmd)
            self.assertIn("--live-odds-cap", cmd)
            self.assertIn("3", cmd)


if __name__ == "__main__":
    unittest.main()
