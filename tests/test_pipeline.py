"""Unit & integration tests for Hockeypicks AI pipeline modules."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_sync import (
    _compute_team_streaks,
    _extract_post_game_streak_state,
    clean_name,
    fetch_nhl_schedule,
    fetch_tims_picks,
    get_nhl_season_context,
    normalize_team_abbrev,
)
from src.report_builder import write_json_report, write_terminal_log_html
from src.skater_model import (
    HAS_SKATER_ML,
    combine_skater_frames,
    compute_empirical_bayes_n_half,
    prep_dynamic_features,
    score_skater_candidates,
)
from src.team_game_model import (
    HAS_GAME_ML,
    _merge_side_metrics,
    apply_temperature,
    compute_power_rankings_and_defense,
    evaluate_kelly_edge,
    expected_remaining_series_games,
    single_game_to_best_of_7_prob,
)
from src.tims_tracker import (
    recompute_tims_summary,
    settle_pending_tims_days,
    upsert_tims_snapshot,
)


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _DummySession:
    def __init__(self, routes: dict[str, tuple[int, dict]]):
        self._routes = routes

    def get(self, url: str, headers=None, timeout: int = 10):
        status, payload = self._routes.get(url, (404, {}))
        return _DummyResponse(status, payload)


class TestDataSyncAndContext(unittest.TestCase):
    def test_season_context_november_regular_season(self):
        ctx = get_nhl_season_context(datetime(2026, 11, 15, 18, 0))
        self.assertEqual(ctx["today_str"], "2026-11-15")
        self.assertEqual(ctx["tomorrow_str"], "2026-11-16")
        self.assertEqual(ctx["nhl_season"], "20262027")
        self.assertEqual(ctx["mp_year"], 2026)
        self.assertEqual(ctx["historical_mp_years"], [2021, 2022, 2023, 2024, 2025, 2026])

    def test_season_context_may_playoffs(self):
        ctx = get_nhl_season_context(datetime(2026, 5, 6, 12, 0))
        self.assertEqual(ctx["nhl_season"], "20252026")
        self.assertEqual(ctx["mp_year"], 2025)

    def test_normalize_team_and_player_name(self):
        self.assertEqual(normalize_team_abbrev("S.J"), "SJS")
        self.assertEqual(normalize_team_abbrev("l.a"), "LAK")
        self.assertEqual(clean_name("Tim Stützle"), "tim stutzle")
        self.assertEqual(clean_name("Alexis Lafrenière"), "alexis lafreniere")

    def test_compute_streaks_leak_free(self):
        df = pd.DataFrame(
            [
                {"Date": "2026-10-10", "Team": "EDM", "Won": 1},
                {"Date": "2026-10-12", "Team": "EDM", "Won": 1},
                {"Date": "2026-10-14", "Team": "EDM", "Won": 0},
            ]
        )
        out = _compute_team_streaks(df)
        self.assertEqual(out["Last_Game_Won"].tolist(), [0, 1, 1])
        self.assertEqual(out["Current_Streak"].tolist(), [0, 1, 2])
        post_ctx = _extract_post_game_streak_state(out)
        self.assertEqual(post_ctx["EDM"], {"Last_Game_Won": 0, "Current_Streak": -1})

    def test_fetch_nhl_schedule_handles_gameweek_list(self):
        ctx = get_nhl_season_context(datetime(2026, 11, 15, 10, 0))
        session = _DummySession(
            {
                "https://api-web.nhle.com/v1/schedule/2026-11-15": (
                    200,
                    {
                        "gameWeek": [
                            {
                                "date": "2026-11-15",
                                "games": [
                                    {
                                        "id": 1001,
                                        "gameType": 2,
                                        "gameState": "FUT",
                                        "startTimeUTC": "2026-11-16T00:00:00Z",
                                        "awayTeam": {"abbrev": "EDM"},
                                        "homeTeam": {"abbrev": "TOR"},
                                    }
                                ],
                            }
                        ]
                    },
                )
            }
        )
        sched = fetch_nhl_schedule(ctx, session=session)
        self.assertEqual(sched["active_mode"], "Regular Season")
        self.assertEqual(sched["active_game_type"], 2)
        self.assertEqual(sched["daily_matchups"], {"EDM": "TOR", "TOR": "EDM"})
        self.assertEqual(len(sched["games"]), 1)

    def test_fetch_nhl_schedule_rolls_slate_to_tomorrow_when_today_finished(self):
        ctx = get_nhl_season_context(datetime(2026, 11, 15, 23, 43))
        session = _DummySession(
            {
                "https://api-web.nhle.com/v1/schedule/2026-11-15": (
                    200,
                    {
                        "gameWeek": [
                            {
                                "date": "2026-11-15",
                                "games": [
                                    {
                                        "id": 1001,
                                        "gameType": 2,
                                        "gameState": "OFF",
                                        "startTimeUTC": "2026-11-16T00:00:00Z",
                                        "awayTeam": {"abbrev": "EDM", "score": 4},
                                        "homeTeam": {"abbrev": "TOR", "score": 2},
                                    }
                                ],
                            }
                        ]
                    },
                ),
                "https://api-web.nhle.com/v1/schedule/2026-11-16": (
                    200,
                    {
                        "gameWeek": [
                            {
                                "date": "2026-11-16",
                                "games": [
                                    {
                                        "id": 1002,
                                        "gameType": 2,
                                        "gameState": "FUT",
                                        "startTimeUTC": "2026-11-17T00:00:00Z",
                                        "awayTeam": {"abbrev": "COL"},
                                        "homeTeam": {"abbrev": "DAL"},
                                    }
                                ],
                            }
                        ]
                    },
                ),
            }
        )
        sched = fetch_nhl_schedule(ctx, session=session)
        self.assertEqual(sched["daily_matchups"], {"COL": "DAL", "DAL": "COL"})
        self.assertEqual(len(sched["games"]), 2)

    def test_fetch_tims_picks_parses_groups(self):
        session = _DummySession(
            {
                "https://api.hockeychallengehelper.com/api/picks": (
                    200,
                    {
                        "playerLists": [
                            {"players": [{"firstName": "Connor", "lastName": "McDavid", "team": "EDM", "opponentTeam": "TOR", "position": "C"}]},
                            {"players": [{"firstName": "Zach", "lastName": "Hyman", "team": "EDM", "opponentTeam": "TOR", "position": "W"}]},
                        ]
                    },
                )
            }
        )
        df, status = fetch_tims_picks(session=session)
        self.assertEqual(status, "ok")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["group"], 1)
        self.assertEqual(df.iloc[1]["group"], 2)


class TestSkaterAndTeamMath(unittest.TestCase):
    def test_combine_skater_frames_weights_rates_instead_of_summing(self):
        df_play = pd.DataFrame(
            [
                {
                    "playerId": 1,
                    "season": 2025,
                    "name": "Test Skater",
                    "team": "EDM",
                    "position": "C",
                    "situation": "all",
                    "games_played": 10,
                    "I_F_goals": 5,
                    "onIce_xGoalsPercentage": 0.60,
                }
            ]
        )
        df_reg = pd.DataFrame(
            [
                {
                    "playerId": 1,
                    "season": 2026,
                    "name": "Test Skater",
                    "team": "EDM",
                    "position": "C",
                    "situation": "all",
                    "games_played": 30,
                    "I_F_goals": 15,
                    "onIce_xGoalsPercentage": 0.50,
                }
            ]
        )
        combined = combine_skater_frames([df_play, df_reg])
        self.assertEqual(int(combined.iloc[0]["games_played"]), 40)
        self.assertEqual(int(combined.iloc[0]["I_F_goals"]), 20)
        self.assertEqual(int(combined.iloc[0]["season"]), 2026)
        # Weighted average: (10*0.60 + 30*0.50) / 40 = 0.525 (NOT summed 1.10!)
        self.assertAlmostEqual(float(combined.iloc[0]["onIce_xGoalsPercentage"]), 0.525, places=6)

    def test_prep_dynamic_features_and_eb(self):
        df = pd.DataFrame(
            [
                {
                    "playerId": 1,
                    "season": 2025,
                    "name": "Player A",
                    "team": "TOR",
                    "position": "C",
                    "situation": "all",
                    "games_played": 20,
                    "I_F_goals": 10,
                    "I_F_shotsOnGoal": 80,
                    "onIce_xGoalsPercentage": 0.55,
                }
            ]
        )
        prepped, feats = prep_dynamic_features(df)
        self.assertIn("I_F_shotsOnGoal_per_game", feats)
        self.assertIn("onIce_xGoalsPercentage", feats)
        self.assertNotIn("I_F_goals_per_game", feats)
        self.assertAlmostEqual(float(prepped.iloc[0]["I_F_shotsOnGoal_per_game"]), 4.0)

        n_half = compute_empirical_bayes_n_half(np.linspace(0.05, 0.55, 60))
        self.assertGreaterEqual(n_half, 2.0)
        self.assertLessEqual(n_half, 35.0)
        self.assertEqual(compute_empirical_bayes_n_half(np.full(30, 0.16), residual_mse=0.072**2), 25.0)
        self.assertEqual(compute_empirical_bayes_n_half(np.full(30, 0.16), residual_mse=0.16**2), 9.0)

    def test_prep_dynamic_features_no_target_duplicate_or_leak(self):
        df = pd.DataFrame(
            [
                {
                    "playerId": 1,
                    "season": 2025,
                    "name": "Player A",
                    "team": "EDM",
                    "position": "C",
                    "situation": "all",
                    "games_played": 20,
                    "I_F_goals": 10,
                    "I_F_shotsOnGoal": 80,
                    "target_gp": 40,
                    "target_goals": 16,
                    "target_goals_per_game": 0.40,
                }
            ]
        )
        prepped, feats = prep_dynamic_features(df)
        self.assertEqual(list(prepped.columns).count("target_goals_per_game"), 1)
        self.assertEqual(prepped["target_goals_per_game"].ndim, 1)
        self.assertAlmostEqual(float(prepped["target_goals_per_game"].iloc[0]), 0.40)
        for forbidden in ("target_gp", "target_goals", "target_gp_per_game", "target_goals_per_game"):
            self.assertNotIn(forbidden, feats)
        self.assertIn("I_F_shotsOnGoal_per_game", feats)

    def test_temperature_markov_series_and_kelly(self):
        p_cal = apply_temperature(0.90, T=3.0)
        self.assertGreater(p_cal, 0.5)
        self.assertLess(p_cal, 0.90)
        self.assertAlmostEqual(apply_temperature(0.5, T=5.0), 0.5, places=6)

        self.assertAlmostEqual(single_game_to_best_of_7_prob(0.5, 0, 0), 0.5, places=6)
        self.assertGreater(single_game_to_best_of_7_prob(0.6, 2, 1), 0.70)
        self.assertAlmostEqual(expected_remaining_series_games(0.5, 3, 3), 1.0, places=6)

        edge = evaluate_kelly_edge(model_prob=0.58, decimal_odds=2.0, bankroll=1000.0)
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge["edge_pct"], 16.0, places=2)
        self.assertGreater(edge["rec_bet"], 0.0)
        self.assertIsNone(evaluate_kelly_edge(model_prob=0.45, decimal_odds=1.90))

    def test_merge_side_metrics_avoids_column_collisions(self):
        games = pd.DataFrame([{"mp_lookup_season": 2025, "Team_A": "EDM", "Opponent": "TOR"}])
        stars = pd.DataFrame(
            [
                {"season": 2025, "team": "EDM", "Top3_Points": 250, "Top3_xGoals": 90.0},
                {"season": 2025, "team": "TOR", "Top3_Points": 220, "Top3_xGoals": 82.0},
            ]
        )
        m1 = _merge_side_metrics(games, stars, "Team_A", {"Top3_Points": "Top3_Points_A", "Top3_xGoals": "Top3_xGoals_A"})
        m2 = _merge_side_metrics(m1, stars, "Opponent", {"Top3_Points": "Top3_Points_B", "Top3_xGoals": "Top3_xGoals_B"})
        self.assertEqual(int(m2.iloc[0]["Top3_Points_A"]), 250)
        self.assertEqual(int(m2.iloc[0]["Top3_Points_B"]), 220)
        self.assertNotIn("season", m2.columns)
        self.assertNotIn("team", m2.columns)

    def test_power_rankings_and_scoring_without_xgb(self):
        class _FakeClf:
            def predict_proba(self, X):
                scores = 0.5 + np.clip(X.iloc[:, 0].to_numpy() * 0.05, -0.35, 0.35)
                return np.column_stack([1.0 - scores, scores])

        class _FakeReg:
            def predict(self, X):
                return np.full(len(X), 0.40)

        class _FakeImputer:
            def transform(self, X):
                return X.to_numpy(dtype=float)

        profiles = {
            "EDM": {
                "xGoalsFor": 100.0,
                "Top3_Points": 200.0,
                "Team_GSAx": 12.5,
                "Starter_SavePct": 0.918,
                "Streak": 3,
                "Last_Game_Won": 1,
                "games_played": 20,
                "goalsAgainst": 45.0,
                "xGoalsAgainst": 48.0,
                "highDangerxGoalsAgainst": 18.0,
            },
            "SJS": {
                "xGoalsFor": 60.0,
                "Top3_Points": 110.0,
                "Team_GSAx": -6.0,
                "Starter_SavePct": 0.889,
                "Streak": -2,
                "Last_Game_Won": 0,
                "games_played": 20,
                "goalsAgainst": 72.0,
                "xGoalsAgainst": 68.0,
                "highDangerxGoalsAgainst": 26.0,
            },
        }
        game_bundle = {
            "model": _FakeClf(),
            "feature_names": ["xGoalsFor_diff"],
            "optimal_T": 2.0,
        }
        rankings, def_mults = compute_power_rankings_and_defense(profiles, game_bundle)
        self.assertEqual(rankings[0]["team"], "EDM")
        self.assertGreater(def_mults["SJS"], def_mults["EDM"])
        self.assertGreaterEqual(min(def_mults.values()), 0.85)
        self.assertLessEqual(max(def_mults.values()), 1.15)
        self.assertAlmostEqual(float(np.mean(list(def_mults.values()))), 1.0, places=3)

        skater_bundle = {
            "model": _FakeReg(),
            "imputer": _FakeImputer(),
            "features": ["I_F_shotsOnGoal_per_game"],
            "n_half": 10.0,
        }
        baseline = pd.DataFrame(
            [
                {
                    "playerId": 97,
                    "name": "Connor McDavid",
                    "name_clean": "connor mcdavid",
                    "team": "EDM",
                    "position": "C",
                    "games_played": 20,
                    "I_F_shotsOnGoal_per_game": 4.2,
                }
            ]
        )
        tims_df = pd.DataFrame(
            [
                {
                    "group": 1,
                    "player_name": "Connor McDavid",
                    "name_clean": "connor mcdavid",
                    "team": "EDM",
                    "opponentTeam": "SJS",
                    "position": "C",
                }
            ]
        )
        scores = score_skater_candidates(
            bundle=skater_bundle,
            df_baseline=baseline,
            df_tims=tims_df,
            daily_matchups={"EDM": "SJS", "SJS": "EDM"},
            live_skaters={"connor mcdavid": {"live_gp": 10, "live_goals": 8}},
            defense_multipliers=def_mults,
            active_mode="Regular Season",
        )
        top_pick = scores["tims_groups"]["1"][0]
        # Base=0.40, Live=0.80, weight=10/20=0.5 -> blended=0.60, times SJS weak-D mult > 1.0
        self.assertGreater(top_pick["final_score"], 0.60)
        self.assertEqual(len(scores["slate_top_scorers"]), 1)

    def test_report_builders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "data.json")
            html_path = os.path.join(tmpdir, "latest_run.html")
            write_json_report({"meta": {"status": "ok"}}, json_path)
            write_terminal_log_html("line1\n<script>alert(1)</script>", {"active_mode": "Regular Season"}, html_path)

            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["meta"]["status"], "ok")
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("<script>alert(1)</script>", content)
            self.assertIn("&lt;script&gt;", content)

    def test_tims_history_snapshot_and_settlement(self):
        history = {"days": [], "summary": {}}
        tims_groups = {
            "1": [
                {"player": "Connor McDavid", "team": "EDM", "opponent": "TOR", "final_score": 0.48, "base_pred": 0.45, "live_gp": 10, "live_goals": 6, "def_multiplier": 1.03},
                {"player": "Auston Matthews", "team": "TOR", "opponent": "EDM", "final_score": 0.44, "base_pred": 0.43, "live_gp": 10, "live_goals": 4, "def_multiplier": 1.01},
            ],
            "2": [
                {"player": "Zach Hyman", "team": "EDM", "opponent": "TOR", "final_score": 0.35, "base_pred": 0.33, "live_gp": 10, "live_goals": 4, "def_multiplier": 1.03},
            ],
            "3": [
                {"player": "Matthew Knies", "team": "TOR", "opponent": "EDM", "final_score": 0.24, "base_pred": 0.22, "live_gp": 10, "live_goals": 3, "def_multiplier": 1.01},
            ],
        }
        saved = upsert_tims_snapshot(
            history,
            date_str="2026-11-15",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=tims_groups,
            captured_et="2026-11-15 10:17 AM ET",
        )
        self.assertTrue(saved)
        # Empty off-hours call must not erase existing day snapshot
        self.assertFalse(
            upsert_tims_snapshot(
                history,
                date_str="2026-11-15",
                context={"nhl_season": "20262027"},
                schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
                skater_bundle={"n_half": 25.0},
                tims_groups={},
                captured_et="2026-11-15 11:43 PM ET",
            )
        )

        session = _DummySession(
            {
                "https://api-web.nhle.com/v1/schedule/2026-11-15": (
                    200,
                    {
                        "gameWeek": [
                            {
                                "date": "2026-11-15",
                                "games": [
                                    {
                                        "id": 9001,
                                        "gameState": "OFF",
                                        "awayTeam": {"abbrev": "EDM"},
                                        "homeTeam": {"abbrev": "TOR"},
                                    }
                                ],
                            }
                        ]
                    },
                ),
                "https://api-web.nhle.com/v1/gamecenter/9001/boxscore": (
                    200,
                    {
                        "playerByGameStats": {
                            "awayTeam": {
                                "forwards": [
                                    {"name": {"default": "Connor McDavid"}, "goals": 1},
                                    {"name": {"default": "Zach Hyman"}, "goals": 0},
                                ],
                                "defense": [],
                            },
                            "homeTeam": {
                                "forwards": [
                                    {"name": {"default": "Auston Matthews"}, "goals": 0},
                                    {"name": {"default": "Matthew Knies"}, "goals": 2},
                                ],
                                "defense": [],
                            },
                        }
                    },
                ),
            }
        )
        settled_cnt = settle_pending_tims_days(
            history,
            today_str="2026-11-16",
            settled_et="2026-11-16 10:17 AM ET",
            session=session,
            allow_network=True,
        )
        self.assertEqual(settled_cnt, 1)
        summary = recompute_tims_summary(history)
        self.assertEqual(summary["settled_days"], 1)
        self.assertEqual(summary["ai_hits"], 2)
        self.assertEqual(summary["ai_possible"], 3)
        self.assertAlmostEqual(summary["ai_hit_rate_pct"], 66.7, places=1)
        self.assertAlmostEqual(summary["random_expected_pct"], 50.0, places=1)

    def test_main_tee_stdout_no_recursion(self):
        import io
        import sys
        from unittest.mock import patch
        import main_pipeline

        fake_stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_payload = {
                "meta": {"updated_et": "test", "nhl_season_label": "2026-27", "active_mode": "Regular Season"}
            }

            def _fake_run(**kwargs):
                print("hello from fake run_pipeline")
                return fake_payload

            with (
                patch.object(sys, "stdout", fake_stdout),
                patch.object(sys, "argv", ["main_pipeline.py", "--docs-dir", tmpdir, "--skip-network"]),
                patch.object(main_pipeline, "run_pipeline", side_effect=_fake_run),
            ):
                rc = main_pipeline.main()

            self.assertEqual(rc, 0)
            self.assertIn("hello from fake run_pipeline", fake_stdout.getvalue())
            log_html = os.path.join(tmpdir, "latest_run.html")
            self.assertTrue(os.path.exists(log_html))
            with open(log_html, "r", encoding="utf-8") as f:
                self.assertIn("hello from fake run_pipeline", f.read())

    @unittest.skipUnless(HAS_SKATER_ML and HAS_GAME_ML, "xgboost/sklearn not installed in current Python runtime")
    def test_end_to_end_xgboost_training_smoke(self):
        from main_pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            mp_dir = os.path.join(data_dir, "moneypuck")
            log_dir = os.path.join(data_dir, "nhl_game_logs")
            os.makedirs(mp_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # Build tiny multi-year synthetic dataset
            rng = np.random.default_rng(42)
            for y in (2024, 2025, 2026):
                skaters = pd.DataFrame(
                    {
                        "playerId": np.arange(1, 31),
                        "season": y,
                        "name": [f"Skater {i}" for i in range(1, 31)],
                        "team": ["EDM" if i <= 15 else "TOR" for i in range(1, 31)],
                        "position": "C",
                        "situation": "all",
                        "games_played": 30,
                        "I_F_goals": rng.integers(3, 25, size=30),
                        "I_F_points": rng.integers(10, 60, size=30),
                        "I_F_xGoals": rng.uniform(5.0, 25.0, size=30),
                        "I_F_shotsOnGoal": rng.integers(30, 140, size=30),
                        "onIce_xGoalsPercentage": rng.uniform(0.42, 0.58, size=30),
                    }
                )
                skaters.to_csv(os.path.join(mp_dir, f"{y}_regular_skaters.csv"), index=False)
                teams = pd.DataFrame(
                    {
                        "team": ["EDM", "TOR"],
                        "season": y,
                        "situation": "all",
                        "games_played": [30, 30],
                        "goalsAgainst": [80, 92],
                        "xGoalsAgainst": [78.0, 90.0],
                        "highDangerxGoalsAgainst": [25.0, 32.0],
                        "xGoalsFor": [102.0, 88.0],
                    }
                )
                teams.to_csv(os.path.join(mp_dir, f"{y}_regular_teams.csv"), index=False)
                goalies = pd.DataFrame(
                    {
                        "team": ["EDM", "TOR"],
                        "situation": "all",
                        "games_played": [22, 22],
                        "goalsSavedAboveExpected": [8.2, -1.5],
                        "savePercentage": [0.916, 0.898],
                    }
                )
                goalies.to_csv(os.path.join(mp_dir, f"{y}_regular_goalies.csv"), index=False)

            logs = []
            for idx in range(60):
                d = f"2025-{(idx // 28) + 10:02d}-{(idx % 28) + 1:02d}"
                w = 1 if idx % 2 == 0 else 0
                logs.append({"Season": "20252026", "Team": "EDM", "Date": d, "GameType": 2, "Opponent": "TOR", "Is_Home": 1, "Team_Score": 4, "Opp_Score": 2, "Won": w, "Last_Game_Won": 1, "Current_Streak": 1})
                logs.append({"Season": "20252026", "Team": "TOR", "Date": d, "GameType": 2, "Opponent": "EDM", "Is_Home": 0, "Team_Score": 2, "Opp_Score": 4, "Won": 1 - w, "Last_Game_Won": 0, "Current_Streak": -1})
            df_l = pd.DataFrame(logs)
            df_l[df_l["Team"] == "EDM"].to_csv(os.path.join(log_dir, "EDM_20252026_gamelog.csv"), index=False)
            df_l[df_l["Team"] == "TOR"].to_csv(os.path.join(log_dir, "TOR_20252026_gamelog.csv"), index=False)

            payload = run_pipeline(
                data_dir=data_dir,
                models_dir=os.path.join(tmpdir, "models"),
                docs_dir=os.path.join(tmpdir, "docs"),
                skip_network=True,
            )
            self.assertEqual(payload["meta"]["status"], "live")
            self.assertEqual(len(payload["power_rankings"]), 2)


if __name__ == "__main__":
    unittest.main()
