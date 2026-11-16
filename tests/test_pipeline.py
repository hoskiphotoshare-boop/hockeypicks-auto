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
    PlayerMatcher,
    _compute_team_streaks,
    _extract_post_game_streak_state,
    clean_name,
    fetch_nhl_schedule,
    fetch_tims_picks,
    get_nhl_season_context,
    normalize_team_abbrev,
)
from src.infer_gate import evaluate_gate, find_due_games, parse_game_start_utc
from src.report_builder import write_json_report, write_terminal_log_html
from src.skater_model import (
    HAS_SKATER_ML,
    blend_skater_goal_rate,
    combine_skater_frames,
    compute_empirical_bayes_n_half,
    prep_dynamic_features,
    score_skater_candidates,
)
from src.tims_timing import compute_tims_deferral_advice
from src.team_game_model import (
    HAS_GAME_ML,
    _merge_side_metrics,
    _pairwise_win_prob,
    add_causal_game_log_features,
    apply_temperature,
    compute_power_rankings_and_defense,
    evaluate_kelly_edge,
    expected_remaining_series_games,
    extract_latest_team_form,
    normalize_team_frame_per_game,
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

    def test_player_matcher_accents_nicknames_and_boxscore_initials(self):
        matcher = PlayerMatcher(
            [
                {"player_id": 8482109, "name": "Alexis Lafrenire", "team": "NYR", "position": "L", "payload": {"tag": "laf"}},
                {"player_id": 8483515, "name": "Juraj Slafkovsk", "team": "MTL", "position": "L", "payload": {"tag": "slaf"}},
                {"player_id": 8480018, "name": "Nick Suzuki", "team": "MTL", "position": "C", "payload": {"tag": "suzuki"}},
                {"player_id": 8482175, "name": "JJ Peterka", "team": "BOS", "position": "L", "payload": {"tag": "jj"}},
                {"player_id": 8480012, "name": "Elias Pettersson", "team": "VAN", "position": "C", "payload": {"tag": "petey_f"}},
                {"player_id": 8483678, "name": "Elias Pettersson", "team": "VAN", "position": "D", "payload": {"tag": "petey_d"}},
            ]
        )
        self.assertEqual(matcher.match("Alexis Lafrenière", team="NYR")["tag"], "laf")
        self.assertEqual(matcher.match("Juraj Slafkovsky", team="MTL")["tag"], "slaf")
        self.assertEqual(matcher.match("Nicholas Suzuki", team="MTL")["tag"], "suzuki")
        self.assertEqual(matcher.match("J.J. Peterka", team="BOS")["tag"], "jj")
        self.assertEqual(matcher.match("Elias Pettersson", team="VAN", position="F")["tag"], "petey_f")
        self.assertEqual(matcher.match("Elias Pettersson", team="VAN", position="D")["tag"], "petey_d")

        # NHL boxscore abbreviated names + team disambiguation
        box_matcher = PlayerMatcher(
            [
                {"name": "C. McDavid", "team": "EDM", "position": "F", "payload": {"goals": 2}},
                {"name": "N. Suzuki", "team": "MTL", "position": "F", "payload": {"goals": 1}},
                {"name": "A. Lafreniere", "team": "NYR", "position": "F", "payload": {"goals": 1}},
                {"name": "S. Aho", "team": "CAR", "position": "F", "payload": {"goals": 1}},
                {"name": "S. Aho", "team": "NYI", "position": "D", "payload": {"goals": 0}},
                {"name": "C. Brown", "team": "NJD", "position": "F", "payload": {"goals": 1}},
            ]
        )
        self.assertEqual(box_matcher.match("Connor McDavid", team="EDM", allow_initial_fallback=True)["goals"], 2)
        self.assertEqual(box_matcher.match("Nicholas Suzuki", team="MTL", allow_initial_fallback=True)["goals"], 1)
        self.assertEqual(box_matcher.match("Alexis Lafrenière", team="NYR", allow_initial_fallback=True)["goals"], 1)
        self.assertEqual(box_matcher.match("Sebastian Aho", team="CAR", allow_initial_fallback=True)["goals"], 1)
        self.assertEqual(box_matcher.match("Sebastian Aho", team="NYI", allow_initial_fallback=True)["goals"], 0)
        # Full name query against full name entry without initial does not match different first name
        full_only = PlayerMatcher([{"name": "Chris Brown", "team": "NJD", "payload": {"goals": 1}}])
        self.assertIsNone(full_only.match("Connor Brown", team="NJD", allow_initial_fallback=True))

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

        # Also verify real NHL abbreviated boxscore names ("C. McDavid", "N. Suzuki", "A. Lafreniere")
        abbr_history = {"days": [], "summary": {}}
        upsert_tims_snapshot(
            abbr_history,
            date_str="2026-11-16",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 30.0},
            tims_groups={
                "1": [
                    {"player": "Nicholas Suzuki", "team": "MTL", "opponent": "NYR", "final_score": 0.44, "base_pred": 0.39, "live_gp": 14, "live_goals": 8, "def_multiplier": 1.02},
                ],
                "2": [
                    {"player": "Alexis Lafrenière", "team": "NYR", "opponent": "MTL", "final_score": 0.38, "base_pred": 0.36, "live_gp": 20, "live_goals": 9, "def_multiplier": 0.99},
                ],
            },
            captured_et="2026-11-16 12:03 AM ET",
        )
        abbr_sess = _DummySession(
            {
                "https://api-web.nhle.com/v1/schedule/2026-11-16": (
                    200,
                    {
                        "gameWeek": [
                            {
                                "date": "2026-11-16",
                                "games": [
                                    {
                                        "id": 9002,
                                        "gameState": "FINAL",
                                        "awayTeam": {"abbrev": "MTL"},
                                        "homeTeam": {"abbrev": "NYR"},
                                    }
                                ],
                            }
                        ]
                    },
                ),
                "https://api-web.nhle.com/v1/gamecenter/9002/boxscore": (
                    200,
                    {
                        "playerByGameStats": {
                            "awayTeam": {"forwards": [{"name": {"default": "N. Suzuki"}, "goals": 1}], "defense": []},
                            "homeTeam": {"forwards": [{"name": {"default": "A. Lafreniere"}, "goals": 2}], "defense": []},
                        }
                    },
                ),
            }
        )
        self.assertEqual(
            settle_pending_tims_days(abbr_history, today_str="2026-11-17", settled_et="now", session=abbr_sess, allow_network=True),
            1,
        )
        self.assertEqual(abbr_history["days"][0]["results"]["hits_of_3"], 2)

    def test_per_game_team_rates_causal_rest_shot_blend_and_schema_gate(self):
        import main_pipeline

        df_t = normalize_team_frame_per_game(
            pd.DataFrame(
                [
                    {"team": "EDM", "season": 2026, "games_played": 20, "goalsFor": 70.0, "xGoalsPercentage": 0.55},
                    {"team": "SJS", "season": 2025, "games_played": 82, "goalsFor": 205.0, "xGoalsPercentage": 0.45},
                ]
            )
        )
        self.assertAlmostEqual(float(df_t.loc[0, "goalsFor_per_game"]), 3.5)
        self.assertAlmostEqual(float(df_t.loc[1, "goalsFor_per_game"]), 2.5)
        self.assertNotIn("xGoalsPercentage_per_game", df_t.columns)

        df_logs = pd.DataFrame(
            [
                {"Season": "20262027", "Team": "EDM", "Date": "2026-11-10", "Won": 1, "Team_Score": 4, "Opp_Score": 2, "Current_Streak": 0},
                {"Season": "20262027", "Team": "EDM", "Date": "2026-11-11", "Won": 0, "Team_Score": 2, "Opp_Score": 3, "Current_Streak": 1},
            ]
        )
        enriched = add_causal_game_log_features(df_logs)
        self.assertEqual(enriched["Is_B2B"].tolist(), [0.0, 1.0])
        self.assertEqual(enriched["Rest_Days"].tolist(), [3.0, 0.0])
        latest = extract_latest_team_form(df_logs)
        self.assertEqual(latest["EDM"]["Last_Game_Date"], "2026-11-11")
        self.assertEqual(latest["EDM"]["Current_Streak"], -1)

        # Verify _pairwise_win_prob applies B2B edge when game_date is tomorrow vs neutral when omitted
        class _B2BClf:
            def predict_proba(self, X):
                val = 0.5 - 0.15 * float(X["Is_B2B_diff"].iloc[0])
                return np.array([[1.0 - val, val]])

        profiles = {
            "EDM": {"Last_Game_Date": "2026-11-15"},
            "TOR": {"Last_Game_Date": "2026-11-12"},
        }
        bnd = {"model": _B2BClf(), "feature_names": ["Is_B2B_diff"], "optimal_T": 1.0}
        self.assertAlmostEqual(_pairwise_win_prob("EDM", "TOR", profiles, bnd), 0.5)
        self.assertAlmostEqual(_pairwise_win_prob("EDM", "TOR", profiles, bnd, game_date="2026-11-16"), 0.35)

        # Verify shot-pace shrinkage dampens high-shooting% hot streaks cleanly
        no_shot_b, w_g = blend_skater_goal_rate(0.35, live_gp=10, live_goals=8, n_half=30.0)
        with_shot_b, _ = blend_skater_goal_rate(0.35, live_gp=10, live_goals=8, n_half=30.0, prior_sog_gp=3.2, live_shots=32)
        self.assertAlmostEqual(w_g, 0.25)
        self.assertLess(with_shot_b, no_shot_b)

        # Old schema v1 metadata triggers retrain readiness=False cleanly
        with tempfile.TemporaryDirectory() as tmpdir:
            for fname in (
                "pairwise_game_bundle.joblib",
                "skater_regular_bundle.joblib",
                "team_profiles_snapshot.json",
                "skater_pool.csv.gz",
            ):
                with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
                    f.write("placeholder")
            with open(os.path.join(tmpdir, "inference_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"artifact_schema_version": 1}, f)
            self.assertFalse(main_pipeline._inference_artifacts_ready(tmpdir))
            with open(os.path.join(tmpdir, "inference_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"artifact_schema_version": main_pipeline.ARTIFACT_SCHEMA_VERSION}, f)
            self.assertTrue(main_pipeline._inference_artifacts_ready(tmpdir))

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

            models_dir = os.path.join(tmpdir, "models")
            docs_dir = os.path.join(tmpdir, "docs")
            train_res = run_pipeline(
                data_dir=data_dir,
                models_dir=models_dir,
                docs_dir=docs_dir,
                skip_network=True,
                mode="train",
            )
            self.assertEqual(train_res["meta"]["status"], "trained")
            for fname in (
                "pairwise_game_bundle.joblib",
                "skater_regular_bundle.joblib",
                "team_profiles_snapshot.json",
                "skater_pool.csv.gz",
                "inference_meta.json",
            ):
                self.assertTrue(os.path.exists(os.path.join(models_dir, fname)))

            # Inference stage runs cleanly from saved models even without MoneyPuck CSVs
            infer_payload = run_pipeline(
                data_dir=os.path.join(tmpdir, "empty_data"),
                models_dir=models_dir,
                docs_dir=docs_dir,
                skip_network=True,
                mode="infer",
            )
            self.assertEqual(infer_payload["meta"]["status"], "live")
            self.assertEqual(infer_payload["meta"]["run_mode"], "infer")
            self.assertEqual(len(infer_payload["power_rankings"]), 2)

            full_payload = run_pipeline(
                data_dir=data_dir,
                models_dir=models_dir,
                docs_dir=docs_dir,
                skip_network=True,
                mode="full",
            )
            self.assertEqual(full_payload["meta"]["status"], "live")
            self.assertEqual(full_payload["meta"]["run_mode"], "full")
            self.assertEqual(len(full_payload["power_rankings"]), 2)

    def test_infer_gate_post_start_window(self):
        from datetime import timezone

        payload = {
            "meta": {
                "status": "live",
                "updated_utc": "2026-11-16 21:18:00 UTC",
            },
            "games": [
                {
                    "away": "MTL",
                    "home": "NYR",
                    "date": "2026-11-16",
                    "start_time_et": "7:30 PM ET",
                    "start_time_utc": "2026-11-17T00:30:00Z",
                    "game_state": "FUT",
                },
                {
                    "away": "VAN",
                    "home": "UTA",
                    "date": "2026-11-16",
                    "start_time_et": "9:30 PM ET",
                    "game_state": "FUT",
                },
            ],
        }
        # Fallback parsing from date + start_time_et resolves 9:30 PM EST -> 02:30 UTC
        fallback_dt = parse_game_start_utc(payload["games"][1])
        self.assertEqual(fallback_dt, datetime(2026, 11, 17, 2, 30, tzinfo=timezone.utc))

        # Before 00:33 UTC (+3 min), nothing is due
        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 32, tzinfo=timezone.utc)),
            [],
        )
        # At 00:33 UTC (+3 min), MTL@NYR is due
        due = find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 33, tzinfo=timezone.utc))
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["away"], "MTL")

        # Once updated_utc >= 00:33:00 UTC, MTL@NYR does not re-trigger
        payload["meta"]["updated_utc"] = "2026-11-17 00:34:10 UTC"
        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc)),
            [],
        )
        # At 02:33 UTC (+3 min for fallback VAN@UTA), VAN@UTA becomes due
        due2 = find_due_games(payload, now_utc=datetime(2026, 11, 17, 2, 33, tzinfo=timezone.utc))
        self.assertEqual(len(due2), 1)
        self.assertEqual(due2[0]["away"], "VAN")

        # After stale max lag (>50 min past trigger), does not re-trigger
        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 3, 25, tzinfo=timezone.utc)),
            [],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "data.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            ok, reason = evaluate_gate(
                json_path,
                event_name="schedule",
                now_utc=datetime(2026, 11, 17, 2, 35, tzinfo=timezone.utc),
            )
            self.assertTrue(ok)
            self.assertIn("VAN@UTA", reason)

            skip, _ = evaluate_gate(
                json_path,
                event_name="schedule",
                now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc),
            )
            self.assertFalse(skip)

            manual_ok, _ = evaluate_gate(json_path, event_name="workflow_dispatch")
            self.assertTrue(manual_ok)

    def test_tims_timing_advisor_select_now_and_selective_defer(self):
        # 1. Single wave -> always SELECT_NOW while preserving current_best
        schedule_single = [
            {
                "away": "MTL",
                "home": "NYR",
                "is_today": True,
                "game_state": "FUT",
                "start_time_utc": "2026-11-17T00:00:00Z",
                "start_time_et": "7:00 PM ET",
            }
        ]
        groups_single = {
            "1": [{"player": "Cole Caufield", "team": "MTL", "opponent": "NYR", "final_score": 0.45}],
            "2": [{"player": "J.T. Miller", "team": "NYR", "opponent": "MTL", "final_score": 0.30}],
            "3": [{"player": "Adam Fox", "team": "NYR", "opponent": "MTL", "final_score": 0.20}],
        }
        res_single = compute_tims_deferral_advice(
            tims_groups=groups_single,
            slate_candidates=[],
            schedule_games=schedule_single,
        )
        self.assertEqual(res_single["decision"], "SELECT_NOW")
        self.assertEqual(res_single["current_best"]["1"]["player"], "Cole Caufield")
        self.assertGreater(res_single["v_now"], 0.7)

        # 2. Multi-wave where high-scoring late skaters are missing now -> SELECT_DEFER
        #    while STILL returning current_best for all 3 groups!
        schedule_multi = [
            {
                "away": "MTL",
                "home": "NYR",
                "is_today": True,
                "game_state": "FUT",
                "start_time_utc": "2026-11-17T00:00:00Z",
                "start_time_et": "7:00 PM ET",
            },
            {
                "away": "VAN",
                "home": "UTA",
                "is_today": True,
                "game_state": "FUT",
                "start_time_utc": "2026-11-17T02:30:00Z",
                "start_time_et": "9:30 PM ET",
            },
        ]
        # Weak early G1 best + weak late visible players, while 3 huge UTA/VAN stars sit hidden
        groups_weak_now = {
            "1": [
                {"player": f"Early G1-{i}", "team": "NYR", "opponent": "MTL", "final_score": 0.25}
                for i in range(6)
            ]
            + [
                {"player": "Jonathan Lekkerimäki", "team": "VAN", "opponent": "UTA", "final_score": 0.24},
                {"player": "Barrett Hayton", "team": "UTA", "opponent": "VAN", "final_score": 0.22},
            ],
            "2": [
                {"player": f"Early G2-{i}", "team": "MTL", "opponent": "NYR", "final_score": 0.20}
                for i in range(6)
            ]
            + [
                {"player": f"Late G2-{i}", "team": "VAN", "opponent": "UTA", "final_score": 0.21}
                for i in range(6)
            ],
            "3": [
                {"player": f"Early G3-{i}", "team": "NYR", "opponent": "MTL", "final_score": 0.14}
                for i in range(6)
            ]
            + [
                {"player": f"Late G3-{i}", "team": "UTA", "opponent": "VAN", "final_score": 0.16}
                for i in range(6)
            ],
        }
        slate_candidates = [
            {"player": "Logan Cooley", "team": "UTA", "opponent": "VAN", "position": "C", "final_score": 0.55},
            {"player": "Dylan Guenther", "team": "UTA", "opponent": "VAN", "position": "R", "final_score": 0.50},
            {"player": "Nick Schmaltz", "team": "UTA", "opponent": "VAN", "position": "C", "final_score": 0.48},
            {"player": "Clayton Keller", "team": "UTA", "opponent": "VAN", "position": "L", "final_score": 0.44},
            # Accent-dropped duplicate of Jonathan Lekkerimäki must NOT be counted as hidden
            {"player": "Jonathan Lekkerimaki", "team": "VAN", "opponent": "UTA", "position": "R", "final_score": 0.24},
        ] + [
            {"player": f"Late Depth-{i}", "team": "VAN", "opponent": "UTA", "position": "F", "final_score": 0.18}
            for i in range(10)
        ]
        res_defer = compute_tims_deferral_advice(
            tims_groups=groups_weak_now,
            slate_candidates=slate_candidates,
            schedule_games=schedule_multi,
        )
        self.assertEqual(res_defer["decision"], "SELECT_DEFER")
        self.assertGreater(res_defer["ev_delta"], 0.02)
        self.assertEqual(res_defer["recommended_wave_et"], "9:30 PM ET")
        self.assertEqual(res_defer["current_best"]["1"]["player"], "Early G1-0")
        self.assertEqual(res_defer["current_best"]["2"]["player"], "Late G2-0")
        hidden_names = {h["player"] for h in res_defer["hidden_upside_players"]}
        self.assertIn("Logan Cooley", hidden_names)
        self.assertNotIn("Jonathan Lekkerimaki", hidden_names)

        # 3. Multi-wave where early stars are elite and late hidden pool is weak -> SELECT_NOW
        groups_strong_now = {
            "1": [{"player": "Auston Matthews", "team": "NYR", "opponent": "MTL", "final_score": 0.65}]
            + [{"player": f"Late G1-{i}", "team": "UTA", "opponent": "VAN", "final_score": 0.25} for i in range(5)],
            "2": [{"player": "Artemi Panarin", "team": "NYR", "opponent": "MTL", "final_score": 0.45}]
            + [{"player": f"Late G2-{i}", "team": "VAN", "opponent": "UTA", "final_score": 0.19} for i in range(6)],
            "3": [{"player": "Chris Kreider", "team": "NYR", "opponent": "MTL", "final_score": 0.35}]
            + [{"player": f"Late G3-{i}", "team": "UTA", "opponent": "VAN", "final_score": 0.14} for i in range(6)],
        }
        weak_hidden = [
            {"player": f"Late Weak-{i}", "team": "VAN", "opponent": "UTA", "position": "F", "final_score": 0.12}
            for i in range(12)
        ]
        res_now = compute_tims_deferral_advice(
            tims_groups=groups_strong_now,
            slate_candidates=weak_hidden,
            schedule_games=schedule_multi,
        )
        self.assertEqual(res_now["decision"], "SELECT_NOW")
        self.assertLess(res_now["ev_delta"], 0.0)
        self.assertEqual(res_now["current_best"]["1"]["player"], "Auston Matthews")


if __name__ == "__main__":
    unittest.main()
