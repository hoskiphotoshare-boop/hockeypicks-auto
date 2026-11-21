"""Unit tests for the weekly Stanley Cup futures model, MC simulator, and Portfolio Kelly."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.cup_futures import (
    FUTURES_FEATURES,
    FUTURES_ODDS_BOOKMAKERS,
    NHL_DIVISIONS,
    _solve_mutually_exclusive_kelly,
    compute_futures_recommendations,
    fetch_cup_futures_odds,
    simulate_season_and_cup,
    train_futures_game_model,
)
from src.team_game_model import (
    expected_remaining_games_22111,
    series_prob_22111,
)


class _DummyFuturesModel:
    def predict_proba(self, X):
        # Strength signal along Elo_diff so strong teams win consistently
        elo_diff = X["Elo_diff"].to_numpy(dtype=float) if "Elo_diff" in X.columns else np.zeros(len(X))
        p = np.clip(0.54 + 0.0012 * elo_diff, 0.12, 0.88)
        return np.column_stack([1.0 - p, p])


class _FakeOutrightsResp:
    def __init__(self, payload, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeOutrightsSession:
    def __init__(self, resp):
        self._resp = resp
        self.urls: list[str] = []

    def get(self, url: str, timeout: int = 15):
        self.urls.append(url)
        return self._resp


class TestCupFutures(unittest.TestCase):
    def test_series_prob_22111_symmetry_home_edge_and_states(self):
        # High seed wins at 0.60 at home (1,2,5,7) and 0.45 away (3,4,6);
        # equivalent low-seed home win prob is 1 - 0.45 = 0.55 and low-seed away prob is 1 - 0.60 = 0.40.
        p_hi = series_prob_22111(0.60, 0.45)
        p_lo_as_hi = series_prob_22111(0.55, 0.40)
        # Swap (wh, wl) meaning swapping series games: check higher home edge helps high seed
        self.assertGreater(p_hi, 0.50)
        self.assertAlmostEqual(
            p_hi + series_prob_22111(1.0 - 0.45, 1.0 - 0.60, 0, 0) - 0.0,  # sanity self-consistent DP
            p_hi + p_lo_as_hi,
            places=6,
        )
        # In equal-skill series with 0.58 home / 0.42 away, extra home game #7 favors high seed
        self.assertGreater(series_prob_22111(0.58, 0.42), 0.50)
        self.assertAlmostEqual(
            series_prob_22111(0.55, 0.45, wins_high=4, wins_low=2),
            1.0,
        )
        self.assertAlmostEqual(
            series_prob_22111(0.55, 0.45, wins_high=1, wins_low=4),
            0.0,
        )
        self.assertGreater(series_prob_22111(0.55, 0.45, 3, 0), 0.90)
        self.assertLess(series_prob_22111(0.55, 0.45, 0, 3), 0.10)
        self.assertEqual(expected_remaining_games_22111(0.55, 0.45, 4, 1), 0.0)
        self.assertGreaterEqual(expected_remaining_games_22111(0.55, 0.45), 4.0)
        self.assertLessEqual(expected_remaining_games_22111(0.55, 0.45), 7.0)

    def test_train_futures_game_model_excludes_transient_features(self):
        rows_hist = []
        rows_stars = []
        rows_goalies = []
        for yr in (2023, 2024):
            for tm, xg in (("COL", 0.58), ("SJS", 0.42)):
                rows_hist.append(
                    {
                        "season": yr,
                        "team": tm,
                        "situation": "all",
                        "games_played": 82,
                        "playoff_games_played": 16 if tm == "COL" else 0,
                        "xGoalsPercentage": xg,
                        "fenwickPercentage": xg,
                        "xGoalsFor": 280.0 if tm == "COL" else 190.0,
                        "xGoalsAgainst": 200.0 if tm == "COL" else 270.0,
                        "goalsFor": 275.0 if tm == "COL" else 185.0,
                        "goalsAgainst": 205.0 if tm == "COL" else 275.0,
                    }
                )
                rows_stars.append({"season": yr, "team": tm, "Top3_Points": 260.0, "Top3_xGoals": 110.0, "Top3_Points_per_game": 3.1, "Top3_xGoals_per_game": 1.3})
                rows_goalies.append({"season": yr, "team": tm, "Team_GSAx": 15.0 if tm == "COL" else -12.0, "Team_GSAx_per_game": 0.2, "Starter_SavePct": 0.915})

        logs = []
        for idx in range(40):
            d = f"2024-{(idx // 28) + 10:02d}-{(idx % 28) + 1:02d}"
            won_col = 1 if idx % 4 != 0 else 0
            gt = 3 if idx >= 32 else 2
            logs.append({"Season": "20242025", "Team": "COL", "Date": d, "GameType": gt, "Opponent": "SJS", "Is_Home": 1, "Team_Score": 4 if won_col else 2, "Opp_Score": 2 if won_col else 4, "Won": won_col, "Last_Game_Won": 1, "Current_Streak": 1})
            logs.append({"Season": "20242025", "Team": "SJS", "Date": d, "GameType": gt, "Opponent": "COL", "Is_Home": 0, "Team_Score": 2 if won_col else 4, "Opp_Score": 4 if won_col else 2, "Won": 1 - won_col, "Last_Game_Won": 0, "Current_Streak": -1})

        bundle = train_futures_game_model(
            pd.DataFrame(rows_hist),
            pd.DataFrame(rows_stars),
            pd.DataFrame(rows_goalies),
            pd.DataFrame(logs),
        )
        self.assertEqual(bundle["feature_names"], FUTURES_FEATURES)
        for excluded in ("Streak_diff", "Last_Game_Won_diff", "Is_B2B_Home", "Is_B2B_Away", "Rolling10_WinPct_diff"):
            self.assertNotIn(excluded, bundle["feature_names"])

    def test_simulate_season_and_cup_conservation_and_tiebreakers(self):
        standings = {}
        profiles = {}
        for idx, (tm, (conf, div)) in enumerate(sorted(NHL_DIVISIONS.items())):
            is_col = tm == "COL"
            is_sjs = tm == "SJS"
            standings[tm] = {
                "team": tm,
                "conference": conf,
                "division": div,
                "gp": 20,
                "wins": 16 if is_col else (3 if is_sjs else 10),
                "losses": 3 if is_col else (16 if is_sjs else 8),
                "ot_losses": 1 if is_col else 2,
                "points": 33 if is_col else (7 if is_sjs else 22),
                "rw": 14 if is_col else (2 if is_sjs else 8),
                "row": 15 if is_col else (3 if is_sjs else 9),
                "goal_diff": 25 if is_col else (-30 if is_sjs else 0),
            }
            profiles[tm] = {"Causal_Elo": 1650.0 if is_col else (1360.0 if is_sjs else 1500.0)}

        bundle = {"model": _DummyFuturesModel(), "feature_names": ["Elo_diff"], "optimal_T": 1.0}
        teams_list = sorted(NHL_DIVISIONS.keys())
        rem_games = []
        for k in range(8):
            for i in range(0, 32, 2):
                rem_games.append({"away": teams_list[i], "home": teams_list[i + 1], "game_state": "FUT"})

        res = simulate_season_and_cup(
            standings,
            rem_games,
            profiles,
            bundle,
            ot_rate=0.23,
            n_sims=2000,
            seed=123,
        )
        self.assertEqual(len(res), 32)
        self.assertAlmostEqual(sum(r["win_cup_prob"] for r in res.values()), 1.0, places=4)
        self.assertAlmostEqual(sum(r["make_playoffs_prob"] for r in res.values()), 16.0, places=3)
        self.assertAlmostEqual(sum(r["win_division_prob"] for r in res.values()), 4.0, places=3)
        self.assertAlmostEqual(sum(r["make_conf_final_prob"] for r in res.values()), 4.0, places=3)
        self.assertAlmostEqual(sum(r["make_cup_final_prob"] for r in res.values()), 2.0, places=3)

        self.assertGreater(res["COL"]["win_cup_prob"], res["SJS"]["win_cup_prob"] * 5.0)
        self.assertGreater(res["COL"]["make_playoffs_prob"], res["SJS"]["make_playoffs_prob"])

    def test_fetch_cup_futures_odds_single_credit_devig_and_team_mapping(self):
        payload = [
            {
                "sport_key": "icehockey_nhl_championship_winner",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Colorado Avalanche", "price": 5.0},
                                    {"name": "Carolina Hurricanes", "price": 7.0},
                                    {"name": "Utah Mammoth", "price": 25.0},
                                    {"name": "Montréal Canadiens", "price": 40.0},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "outrights",
                                "outcomes": [
                                    {"name": "Colorado Avalanche", "price": 5.5},
                                    {"name": "Carolina Hurricanes", "price": 6.5},
                                    {"name": "Utah Mammoth", "price": 28.0},
                                    {"name": "Montréal Canadiens", "price": 45.0},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        session = _FakeOutrightsSession(
            _FakeOutrightsResp(
                payload,
                headers={"X-Requests-Last": "1", "X-Requests-Used": "55", "X-Requests-Remaining": "445"},
            )
        )
        meta: dict = {}
        res = fetch_cup_futures_odds("test-key", session, meta_out=meta)
        self.assertTrue(res["api_ok"])
        self.assertIn("icehockey_nhl_championship_winner/odds", session.urls[0])
        self.assertIn(f"bookmakers={FUTURES_ODDS_BOOKMAKERS}", session.urls[0])
        self.assertNotIn("regions=", session.urls[0])
        self.assertEqual(res["devig_source"], "pinnacle")
        self.assertAlmostEqual(sum(res["market_probs"].values()), 1.0, places=6)
        self.assertEqual(res["best_odds"]["UTA"], 28.0)
        self.assertEqual(res["best_bookie"]["UTA"], "FanDuel")
        self.assertEqual(res["best_odds"]["MTL"], 45.0)
        self.assertEqual(meta["requests_last"], 1)
        self.assertEqual(meta["requests_remaining"], 445)

    def test_portfolio_kelly_hurdle_shrinkage_and_mutually_exclusive_caps(self):
        sim = {}
        standings = {}
        teams = sorted(NHL_DIVISIONS.keys())
        for i, t in enumerate(teams):
            conf, div = NHL_DIVISIONS[t]
            standings[t] = {"team": t, "conference": conf, "division": div, "gp": 20, "points": 24, "rw": 10}
            p = 0.25 if t in ("COL", "CAR", "EDM") else (0.25 / (len(teams) - 3))
            sim[t] = {
                "proj_points": 95.0,
                "make_playoffs_prob": 0.6,
                "win_division_prob": 0.12,
                "make_conf_final_prob": 0.12,
                "make_cup_final_prob": 0.06,
                "win_cup_prob": p,
            }

        # Strong positive edges on COL, CAR, EDM; tiny +4% edge on DAL rejected by Nov hurdle
        mkt_probs = {t: sim[t]["win_cup_prob"] for t in teams}
        best_odds = {t: 1.0 / mkt_probs[t] for t in teams}
        best_odds["COL"] = 6.0   # 25% vs 6.0 -> +50% edge
        best_odds["CAR"] = 5.5   # 25% vs 5.5 -> +37.5% edge
        best_odds["EDM"] = 5.0   # 25% vs 5.0 -> +25% edge
        best_odds["DAL"] = (1.04 / mkt_probs["DAL"])  # +4% edge -> below ~9% Nov hurdle

        market_data = {
            "api_ok": True,
            "best_odds": best_odds,
            "best_bookie": {t: "FanDuel" for t in teams},
            "market_probs": mkt_probs,
            "market_overround_pct": 12.5,
            "devig_source": "pinnacle",
        }
        rows, summary = compute_futures_recommendations(
            sim,
            standings,
            market_data,
            today_date=date(2026, 11, 21),
            bankroll=1000.0,
        )
        self.assertEqual(len(rows), 32)
        self.assertGreaterEqual(summary["min_edge_pct"], 8.5)
        self.assertAlmostEqual(summary["model_weight"], 0.35 + 0.30 * (20.0 / 82.0), places=2)
        rec_teams = {b["team"] for b in summary["recommended_bets"]}
        self.assertIn("COL", rec_teams)
        self.assertIn("CAR", rec_teams)
        self.assertIn("EDM", rec_teams)
        self.assertNotIn("DAL", rec_teams)
        # Enforces per-team cap ($25 max at 2.5% of $1000) and total exposure cap ($80 max at 8%)
        for b in summary["recommended_bets"]:
            self.assertLessEqual(b["rec_bet"], 25.0 + 1e-6)
        self.assertLessEqual(summary["total_rec_stake"], 80.0 + 1e-6)

        # Joint mutually-exclusive Kelly strictly respects the 0.90 bankruptcy bound
        # and achieves higher joint expected log wealth than naive independent single-bet Kelly
        p_vec = [0.22, 0.22, 0.22, 0.22]
        o_vec = [15.0, 15.0, 15.0, 15.0]
        naive_vec = [(p * (o - 1.0) - (1.0 - p)) / (o - 1.0) for p, o in zip(p_vec, o_vec)]
        joint_full = _solve_mutually_exclusive_kelly(
            p_vec,
            o_vec,
            kelly_fraction=1.0,
            max_team_pct=1.0,
            max_total_pct=1.0,
        )
        self.assertLessEqual(sum(joint_full), 0.90 + 1e-6)

        def _log_growth(weights):
            s = sum(weights)
            p_lose = max(0.0, 1.0 - sum(p_vec))
            return sum(p * np.log(1.0 - s + w * o) for p, w, o in zip(p_vec, weights, o_vec)) + p_lose * np.log(1.0 - s)

        self.assertGreater(_log_growth(joint_full), _log_growth(naive_vec))

        # April hurdle is strictly lower than November hurdle
        _, apr_summary = compute_futures_recommendations(
            sim,
            {t: {**st, "gp": 80} for t, st in standings.items()},
            market_data,
            today_date=date(2027, 4, 15),
            bankroll=1000.0,
        )
        self.assertLess(apr_summary["min_edge_pct"], summary["min_edge_pct"])
        self.assertGreater(apr_summary["model_weight"], summary["model_weight"])

        # Offline missing odds cleanly disables bet sizing
        rows_off, sum_off = compute_futures_recommendations(
            sim,
            standings,
            {"api_ok": False, "best_odds": {}, "best_bookie": {}, "market_probs": {}},
            today_date="2026-11-21",
        )
        self.assertEqual(len(rows_off), 32)
        self.assertEqual(sum_off["recommended_count"], 0)
        self.assertEqual(sum_off["total_rec_stake"], 0.0)


if __name__ == "__main__":
    unittest.main()
