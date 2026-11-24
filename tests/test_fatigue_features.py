import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from src.team_game_model import (
    BASE_PAIRWISE_FEATURES,
    CAUSAL_PAIRWISE_FEATURES,
    ENABLE_FATIGUE_FEATURES,
    FATIGUE_PAIRWISE_FEATURES,
    _fatigue_metrics_for_date,
    _pairwise_win_prob,
    add_causal_game_log_features,
    compute_power_rankings_and_defense,
    extract_latest_team_form,
    pairwise_feature_names,
)
from src.team_geo import team_coords


class TestSchemaGate(unittest.TestCase):
    """Fatigue regressed on the 2026 holdout, so the schema must default to 21 features."""

    def test_fatigue_is_off_by_default(self):
        self.assertFalse(ENABLE_FATIGUE_FEATURES)
        self.assertEqual(len(BASE_PAIRWISE_FEATURES), 21)
        self.assertEqual(CAUSAL_PAIRWISE_FEATURES, BASE_PAIRWISE_FEATURES)
        self.assertEqual(pairwise_feature_names(), BASE_PAIRWISE_FEATURES)

    def test_opt_in_adds_only_fatigue_features(self):
        full = pairwise_feature_names(include_fatigue=True)
        self.assertEqual(full, BASE_PAIRWISE_FEATURES + FATIGUE_PAIRWISE_FEATURES)
        for feat in FATIGUE_PAIRWISE_FEATURES:
            self.assertNotIn(feat, BASE_PAIRWISE_FEATURES)


class TestFatigueFeatures(unittest.TestCase):
    def _synthetic_logs(self):
        rows = []
        # 5 games, last three prior to 01-09 are away
        games = [
            ("2026-01-01", 1, "CGY"),
            ("2026-01-03", 0, "CGY"),
            ("2026-01-05", 0, "CGY"),
            ("2026-01-07", 0, "CGY"),
            ("2026-01-09", 1, "CGY"),
        ]
        for d, home, opp in games:
            rows.append(
                {
                    "Season": "20262027",
                    "Team": "EDM",
                    "Date": d,
                    "Won": 1,
                    "Team_Score": 3,
                    "Opp_Score": 2,
                    "Is_Home": home,
                    "Opp": opp,
                    "Opponent": opp,
                }
            )
        return pd.DataFrame(rows)

    def test_causal_window_and_road_trip(self):
        df = self._synthetic_logs()
        enriched = add_causal_game_log_features(df)
        # inaugural game
        first = enriched[enriched["Date"] == "2026-01-01"].iloc[0]
        self.assertEqual(first["Games_Last7"], 0.0)
        self.assertEqual(first["RoadTrip_Length"], 0.0)
        # game on 01-09: 4 prior games within 7 days
        last = enriched[enriched["Date"] == "2026-01-09"].iloc[0]
        self.assertEqual(last["Games_Last7"], 3.0)
        self.assertEqual(last["RoadTrip_Length"], 3.0)

    def test_live_vs_historical_parity(self):
        df = self._synthetic_logs()
        # target game appended
        target_date = "2026-01-11"
        target = pd.DataFrame(
            [
                {
                    "Season": "20262027",
                    "Team": "EDM",
                    "Date": target_date,
                    "Won": 0,
                    "Team_Score": 2,
                    "Opp_Score": 3,
                    "Is_Home": 1,
                    "Opp": "CGY",
                    "Opponent": "CGY",
                }
            ]
        )
        full = pd.concat([df, target], ignore_index=True)
        enriched = add_causal_game_log_features(full)
        trow = enriched[enriched["Date"] == target_date].iloc[0]

        prior_logs = df.copy()
        profiles = extract_latest_team_form(prior_logs)
        stat = profiles["EDM"]
        live = _fatigue_metrics_for_date(stat, target_date, [], "EDM")
        self.assertAlmostEqual(live["Games_Last7"], float(trow["Games_Last7"]), places=3)
        self.assertAlmostEqual(live["Travel_Last7"], float(trow["Travel_Last7"]), places=3)
        self.assertAlmostEqual(live["TZ_Shift"], float(trow["TZ_Shift"]), places=3)
        self.assertAlmostEqual(live["RoadTrip_Length"], float(trow["RoadTrip_Length"]), places=3)
        self.assertAlmostEqual(live["Rest_x_Travel"], float(trow["Rest_x_Travel"]), places=3)

    def test_neutral_power_ranking_behavior(self):
        class Recorder:
            def __init__(self):
                self.last = None

            def predict_proba(self, X):
                self.last = X.copy()
                return np.array([[0.5, 0.5]])

        profiles = {
            "EDM": {"Last_Game_Date": "2026-01-09", "Recent_Game_Dates": [], "Recent_Game_Venues": [], "Recent_Is_Home": []},
            "TOR": {"Last_Game_Date": "2026-01-09", "Recent_Game_Dates": [], "Recent_Game_Venues": [], "Recent_Is_Home": []},
        }
        rec = Recorder()
        bundle = {"model": rec, "feature_names": ["Games_Last7_diff"], "optimal_T": 1.0, "lr_model": None, "lr_weight": 0.0}
        # no game_date/venue -> neutral
        _pairwise_win_prob("EDM", "TOR", profiles, bundle)
        self.assertEqual(float(rec.last["Games_Last7_diff"].iloc[0]), 0.0)
        # game_date but no venue -> neutral
        rec2 = Recorder()
        bundle2 = {"model": rec2, "feature_names": ["Games_Last7_diff"], "optimal_T": 1.0, "lr_model": None, "lr_weight": 0.0}
        _pairwise_win_prob("EDM", "TOR", profiles, bundle2, game_date="2026-01-11")
        self.assertEqual(float(rec2.last["Games_Last7_diff"].iloc[0]), 0.0)

    def test_unknown_team_fallback(self):
        self.assertIsNone(team_coords("ZZZ"))
        m = _fatigue_metrics_for_date({}, "2026-01-11", [], "ZZZ")
        self.assertEqual(m["Games_Last7"], 0.0)
        self.assertEqual(m["Travel_Last7"], 0.0)
        self.assertEqual(m["TZ_Shift"], 0.0)
        self.assertEqual(m["RoadTrip_Length"], 0.0)
        self.assertEqual(m["Rest_x_Travel"], 0.0)


if __name__ == "__main__":
    unittest.main()
