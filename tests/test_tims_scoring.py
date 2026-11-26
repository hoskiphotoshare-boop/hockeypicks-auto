"""Unit tests for the Tims pick-scoring strategy switch."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.skater_model import resolve_tims_final_score, score_skater_candidates, tims_score_strategy
from src.tims_tracker import _compact_candidate


class _FakeImputer:
    def transform(self, X):
        return X


class _FakeReg:
    def predict(self, X):
        return np.full(len(X), 0.30)


class TimsScoreStrategyTests(unittest.TestCase):
    def test_default_strategy_is_model(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TIMS_SCORE_STRATEGY", None)
            self.assertEqual(tims_score_strategy(), "model")

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"TIMS_SCORE_STRATEGY": "model"}):
            self.assertEqual(tims_score_strategy(), "model")

    def test_prior_rate_wins_when_selected(self) -> None:
        self.assertAlmostEqual(resolve_tims_final_score(0.55, 0.61, "prior_gp"), 0.55)

    def test_missing_prior_falls_back_to_model(self) -> None:
        self.assertAlmostEqual(resolve_tims_final_score(None, 0.61), 0.61)

    def test_model_strategy_ignores_prior(self) -> None:
        self.assertAlmostEqual(resolve_tims_final_score(0.55, 0.61, "model"), 0.61)

    def test_zero_prior_rate_is_respected(self) -> None:
        self.assertAlmostEqual(resolve_tims_final_score(0.0, 0.61, "prior_gp"), 0.0)

    def test_scoring_uses_prior_rate_when_pool_has_it(self) -> None:
        bundle = {
            "model": _FakeReg(),
            "imputer": _FakeImputer(),
            "features": ["I_F_shotsOnGoal_per_game"],
            "n_half": 10.0,
        }
        baseline = pd.DataFrame(
            [
                {"playerId": 1, "name": "High Prior", "name_clean": "high prior", "team": "EDM",
                 "position": "C", "games_played": 70, "I_F_shotsOnGoal_per_game": 3.0,
                 "prior_goals_per_game": 0.55},
                {"playerId": 2, "name": "Low Prior", "name_clean": "low prior", "team": "EDM",
                 "position": "C", "games_played": 70, "I_F_shotsOnGoal_per_game": 3.0,
                 "prior_goals_per_game": 0.10},
            ]
        )
        tims_df = pd.DataFrame(
            [
                {"group": 1, "player_name": "High Prior", "name_clean": "high prior",
                 "team": "EDM", "opponentTeam": "SJS", "position": "C"},
                {"group": 1, "player_name": "Low Prior", "name_clean": "low prior",
                 "team": "EDM", "opponentTeam": "SJS", "position": "C"},
            ]
        )
        with patch.dict(os.environ, {"TIMS_SCORE_STRATEGY": "prior_gp"}):
            scores = score_skater_candidates(
                bundle=bundle,
                df_baseline=baseline,
                df_tims=tims_df,
                daily_matchups={"EDM": "SJS", "SJS": "EDM"},
                live_skaters={},
                defense_multipliers={"SJS": 1.0, "EDM": 1.0},
                active_mode="Regular Season",
            )
        top = scores["tims_groups"]["1"][0]
        self.assertEqual(top["player"], "High Prior")
        self.assertAlmostEqual(top["final_score"], 0.55, places=3)
        self.assertAlmostEqual(top["model_score"], 0.30, places=3)

    def test_compact_candidate_keeps_model_score_for_shadow(self) -> None:
        item = {"player": "X", "team": "EDM", "final_score": 0.55, "model_score": 0.61}
        self.assertAlmostEqual(_compact_candidate(item)["model_score"], 0.61)


if __name__ == "__main__":
    unittest.main()
