"""Tests for the live Tims shadow report."""

from __future__ import annotations

import unittest

from scripts.tims_shadow_report import summarize, to_markdown


def _cand(player: str, model: float, prior: float, goals: int, prior_gp: int = 80, live_gp: int = 10) -> dict:
    return {
        "player": player,
        "model_score": model,
        "prior_goals_per_game": prior,
        "prior_gp": prior_gp,
        "prior_goals": round(prior * prior_gp),
        "live_gp": live_gp,
        "live_goals": goals,
        "injury_out": False,
    }


def _day(date: str, groups: dict, player_goals: dict, settled: bool = True, hits_of_3: int | None = None) -> dict:
    results: dict = {"player_goals": player_goals}
    if hits_of_3 is not None:
        results["hits_of_3"] = hits_of_3
    return {"date": date, "settled": settled, "groups": groups, "results": results}


class TimsShadowReportTests(unittest.TestCase):
    def test_strategies_pick_different_players_and_grade(self) -> None:
        history = {
            "days": [
                _day(
                    "2026-11-27",
                    {
                        "1": [_cand("A", 0.40, 0.50, 6), _cand("B", 0.60, 0.10, 1)],
                        "2": [_cand("C", 0.30, 0.30, 3), _cand("D", 0.20, 0.20, 2)],
                        "3": [_cand("E", 0.15, 0.15, 1), _cand("F", 0.25, 0.05, 1)],
                    },
                    {"A": 1, "B": 0, "C": 0, "D": 1, "E": 0, "F": 0},
                    hits_of_3=1,
                )
            ]
        }
        report = summarize(history, days=30)
        self.assertFalse(report["ready"])
        self.assertEqual(report["remaining_days"], 29)
        # Model picks the highest model score: B, C, F -> zero goals.
        self.assertEqual(report["summary"]["model"]["mean_hits"], 0.0)
        # Prior/blend/emp pick the prior-rate leaders: A, C, E -> one goal.
        self.assertEqual(report["summary"]["prior_gp"]["mean_hits"], 1.0)
        self.assertEqual(report["summary"]["prior_gp"]["expected_points_per_day"], 10.0)
        self.assertEqual(report["summary"]["blend_50"]["mean_hits"], 1.0)
        md = to_markdown(report)
        self.assertIn("| Model (shipped)", md)
        self.assertIn("Window:", md)
        self.assertIn("E[pts/day]", md)
        # The shipped ticket is carried through when hits_of_3 exists.
        self.assertIn("shipped_actual", report["summary"])

    def test_missing_prior_fields_are_skipped(self) -> None:
        history = {"days": [_day("2026-11-26", {"1": [{"player": "X", "final_score": 0.4}]}, {})]}
        report = summarize(history, days=30)
        self.assertEqual(report["eligible_days"], 0)
        self.assertEqual(report["window"]["days"], 0)

    def test_ready_when_window_is_full(self) -> None:
        day = _day(
            "2026-11-27",
            {
                "1": [_cand("A", 0.4, 0.5, 6)],
                "2": [_cand("C", 0.3, 0.3, 3)],
                "3": [_cand("E", 0.2, 0.2, 1)],
            },
            {"A": 1, "C": 1, "E": 0},
        )
        report = summarize({"days": [day]}, days=1)
        self.assertTrue(report["ready"])
        self.assertEqual(report["remaining_days"], 0)


if __name__ == "__main__":
    unittest.main()
