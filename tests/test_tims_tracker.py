"""Tests for Tim Hortons snapshot tracking with timing advice."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.tims_tracker import recompute_tims_summary, upsert_tims_snapshot


def _cand(player, team, final_score, opponent="OPP"):
    return {
        "player": player,
        "team": team,
        "opponent": opponent,
        "position": "F",
        "base_pred": final_score,
        "live_gp": 10,
        "live_goals": 4,
        "def_multiplier": 1.0,
        "final_score": final_score,
        "injury_out": False,
        "injury_dtd": False,
        "injury_status": None,
    }


class TestTimsTracker(unittest.TestCase):
    def _groups(self):
        return {
            "1": [_cand("Early G1", "AAA", 0.5), _cand("Deferred G1", "CCC", 0.4)],
            "2": [_cand("Early G2", "AAA", 0.45), _cand("Deferred G2", "CCC", 0.35)],
            "3": [_cand("Early G3", "AAA", 0.3), _cand("Deferred G3", "CCC", 0.25)],
        }

    def _advice(self, lock_utc):
        return {
            "decision": "SELECT_DEFER",
            "selection_basis": "deferred",
            "selection": {
                "1": {"player": "Deferred G1", "team": "CCC", "final_score": 0.4, "wave_et": "9:30 PM ET"},
                "2": {"player": "Deferred G2", "team": "CCC", "final_score": 0.35, "wave_et": "9:30 PM ET"},
                "3": {"player": "Deferred G3", "team": "CCC", "final_score": 0.25, "wave_et": "9:30 PM ET"},
            },
            "selection_lock_utc": lock_utc,
            "selection_lock_et": "9:30 PM ET",
            "recommended_wave_et": "9:30 PM ET",
            "v_now": 1.25,
            "v_defer_best": 1.35,
            "ev_delta": 0.10,
        }

    def test_records_deferred_ticket(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        lock_utc = "2026-11-17T02:30:00Z"
        now_utc = datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc)
        saved = upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 7:00 PM ET",
            timing_advice=self._advice(lock_utc),
            now_utc=now_utc,
        )
        self.assertTrue(saved)
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Deferred G1")
        self.assertEqual(entry["ai_picks"]["2"]["player"], "Deferred G2")
        self.assertEqual(entry["ai_picks"]["3"]["player"], "Deferred G3")
        self.assertEqual(entry["timing"]["basis"], "deferred")
        self.assertFalse(entry["picks_locked"])
        self.assertEqual(entry["locked_et"], "")

    def test_freezes_after_lock(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        lock_utc = "2026-11-17T02:30:00Z"
        now_utc = datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc)
        upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 7:00 PM ET",
            timing_advice=self._advice(lock_utc),
            now_utc=now_utc,
        )
        original_picks = history["days"][0]["ai_picks"]
        later_now = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
        saved2 = upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 10:00 PM ET",
            timing_advice=self._advice(lock_utc),
            now_utc=later_now,
        )
        self.assertFalse(saved2)
        self.assertEqual(history["days"][0]["ai_picks"], original_picks)
        saved3 = upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 10:30 PM ET",
            now_utc=later_now,
        )
        self.assertFalse(saved3)

    def test_falls_back_to_group_tops_without_advice(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        saved = upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 7:00 PM ET",
        )
        self.assertTrue(saved)
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Early G1")
        self.assertEqual(entry["ai_picks"]["2"]["player"], "Early G2")
        self.assertEqual(entry["ai_picks"]["3"]["player"], "Early G3")

    def test_locked_flag_when_captured_after_lock(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        lock_utc = "2026-11-17T02:30:00Z"
        now_utc = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)
        saved = upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="2026-11-17 10:00 PM ET",
            timing_advice=self._advice(lock_utc),
            now_utc=now_utc,
        )
        self.assertTrue(saved)
        entry = history["days"][0]
        self.assertTrue(entry["picks_locked"])
        self.assertEqual(entry["locked_et"], "2026-11-17 10:00 PM ET")

    def test_recompute_summary_includes_timing_and_lock(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        lock_utc = "2026-11-17T02:30:00Z"
        upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et="cap",
            timing_advice=self._advice(lock_utc),
            now_utc=datetime(2026, 11, 17, 0, 0, tzinfo=timezone.utc),
        )
        summary = recompute_tims_summary(history)
        rd = summary["recent_days"][0]
        self.assertIn("timing", rd)
        self.assertIn("picks_locked", rd)


if __name__ == "__main__":
    unittest.main()