"""Tests for Tim Hortons snapshot tracking with timing advice."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.tims_tracker import recompute_tims_summary, upsert_tims_snapshot

# 9:30 PM ET on 2026-11-17 is 02:30Z on 2026-11-18.
LOCK_UTC = "2026-11-18T02:30:00Z"
CAPTURE_NOW = datetime(2026, 11, 17, 21, 17, tzinfo=timezone.utc)  # 4:17 PM ET
AFTER_LOCK = datetime(2026, 11, 18, 3, 36, tzinfo=timezone.utc)  # 10:36 PM ET


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
        """Early (7:00 PM) players outscore the later (9:30 PM) ones."""
        return {
            "1": [_cand("Early G1", "AAA", 0.5), _cand("Late G1", "CCC", 0.4)],
            "2": [_cand("Early G2", "BBB", 0.45), _cand("Late G2", "CCC", 0.35)],
            "3": [_cand("Early G3", "AAA", 0.3), _cand("Late G3", "CCC", 0.25)],
        }

    def _late_groups(self):
        """Only the last wave is left; the early players are gone."""
        return {
            "1": [_cand("Last G1", "DDD", 0.22)],
            "2": [_cand("Last G2", "DDD", 0.21)],
            "3": [_cand("Last G3", "CCC", 0.20)],
        }

    def _advice(self, selection, lock_utc=LOCK_UTC, lock_basis="deferred"):
        return {
            "decision": "SELECT_DEFER" if lock_basis == "deferred" else "SELECT_NOW",
            "lock_basis": lock_basis,
            "selection": selection,
            "selection_lock_utc": lock_utc,
            "selection_lock_et": "9:30 PM ET" if lock_utc else "",
            "recommended_wave_et": "9:30 PM ET",
            "v_now": 1.25,
            "v_defer_best": 1.35,
            "ev_delta": 0.10,
        }

    def _upsert(self, history, groups, *, advice=None, captured="cap", now=CAPTURE_NOW):
        return upsert_tims_snapshot(
            history,
            date_str="2026-11-17",
            context={"nhl_season": "20262027"},
            schedule_info={"active_game_type": 2, "active_mode": "Regular Season"},
            skater_bundle={"n_half": 25.0},
            tims_groups=groups,
            captured_et=captured,
            timing_advice=advice,
            now_utc=now,
        )

    def _picks(self, selection):
        return {
            g: {"player": p["player"], "team": p["team"], "final_score": p["final_score"]}
            for g, p in selection.items()
        }

    def test_records_current_best_with_deferral_lock(self):
        history = {"days": [], "summary": {}}
        groups = self._groups()
        advice = self._advice(self._picks({
            "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
            "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
            "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
        }))
        self.assertTrue(self._upsert(history, groups, advice=advice))
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Early G1")
        self.assertEqual(entry["ai_picks"]["2"]["player"], "Early G2")
        self.assertEqual(entry["ai_picks"]["3"]["player"], "Early G3")
        self.assertEqual(entry["timing"]["basis"], "deferred")
        self.assertEqual(entry["timing"]["lock_utc"], LOCK_UTC)
        self.assertFalse(entry["picks_locked"])
        self.assertEqual(entry["locked_et"], "")

    def test_refreshes_to_ticket_in_effect_at_lock(self):
        history = {"days": [], "summary": {}}
        early = self._advice(self._picks({
            "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
            "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
            "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
        }))
        self.assertTrue(self._upsert(history, self._groups(), advice=early))

        # 7:05 PM ET: the 7:00 PM games are live, so the 9:30 PM list is the
        # best available ticket. A pre-lock run must be able to refresh it.
        groups = self._late_groups()
        at_lock = self._advice(
            self._picks({
                "1": {"player": "Last G1", "team": "DDD", "final_score": 0.22},
                "2": {"player": "Last G2", "team": "DDD", "final_score": 0.21},
                "3": {"player": "Last G3", "team": "CCC", "final_score": 0.20},
            }),
            lock_utc="2026-11-18T02:25:00Z",
        )
        before_lock = datetime(2026, 11, 18, 0, 5, tzinfo=timezone.utc)
        self.assertTrue(
            self._upsert(history, groups, advice=at_lock, captured="2026-11-17 7:05 PM ET", now=before_lock)
        )
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Last G1")
        self.assertEqual(entry["timing"]["basis"], "deferred")
        self.assertFalse(entry["picks_locked"])

    def test_freezes_after_lock(self):
        history = {"days": [], "summary": {}}
        early = self._advice(self._picks({
            "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
            "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
            "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
        }))
        self.assertTrue(self._upsert(history, self._groups(), advice=early))
        original_picks = history["days"][0]["ai_picks"]

        late = self._advice(self._picks({
            "1": {"player": "Last G1", "team": "DDD", "final_score": 0.22},
            "2": {"player": "Last G2", "team": "DDD", "final_score": 0.21},
            "3": {"player": "Last G3", "team": "CCC", "final_score": 0.20},
        }))
        self.assertFalse(
            self._upsert(history, self._late_groups(), advice=late, captured="2026-11-17 10:36 PM ET", now=AFTER_LOCK)
        )
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"], original_picks)
        self.assertEqual(entry["timing"]["basis"], "deferred")
        self.assertTrue(entry["picks_locked"])
        self.assertEqual(entry["locked_et"], "cap")

        # A third run without advice must also leave the frozen ticket alone.
        self.assertFalse(
            self._upsert(history, self._late_groups(), captured="2026-11-17 10:45 PM ET", now=AFTER_LOCK)
        )
        self.assertEqual(history["days"][0]["ai_picks"], original_picks)

    def test_falls_back_to_group_tops_without_advice(self):
        history = {"days": [], "summary": {}}
        self.assertTrue(self._upsert(history, self._groups()))
        entry = history["days"][0]
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Early G1")
        self.assertEqual(entry["ai_picks"]["2"]["player"], "Early G2")
        self.assertEqual(entry["ai_picks"]["3"]["player"], "Early G3")
        self.assertEqual(entry["timing"], {})
        self.assertFalse(entry["picks_locked"])

    def test_locked_flag_when_captured_after_lock(self):
        history = {"days": [], "summary": {}}
        advice = self._advice(self._picks({
            "1": {"player": "Last G1", "team": "DDD", "final_score": 0.22},
            "2": {"player": "Last G2", "team": "DDD", "final_score": 0.21},
            "3": {"player": "Last G3", "team": "CCC", "final_score": 0.20},
        }))
        self.assertTrue(
            self._upsert(history, self._late_groups(), advice=advice, captured="2026-11-17 10:36 PM ET", now=AFTER_LOCK)
        )
        entry = history["days"][0]
        self.assertTrue(entry["picks_locked"])
        self.assertEqual(entry["locked_et"], "2026-11-17 10:36 PM ET")

    def test_recompute_summary_includes_timing_and_lock(self):
        history = {"days": [], "summary": {}}
        advice = self._advice(self._picks({
            "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
            "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
            "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
        }))
        self._upsert(history, self._groups(), advice=advice)
        summary = recompute_tims_summary(history)
        rd = summary["recent_days"][0]
        self.assertEqual(rd["timing"]["lock_utc"], LOCK_UTC)
        self.assertEqual(rd["timing"]["basis"], "deferred")
        self.assertIs(rd["picks_locked"], False)

    def test_preserves_deferred_basis_when_wave_one_starts_and_advisor_switches_to_now(self):
        history = {"days": [], "summary": {}}
        early = self._advice(
            self._picks({
                "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
                "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
                "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
            }),
            lock_utc=LOCK_UTC,
            lock_basis="deferred",
        )
        early["first_wave_utc"] = "2026-11-18T00:00:00Z"
        self.assertTrue(self._upsert(history, self._groups(), advice=early))

        # 7:05 PM ET (after Wave 1 7:00 PM started, before 9:30 PM lock):
        # advisor now sees 9:30 PM as Wave 1 and reports lock_basis="now".
        post_w1 = self._advice(
            self._picks({
                "1": {"player": "Late G1", "team": "CCC", "final_score": 0.4},
                "2": {"player": "Late G2", "team": "CCC", "final_score": 0.35},
                "3": {"player": "Late G3", "team": "CCC", "final_score": 0.25},
            }),
            lock_utc=LOCK_UTC,
            lock_basis="now",
        )
        post_w1["first_wave_utc"] = LOCK_UTC
        after_w1 = datetime(2026, 11, 18, 0, 5, tzinfo=timezone.utc)
        self.assertTrue(
            self._upsert(history, self._late_groups(), advice=post_w1, captured="2026-11-17 7:05 PM ET", now=after_w1)
        )
        entry = history["days"][0]
        self.assertEqual(entry["timing"]["basis"], "deferred")
        self.assertEqual(entry["timing"]["first_wave_utc"], "2026-11-18T00:00:00Z")
        self.assertEqual(entry["ai_picks"]["1"]["player"], "Late G1")
        self.assertFalse(entry["picks_locked"])

    def test_resets_to_now_when_advisor_reverts_before_wave_one(self):
        history = {"days": [], "summary": {}}
        early = self._advice(
            self._picks({
                "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
                "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
                "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
            }),
            lock_utc=LOCK_UTC,
            lock_basis="deferred",
        )
        early["first_wave_utc"] = "2026-11-18T00:00:00Z"
        self.assertTrue(self._upsert(history, self._groups(), advice=early))

        # 5:00 PM ET (still before Wave 1 7:00 PM): advisor reverts to Wave 1 NOW lock.
        reverted = self._advice(
            self._picks({
                "1": {"player": "Early G1", "team": "AAA", "final_score": 0.5},
                "2": {"player": "Early G2", "team": "BBB", "final_score": 0.45},
                "3": {"player": "Early G3", "team": "AAA", "final_score": 0.3},
            }),
            lock_utc="2026-11-18T00:00:00Z",
            lock_basis="now",
        )
        reverted["first_wave_utc"] = "2026-11-18T00:00:00Z"
        before_w1 = datetime(2026, 11, 17, 22, 0, tzinfo=timezone.utc)
        self.assertTrue(
            self._upsert(history, self._groups(), advice=reverted, captured="2026-11-17 5:00 PM ET", now=before_w1)
        )
        entry = history["days"][0]
        self.assertEqual(entry["timing"]["basis"], "now")
        self.assertEqual(entry["timing"]["lock_utc"], "2026-11-18T00:00:00Z")

    def test_finalize_locks_prior_day_before_first_game(self):
        from datetime import datetime, timezone
        from src.tims_tracker import finalize_tims_locks

        history = {
            "days": [
                {
                    "date": "2026-11-24",
                    "captured_et": "2026-11-24 07:41 PM ET",
                    "picks_locked": False,
                    "timing": {
                        "lock_utc": "2026-11-25T01:00:00Z",
                        "lock_et": "8:00 PM ET",
                        "first_wave_utc": "2026-11-25T01:00:00Z",
                    },
                }
            ]
        }
        changed = finalize_tims_locks(history, now_utc=datetime(2026, 11, 25, 3, 0, tzinfo=timezone.utc))
        day = history["days"][0]
        self.assertEqual(changed, 1)
        self.assertTrue(day["picks_locked"])
        self.assertTrue(day["locked_before_first_game"])
        self.assertEqual(day["lock_wave"], 1)
        self.assertEqual(day["locked_et"], "2026-11-24 07:41 PM ET")


if __name__ == "__main__":
    unittest.main()
