"""Tests for live availability adjustments (star + mid-tier depth)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.availability import apply_live_availability_to_profiles
from src.data_sync import normalize_team_abbrev


def _make_df(rows):
    return pd.DataFrame(rows)


def _base_profile():
    return {"Prior_xGoalsPct": 0.52, "Prior_FenwickPct": 0.51, "Top3_Points_per_game": 2.5, "Top3_xGoals_per_game": 1.1}


def _injury(player, team="TOR", position="C", status="Out", is_out=True, is_dtd=False):
    return {
        "player": player,
        "team": team,
        "position": position,
        "status": status,
        "is_out": is_out,
        "is_dtd": is_dtd,
    }


class TestStarAvailability(unittest.TestCase):
    """Top-3 star damping, including day-to-day (partial availability) weighting."""

    def _df(self):
        rows = [{"name": f"F{i}", "team": "TOR", "position": "C", "base_pred": 1.0 - i * 0.05} for i in range(10)]
        rows += [{"name": f"D{i}", "team": "TOR", "position": "D", "base_pred": 0.6 - i * 0.05} for i in range(4)]
        return pd.DataFrame(rows)

    def _apply(self, injuries):
        profiles = {"TOR": dict(_base_profile())}
        apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=injuries,
            skater_baseline_df=self._df(),
        )
        return profiles["TOR"]

    def test_top3_out_damps_star_features(self):
        st = self._apply([_injury("F0")])
        # orig top3 = 1.0 + 0.95 + 0.90 = 2.85; replacements F3..F5 = 2.70
        # ratio 2.70/2.85 -> eff 0.5 + 0.5*ratio
        eff = 0.5 + 0.5 * (2.70 / 2.85)
        self.assertAlmostEqual(st["Top3_Points_per_game"], 2.5 * eff, places=4)
        self.assertAlmostEqual(st["Top3_xGoals_per_game"], 1.1 * eff, places=4)

    def test_top3_dtd_is_partial_and_less_severe_than_out(self):
        dtd = self._apply([_injury("F0", status="DTD", is_out=False, is_dtd=True)])
        out = self._apply([_injury("F0")])
        # DTD blends "plays" (ratio 1.0) and "sits" (ratio 2.70/2.85).
        ratio_sit = 2.70 / 2.85
        eff_dtd = 0.5 + 0.5 * (0.5 * (1.0 + ratio_sit))
        eff_out = 0.5 + 0.5 * ratio_sit
        self.assertAlmostEqual(dtd["Top3_Points_per_game"], 2.5 * eff_dtd, places=4)
        self.assertGreater(dtd["Top3_Points_per_game"], out["Top3_Points_per_game"])
        self.assertLess(dtd["Top3_Points_per_game"], 2.5)

    def test_out_plus_dtd_combines(self):
        st = self._apply([_injury("F0"), _injury("F1", status="DTD", is_out=False, is_dtd=True)])
        ratio_hard = 2.70 / 2.85          # F0 out
        ratio_both = 2.55 / 2.85          # F0 + F1 out
        eff = 0.5 + 0.5 * (0.5 * (ratio_hard + ratio_both))
        self.assertAlmostEqual(st["Top3_Points_per_game"], 2.5 * eff, places=4)

    def test_out_wins_over_dtd_for_same_player(self):
        both = self._apply([_injury("F0", status="Out", is_out=True, is_dtd=True)])
        out_only = self._apply([_injury("F0")])
        self.assertAlmostEqual(both["Top3_Points_per_game"], out_only["Top3_Points_per_game"], places=6)

    def test_dtd_outside_top3_leaves_star_features_untouched(self):
        st = self._apply([_injury("F3", status="DTD", is_out=False, is_dtd=True)])
        self.assertEqual(st["Top3_Points_per_game"], 2.5)
        self.assertEqual(st["Top3_xGoals_per_game"], 1.1)
        # Mid-tier still sees F3 (outside the base_pred top-3).
        self.assertNotEqual(st["Prior_xGoalsPct"], 0.52)

    def test_no_injuries_leaves_star_features_untouched(self):
        st = self._apply([])
        self.assertEqual(st["Top3_Points_per_game"], 2.5)
        self.assertEqual(st["Top3_xGoals_per_game"], 1.1)


class TestAvailabilityMidTier(unittest.TestCase):
    def test_flag_can_disable_mid_tier_adjustment(self):
        import src.availability as availability

        profiles = {"TOR": dict(_base_profile())}
        df = self._df_ten_forwards()
        injuries = [_injury("F5"), _injury("F6")]
        original = availability.ENABLE_MID_TIER_INJURY_ADJUSTMENT
        try:
            availability.ENABLE_MID_TIER_INJURY_ADJUSTMENT = True
            enabled = {"TOR": dict(_base_profile())}
            apply_live_availability_to_profiles(
                profiles=enabled, starting_goalies={}, injury_records=injuries, skater_baseline_df=df
            )
            availability.ENABLE_MID_TIER_INJURY_ADJUSTMENT = False
            disabled = {"TOR": dict(_base_profile())}
            avail = apply_live_availability_to_profiles(
                profiles=disabled, starting_goalies={}, injury_records=injuries, skater_baseline_df=df
            )
        finally:
            availability.ENABLE_MID_TIER_INJURY_ADJUSTMENT = original

        self.assertLess(enabled["TOR"]["Prior_xGoalsPct"], 0.52)
        self.assertEqual(disabled["TOR"]["Prior_xGoalsPct"], 0.52)
        self.assertEqual(disabled["TOR"]["Prior_FenwickPct"], 0.51)
        # Diagnostics still report the mid-tier outs even when the scaling is off.
        self.assertEqual(len(avail["TOR"]["mid_tier_out"]), 2)

    def _df_ten_forwards(self):
        rows = []
        # top-3 high values, then mid-tier
        for i in range(10):
            rows.append({"name": f"F{i}", "team": "TOR", "position": "C", "base_pred": 1.0 - i * 0.05})
        # a couple defensemen
        for i in range(4):
            rows.append({"name": f"D{i}", "team": "TOR", "position": "D", "base_pred": 0.6 - i * 0.05})
        return _make_df(rows)

    def test_no_injuries_unchanged(self):
        profiles = {"TOR": dict(_base_profile())}
        df = self._df_ten_forwards()
        before = {k: profiles["TOR"][k] for k in ("Prior_xGoalsPct", "Prior_FenwickPct")}
        avail = apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=[],
            skater_baseline_df=df,
        )
        self.assertEqual(avail["TOR"]["mid_tier_out"], [])
        for k, v in before.items():
            self.assertEqual(profiles["TOR"][k], v)

    def test_two_mid_forwards_out_modest_scale(self):
        profiles = {"TOR": dict(_base_profile())}
        df = self._df_ten_forwards()
        # F3 and F4 are mid-tier (outside top3)
        injuries = [_injury("F3"), _injury("F4")]
        apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=injuries,
            skater_baseline_df=df,
        )
        orig = 0.52
        new = profiles["TOR"]["Prior_xGoalsPct"]
        ratio = new / orig
        self.assertLess(ratio, 1.0)
        self.assertGreaterEqual(ratio, 0.90)
        self.assertLessEqual(ratio, 0.99)
        # floor check
        self.assertGreaterEqual(profiles["TOR"]["Prior_xGoalsPct"], 0.30)

    def test_only_top3_out_noop_midtier(self):
        profiles = {"TOR": dict(_base_profile())}
        df = self._df_ten_forwards()
        injuries = [_injury("F0"), _injury("F1"), _injury("F2")]
        avail = apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=injuries,
            skater_baseline_df=df,
        )
        # Prior rates unchanged by mid-tier (top-3 handled by star path on Top3_* only)
        self.assertEqual(profiles["TOR"]["Prior_xGoalsPct"], 0.52)
        self.assertEqual(profiles["TOR"]["Prior_FenwickPct"], 0.51)
        # mid_tier_out excludes top3
        self.assertEqual(avail["TOR"]["mid_tier_out"], [])

    def test_dtd_half_weight_vs_out(self):
        df = self._df_ten_forwards()
        # out case
        profiles_out = {"TOR": dict(_base_profile())}
        apply_live_availability_to_profiles(
            profiles=profiles_out,
            starting_goalies={},
            injury_records=[_injury("F3")],
            skater_baseline_df=df,
        )
        # dtd case
        profiles_dtd = {"TOR": dict(_base_profile())}
        apply_live_availability_to_profiles(
            profiles=profiles_dtd,
            starting_goalies={},
            injury_records=[_injury("F3", status="DTD", is_out=False, is_dtd=True)],
            skater_baseline_df=df,
        )
        # dtd should reduce less (higher Prior)
        self.assertGreater(profiles_dtd["TOR"]["Prior_xGoalsPct"], profiles_out["TOR"]["Prior_xGoalsPct"])

    def test_missing_rate_keys_no_crash(self):
        profiles = {"TOR": {"Top3_Points_per_game": 2.5}}  # no Prior keys
        df = self._df_ten_forwards()
        avail = apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=[_injury("F3")],
            skater_baseline_df=df,
        )
        self.assertIn("mid_tier_out", avail["TOR"])
        # string non-numeric value not crashed
        profiles2 = {"TOR": {"Prior_xGoalsPct": "n/a", "Prior_FenwickPct": None}}
        apply_live_availability_to_profiles(
            profiles=profiles2,
            starting_goalies={},
            injury_records=[_injury("F3")],
            skater_baseline_df=df,
        )
        self.assertEqual(profiles2["TOR"]["Prior_xGoalsPct"], "n/a")

    def test_accent_initial_mismatch_matches(self):
        df = pd.DataFrame(
            [
                {"name": "S. Aho", "team": "CAR", "position": "C", "base_pred": 0.5},
                {"name": "F1", "team": "CAR", "position": "C", "base_pred": 0.4},
                {"name": "F2", "team": "CAR", "position": "C", "base_pred": 0.3},
                {"name": "F3", "team": "CAR", "position": "C", "base_pred": 0.2},
            ]
        )
        profiles = {"CAR": {"Prior_xGoalsPct": 0.50, "Prior_FenwickPct": 0.49}}
        injuries = [_injury("Sebastian Aho", team="CAR", position="C")]
        avail = apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=injuries,
            skater_baseline_df=df,
        )
        # S. Aho is top-3? value 0.5 highest -> top3, so mid-tier should NOT scale (top3 owns it)
        # To test mid-tier matching, injure a mid-tier player via accented/initial name instead:
        self.assertTrue(True)
        # Now mid-tier scenario: baseline initial name for a mid-tier player
        df2 = pd.DataFrame(
            [
                {"name": "Star1", "team": "CAR", "position": "C", "base_pred": 0.9},
                {"name": "Star2", "team": "CAR", "position": "C", "base_pred": 0.8},
                {"name": "Star3", "team": "CAR", "position": "C", "base_pred": 0.7},
                {"name": "S. Aho", "team": "CAR", "position": "C", "base_pred": 0.4},
            ]
        )
        profiles2 = {"CAR": {"Prior_xGoalsPct": 0.50, "Prior_FenwickPct": 0.49}}
        injuries2 = [_injury("Sebastian Aho", team="CAR", position="C")]
        avail2 = apply_live_availability_to_profiles(
            profiles=profiles2,
            starting_goalies={},
            injury_records=injuries2,
            skater_baseline_df=df2,
        )
        self.assertLess(profiles2["CAR"]["Prior_xGoalsPct"], 0.50)
        # mid_tier_out should list the player
        names = [x["player"] for x in avail2["CAR"]["mid_tier_out"]]
        self.assertTrue(any("Aho" in n or "Sebastian" in n for n in names) or avail2["CAR"]["mid_tier_out"] == [])
        # At least not top3, so mid_tier_out non-empty
        self.assertEqual(len(avail2["CAR"]["mid_tier_out"]), 1)

    def test_star_vs_midtier_ordering_disagreement(self):
        df = pd.DataFrame(
            [
                {"name": "A", "team": "TOR", "position": "C", "base_pred": 0.40, "gameScore": 5},
                {"name": "B", "team": "TOR", "position": "C", "base_pred": 0.35, "gameScore": 9},
                {"name": "C", "team": "TOR", "position": "C", "base_pred": 0.30, "gameScore": 8},
                {"name": "D", "team": "TOR", "position": "C", "base_pred": 0.10, "gameScore": 1},
                {"name": "E", "team": "TOR", "position": "C", "base_pred": 0.08, "gameScore": 0.5},
            ]
        )
        # B only (star top-3) -> mid-tier no-op
        profiles_b = {"TOR": {"Prior_xGoalsPct": 0.52, "Prior_FenwickPct": 0.51}}
        avail_b = apply_live_availability_to_profiles(
            profiles=profiles_b,
            starting_goalies={},
            injury_records=[_injury("B")],
            skater_baseline_df=df,
        )
        self.assertEqual(profiles_b["TOR"]["Prior_xGoalsPct"], 0.52)
        self.assertEqual(avail_b["TOR"]["mid_tier_out"], [])

        # D only (mid-tier) -> small scale
        profiles_d = {"TOR": {"Prior_xGoalsPct": 0.52, "Prior_FenwickPct": 0.51}}
        avail_d = apply_live_availability_to_profiles(
            profiles=profiles_d,
            starting_goalies={},
            injury_records=[_injury("D")],
            skater_baseline_df=df,
        )
        self.assertLess(profiles_d["TOR"]["Prior_xGoalsPct"], 0.52)
        self.assertEqual([x["player"] for x in avail_d["TOR"]["mid_tier_out"]], ["D"])

        # B + D -> same scale as D only (no double scaling from B)
        profiles_bd = {"TOR": {"Prior_xGoalsPct": 0.52, "Prior_FenwickPct": 0.51}}
        avail_bd = apply_live_availability_to_profiles(
            profiles=profiles_bd,
            starting_goalies={},
            injury_records=[_injury("B"), _injury("D")],
            skater_baseline_df=df,
        )
        self.assertAlmostEqual(profiles_bd["TOR"]["Prior_xGoalsPct"], profiles_d["TOR"]["Prior_xGoalsPct"], places=9)
        mid_names = [x["player"] for x in avail_bd["TOR"]["mid_tier_out"]]
        self.assertNotIn("B", mid_names)
        self.assertIn("D", mid_names)

    def test_gamescore_metric_precedence(self):
        # gameScore present -> used; base_pred ignored
        df = pd.DataFrame(
            [
                {"name": f"P{i}", "team": "TOR", "position": "C", "gameScore": 10 - i, "base_pred": 0.1}
                for i in range(8)
            ]
        )
        profiles = {"TOR": dict(_base_profile())}
        apply_live_availability_to_profiles(
            profiles=profiles,
            starting_goalies={},
            injury_records=[_injury("P3"), _injury("P4")],
            skater_baseline_df=df,
        )
        self.assertLess(profiles["TOR"]["Prior_xGoalsPct"], 0.52)


if __name__ == "__main__":
    unittest.main()