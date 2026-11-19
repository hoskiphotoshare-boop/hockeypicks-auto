"""Tests for the Tims timing helpers and the current-best selection ticket."""

from __future__ import annotations

import unittest
from datetime import datetime

from src.tims_timing import build_group_ticket, compute_tims_deferral_advice


def _game(away: str, home: str, start_utc: str, label: str, state: str = "FUT") -> dict:
    return {
        "away": away,
        "home": home,
        "is_today": True,
        "game_state": state,
        "start_time_utc": start_utc,
        "start_time_et": label,
    }


def _deferral_scenario() -> tuple[dict, list, list]:
    schedule = [
        _game("MTL", "NYR", "2026-11-17T00:00:00Z", "7:00 PM ET"),
        _game("VAN", "UTA", "2026-11-17T02:30:00Z", "9:30 PM ET"),
    ]
    groups = {
        "1": [{"player": f"Early G1-{i}", "team": "NYR", "opponent": "MTL", "final_score": 0.25} for i in range(6)]
        + [
            {"player": "Jonathan Lekkerimäki", "team": "VAN", "opponent": "UTA", "final_score": 0.24},
            {"player": "Barrett Hayton", "team": "UTA", "opponent": "VAN", "final_score": 0.22},
        ],
        "2": [{"player": f"Early G2-{i}", "team": "MTL", "opponent": "NYR", "final_score": 0.20} for i in range(6)]
        + [
            {"player": f"Late G2-{i}", "team": "VAN", "opponent": "UTA", "final_score": 0.21}
            for i in range(6)
        ],
        "3": [{"player": f"Early G3-{i}", "team": "NYR", "opponent": "MTL", "final_score": 0.14} for i in range(6)]
        + [
            {"player": f"Late G3-{i}", "team": "UTA", "opponent": "VAN", "final_score": 0.16}
            for i in range(6)
        ],
    }
    slate = [
        {"player": "Logan Cooley", "team": "UTA", "opponent": "VAN", "position": "C", "final_score": 0.55},
        {"player": "Dylan Guenther", "team": "UTA", "opponent": "VAN", "position": "R", "final_score": 0.50},
        {"player": "Nick Schmaltz", "team": "UTA", "opponent": "VAN", "position": "C", "final_score": 0.48},
        {"player": "Clayton Keller", "team": "UTA", "opponent": "VAN", "position": "L", "final_score": 0.44},
        {"player": "Jonathan Lekkerimaki", "team": "VAN", "opponent": "UTA", "position": "R", "final_score": 0.24},
    ] + [
        {"player": f"Late Depth-{i}", "team": "VAN", "opponent": "UTA", "position": "F", "final_score": 0.18}
        for i in range(10)
    ]
    return groups, slate, schedule


class TestTimsTiming(unittest.TestCase):
    def test_build_group_ticket_uses_only_unstarted_teams(self):
        groups = {
            "1": [
                {"player": "Started Star", "team": "AAA", "final_score": 0.9},
                {"player": "Live Pick", "team": "BBB", "final_score": 0.5},
            ]
        }
        ticket = build_group_ticket(
            groups,
            team_wave_idx={"AAA": 0, "BBB": 1},
            wave_labels=["7:00 PM ET", "9:30 PM ET"],
            allowed_teams={"BBB"},
        )
        self.assertEqual(ticket["1"]["player"], "Live Pick")
        self.assertEqual(ticket["1"]["wave_idx"], 2)

    def test_build_group_ticket_falls_back_when_no_unstarted_team(self):
        groups = {"1": [{"player": "Started Top", "team": "AAA", "final_score": 0.9}]}
        ticket = build_group_ticket(
            groups,
            team_wave_idx={"AAA": 0},
            wave_labels=["7:00 PM ET"],
            allowed_teams=set(),
        )
        self.assertEqual(ticket["1"]["player"], "Started Top")

    def test_deferral_keeps_current_best_and_sets_lock(self):
        groups, slate, schedule = _deferral_scenario()
        res = compute_tims_deferral_advice(
            tims_groups=groups,
            slate_candidates=slate,
            schedule_games=schedule,
        )
        self.assertEqual(res["decision"], "SELECT_DEFER")
        self.assertEqual(res["lock_basis"], "deferred")
        self.assertEqual(res["selection_lock_et"], "9:30 PM ET")
        lock_dt = datetime.fromisoformat(res["selection_lock_utc"].replace("Z", "+00:00"))
        self.assertIsNotNone(lock_dt)
        # The displayed/tracked ticket is the best available now, not a guess at
        # who might be listed in the later wave.
        self.assertEqual(res["selection"], res["current_best"])
        self.assertEqual(res["selection"]["1"]["player"], "Early G1-0")
        self.assertEqual(res["selection"]["2"]["player"], "Late G2-0")

    def test_started_wave_drops_out_of_current_best(self):
        groups, slate, schedule = _deferral_scenario()
        schedule[0]["game_state"] = "LIVE"  # MTL/NYR already started
        res = compute_tims_deferral_advice(
            tims_groups=groups,
            slate_candidates=slate,
            schedule_games=schedule,
        )
        self.assertEqual(res["decision"], "SELECT_NOW")
        for pick in res["selection"].values():
            self.assertIn(pick["team"], {"VAN", "UTA"})

    def test_single_wave_uses_now_ticket(self):
        schedule = [_game("MTL", "NYR", "2026-11-17T00:00:00Z", "7:00 PM ET")]
        groups = {
            "1": [{"player": "Cole Caufield", "team": "MTL", "opponent": "NYR", "final_score": 0.45}],
            "2": [{"player": "J.T. Miller", "team": "NYR", "opponent": "MTL", "final_score": 0.30}],
            "3": [{"player": "Adam Fox", "team": "NYR", "opponent": "MTL", "final_score": 0.20}],
        }
        res = compute_tims_deferral_advice(
            tims_groups=groups,
            slate_candidates=[],
            schedule_games=schedule,
        )
        self.assertEqual(res["decision"], "SELECT_NOW")
        self.assertEqual(res["lock_basis"], "now")
        self.assertEqual(res["selection"], res["current_best"])


if __name__ == "__main__":
    unittest.main()
