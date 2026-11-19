"""Tests for Tims timing helpers and deferral advice selection ticket."""

from __future__ import annotations

import unittest
from datetime import datetime

from src.tims_timing import build_group_ticket, compute_tims_deferral_advice


class TestTimsTiming(unittest.TestCase):
    def test_build_group_ticket_excludes_wave_before_target(self):
        groups = {
            "1": [
                {"player": "Wave0", "team": "AAA", "final_score": 0.9},
                {"player": "Wave1", "team": "BBB", "final_score": 0.5},
            ]
        }
        team_wave_idx = {"AAA": 0, "BBB": 1}
        wave_labels = ["7:00 PM ET", "9:30 PM ET"]
        ticket = build_group_ticket(
            groups,
            team_wave_idx=team_wave_idx,
            wave_labels=wave_labels,
            allowed_teams={"AAA", "BBB"},
            min_wave_idx=1,
        )
        self.assertEqual(ticket["1"]["player"], "Wave1")
        self.assertEqual(ticket["1"]["wave_idx"], 2)

    def test_build_group_ticket_falls_back(self):
        groups = {
            "1": [
                {"player": "Wave0Top", "team": "AAA", "final_score": 0.9},
                {"player": "Wave0Low", "team": "AAA", "final_score": 0.5},
            ]
        }
        team_wave_idx = {"AAA": 0}
        wave_labels = ["7:00 PM ET"]
        ticket = build_group_ticket(
            groups,
            team_wave_idx=team_wave_idx,
            wave_labels=wave_labels,
            allowed_teams={"AAA"},
            min_wave_idx=1,
        )
        self.assertEqual(ticket["1"]["player"], "Wave0Top")

    def test_advice_selection_excludes_started_wave(self):
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
        ] + [
            {"player": f"Late Depth-{i}", "team": "VAN", "opponent": "UTA", "position": "F", "final_score": 0.18}
            for i in range(10)
        ]
        res = compute_tims_deferral_advice(
            tims_groups=groups_weak_now,
            slate_candidates=slate_candidates,
            schedule_games=schedule_multi,
        )
        self.assertEqual(res["decision"], "SELECT_DEFER")
        self.assertEqual(res["selection_basis"], "deferred")
        self.assertTrue(res["selection_lock_utc"])
        lock_dt = datetime.fromisoformat(res["selection_lock_utc"].replace("Z", "+00:00"))
        self.assertIsNotNone(lock_dt)
        selection = res["selection"]
        self.assertTrue(selection)
        for g, pick in selection.items():
            self.assertGreaterEqual(pick["wave_idx"], 2)
        wave0_teams = {"NYR", "MTL"}
        for pick in selection.values():
            self.assertNotIn(pick["team"], wave0_teams)
        # group 1 should not keep the wave-0 NYR player
        self.assertNotEqual(selection["1"]["player"], "Early G1-0")
        self.assertIn(selection["1"]["team"], {"VAN", "UTA"})

    def test_single_wave_uses_now_ticket(self):
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
        res = compute_tims_deferral_advice(
            tims_groups=groups_single,
            slate_candidates=[],
            schedule_games=schedule_single,
        )
        self.assertEqual(res["decision"], "SELECT_NOW")
        self.assertEqual(res["selection_basis"], "now")
        self.assertEqual(res["selection"], res["current_best"])


if __name__ == "__main__":
    unittest.main()