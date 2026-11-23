"""Tests for the persisted Tims board union and the live Tims scoreboard."""

from __future__ import annotations

import unittest
from unittest import mock

from src.live_scoreboard import build_tims_scoreboard
from src.tims_boards import board_players_for, update_board_day


def _box(away_forwards, away_defense=None, home_forwards=None, home_defense=None):
    def side(forwards, defense):
        return {
            "forwards": [
                {"name": {"default": name}, "goals": goals, "position": "C"}
                for name, goals in (forwards or [])
            ],
            "defense": [
                {"name": {"default": name}, "goals": goals, "position": "D"}
                for name, goals in (defense or [])
            ],
        }

    return {
        "playerByGameStats": {
            "awayTeam": side(away_forwards, away_defense),
            "homeTeam": side(home_forwards, home_defense),
        }
    }


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class TestTimsBoards(unittest.TestCase):
    def test_union_keeps_players_dropped_by_later_waves(self):
        boards = {"days": {}}
        update_board_day(
            boards,
            "2026-11-22",
            {"2": [{"player": "Sebastian Aho", "team": "CAR"}]},
            "2026-11-22 12:11 PM ET",
        )
        update_board_day(
            boards,
            "2026-11-22",
            {"2": [{"player": "Tage Thompson", "team": "BUF"}]},
            "2026-11-22 05:00 PM ET",
        )
        players = {p["player"]: p for p in board_players_for(boards, "2026-11-22")}
        self.assertIn("Sebastian Aho", players)
        self.assertIn("Tage Thompson", players)
        self.assertEqual(players["Sebastian Aho"]["group"], "2")
        self.assertEqual(players["Sebastian Aho"]["first_seen_et"], "2026-11-22 12:11 PM ET")
        self.assertEqual(players["Tage Thompson"]["last_seen_et"], "2026-11-22 05:00 PM ET")

    def test_old_days_are_pruned(self):
        boards = {"days": {}}
        for day in range(1, 16):
            update_board_day(
                boards,
                f"2026-11-{day:02d}",
                {"1": [{"player": f"Player {day}", "team": "AAA"}]},
                f"2026-11-{day:02d} 12:00 PM ET",
                keep_days=14,
            )
        self.assertEqual(len(boards["days"]), 14)
        self.assertNotIn("2026-11-01", boards["days"])
        self.assertIn("2026-11-15", boards["days"])


class TestLiveScoreboard(unittest.TestCase):
    def _games(self):
        return [
            {
                "id": 101,
                "is_today": True,
                "away": "CAR",
                "home": "NYR",
                "start_time_utc": "2026-11-22T17:30:00Z",
                "start_time_et": "12:30 PM ET",
                "game_state": "FINAL",
            },
            {
                "id": 102,
                "is_today": True,
                "away": "BUF",
                "home": "TBL",
                "start_time_utc": "2026-11-22T22:00:00Z",
                "start_time_et": "5:00 PM ET",
                "game_state": "LIVE",
            },
        ]

    def _board(self):
        return [
            {"player": "Sebastian Aho", "team": "CAR", "group": "2"},
            {"player": "Eric Robinson", "team": "CAR", "group": "3"},
            {"player": "Sean Walker", "team": "CAR", "group": "3"},
            {"player": "Tage Thompson", "team": "BUF", "group": "1"},
            {"player": "Leon Draisaitl", "team": "EDM", "group": "1"},
        ]

    def test_only_board_players_are_listed_with_wave_and_goals(self):
        boxes = {
            101: _box(
                [("S. Aho", 2), ("E. Robinson", 1), ("J. Martinook", 1)],
                [("S. Walker", 1)],
                [("M. Zibanejad", 1)],
            ),
            102: _box([("T. Thompson", 1)], home_forwards=[("J. Guentzel", 1)]),
        }

        def fake_fetch(_session, url, **_kwargs):
            return _FakeResponse(boxes[int(url.split("/gamecenter/")[1].split("/")[0])])

        with mock.patch("src.live_scoreboard._fetch_with_retries", side_effect=fake_fetch):
            rows = build_tims_scoreboard(self._games(), self._board())

        names = [row["player"] for row in rows]
        self.assertEqual(names, ["Sebastian Aho", "Eric Robinson", "Sean Walker", "Tage Thompson"])
        by_name = {row["player"]: row for row in rows}
        self.assertEqual(by_name["Sebastian Aho"]["goals"], 2)
        self.assertEqual(by_name["Sebastian Aho"]["wave"], 1)
        self.assertEqual(by_name["Sebastian Aho"]["wave_et"], "12:30 PM ET")
        self.assertEqual(by_name["Sebastian Aho"]["group"], "2")
        self.assertEqual(by_name["Tage Thompson"]["wave"], 2)
        self.assertEqual(by_name["Tage Thompson"]["game_state"], "LIVE")
        self.assertNotIn("M. Zibanejad", names)

    def test_network_disabled_returns_empty(self):
        self.assertEqual(build_tims_scoreboard(self._games(), self._board(), allow_network=False), [])

    def test_missing_boxscore_is_skipped(self):
        def fake_fetch(_session, _url, **_kwargs):
            return None

        with mock.patch("src.live_scoreboard._fetch_with_retries", side_effect=fake_fetch):
            self.assertEqual(build_tims_scoreboard(self._games(), self._board()), [])


if __name__ == "__main__":
    unittest.main()
