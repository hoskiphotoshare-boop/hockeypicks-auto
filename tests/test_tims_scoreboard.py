"""Tests for the persisted Tims board union and the live Tims scoreboard."""

from __future__ import annotations

import unittest
from unittest import mock

from src.live_scoreboard import build_tims_scoreboard, choose_scoreboard_slate
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

    def test_date_str_selects_games_that_are_not_is_today(self):
        previous = [
            {
                "id": 201,
                "date": "2026-11-21",
                "is_today": False,
                "away": "CAR",
                "home": "NYR",
                "start_time_utc": "2026-11-21T17:30:00Z",
                "start_time_et": "12:30 PM ET",
                "game_state": "FINAL",
            }
        ]
        today = [
            {
                "id": 301,
                "date": "2026-11-22",
                "is_today": True,
                "away": "BUF",
                "home": "TBL",
                "start_time_utc": "2026-11-22T22:00:00Z",
                "start_time_et": "5:00 PM ET",
                "game_state": "FUT",
            }
        ]
        boxes = {
            201: _box([("S. Aho", 2)]),
            301: _box([("T. Thompson", 1)]),
        }

        def fake_fetch(_session, url, **_kwargs):
            return _FakeResponse(boxes[int(url.split("/gamecenter/")[1].split("/")[0])])

        with mock.patch("src.live_scoreboard._fetch_with_retries", side_effect=fake_fetch):
            rows = build_tims_scoreboard(
                previous + today, self._board(), date_str="2026-11-21"
            )

        self.assertEqual([row["player"] for row in rows], ["Sebastian Aho"])
        self.assertEqual(rows[0]["game_state"], "FINAL")
        self.assertEqual(rows[0]["wave_et"], "12:30 PM ET")


class TestChooseScoreboardSlate(unittest.TestCase):
    @staticmethod
    def _game(state: str, date: str) -> dict:
        return {"id": 1, "date": date, "game_state": state}

    def test_stays_on_today_once_any_today_game_started(self):
        slate, previous = choose_scoreboard_slate(
            [self._game("FUT", "2026-11-22"), self._game("LIVE", "2026-11-22")],
            [self._game("FINAL", "2026-11-21")],
        )
        self.assertFalse(previous)
        self.assertEqual([g["game_state"] for g in slate], ["FUT", "LIVE"])

    def test_carries_previous_while_today_has_not_started(self):
        previous_games = [self._game("FINAL", "2026-11-21")]
        slate, previous = choose_scoreboard_slate(
            [self._game("FUT", "2026-11-22")], previous_games
        )
        self.assertTrue(previous)
        self.assertEqual(slate, previous_games)

    def test_carries_previous_when_today_has_no_games(self):
        previous_games = [self._game("OFF", "2026-11-21")]
        slate, previous = choose_scoreboard_slate([], previous_games)
        self.assertTrue(previous)
        self.assertEqual(slate, previous_games)

    def test_today_when_previous_has_no_games(self):
        today = [self._game("FUT", "2026-11-22")]
        slate, previous = choose_scoreboard_slate(today, [])
        self.assertFalse(previous)
        self.assertEqual(slate, today)

    def test_today_when_today_game_already_final(self):
        today = [self._game("FINAL", "2026-11-22")]
        slate, previous = choose_scoreboard_slate(
            today, [self._game("FINAL", "2026-11-21")]
        )
        self.assertFalse(previous)
        self.assertEqual(slate, today)

    def test_treats_pregame_and_unknown_states_as_not_started(self):
        slate, previous = choose_scoreboard_slate(
            [self._game("PRE", "2026-11-22"), self._game("", "2026-11-22")],
            [self._game("FINAL", "2026-11-21")],
        )
        self.assertTrue(previous)
        self.assertEqual([g["game_state"] for g in slate], ["FINAL"])

    def test_handles_none_inputs(self):
        self.assertEqual(choose_scoreboard_slate(None, None), ([], False))


if __name__ == "__main__":
    unittest.main()
