"""Unit tests for the live game-window loop."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.live_loop import parse_now, resolve_window_end, run_iteration, window_has_games

ET = ZoneInfo("America/New_York")


def _payload(updated_utc: str, start_utc: str = "2026-11-18T00:00:00Z", state: str = "LIVE") -> dict:
    return {
        "meta": {"status": "live", "updated_utc": updated_utc},
        "games": [
            {
                "away": "MTL",
                "home": "NYR",
                "date": "2026-11-17",
                "start_time_et": "7:00 PM ET",
                "start_time_utc": start_utc,
                "game_state": state,
            }
        ],
    }


def _write(tmpdir: str, payload: dict) -> str:
    path = os.path.join(tmpdir, "data.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


class TestLiveLoop(unittest.TestCase):
    def test_matinee_window_est(self):
        now = datetime(2026, 11, 17, 12, 30, tzinfo=ET)
        end = resolve_window_end(now)
        expected = datetime(2026, 11, 17, 16, 15, tzinfo=ET).astimezone(timezone.utc)
        self.assertEqual(end, expected)
        self.assertEqual(end.strftime("%H:%M"), "21:15")

    def test_matinee_window_edt(self):
        now = datetime(2026, 6, 15, 12, 30, tzinfo=ET)
        end = resolve_window_end(now)
        expected = datetime(2026, 6, 15, 16, 15, tzinfo=ET).astimezone(timezone.utc)
        self.assertEqual(end, expected)
        self.assertEqual(end.strftime("%H:%M"), "20:15")

    def test_evening_windows_across_dst(self):
        est_end = resolve_window_end(datetime(2026, 11, 17, 19, 0, tzinfo=ET))
        self.assertEqual(
            est_end,
            datetime(2026, 11, 17, 21, 45, tzinfo=ET).astimezone(timezone.utc),
        )
        self.assertEqual(est_end.strftime("%Y-%m-%d %H:%M"), "2026-11-18 02:45")

        edt_end = resolve_window_end(datetime(2026, 6, 15, 19, 0, tzinfo=ET))
        self.assertEqual(
            edt_end,
            datetime(2026, 6, 15, 21, 45, tzinfo=ET).astimezone(timezone.utc),
        )
        self.assertEqual(edt_end.strftime("%Y-%m-%d %H:%M"), "2026-06-16 01:45")

    def test_late_window_across_dst(self):
        est_end = resolve_window_end(datetime(2026, 11, 17, 22, 30, tzinfo=ET))
        self.assertEqual(
            est_end,
            datetime(2026, 11, 17, 23, 40, tzinfo=ET).astimezone(timezone.utc),
        )
        self.assertEqual(est_end.strftime("%Y-%m-%d %H:%M"), "2026-11-18 04:40")

        edt_end = resolve_window_end(datetime(2026, 6, 15, 22, 30, tzinfo=ET))
        self.assertEqual(
            edt_end,
            datetime(2026, 6, 15, 23, 40, tzinfo=ET).astimezone(timezone.utc),
        )
        self.assertEqual(edt_end.strftime("%Y-%m-%d %H:%M"), "2026-06-16 03:40")

    def test_outside_windows(self):
        for hour, minute in ((12, 14), (16, 15), (18, 39), (21, 45), (22, 14), (23, 40), (2, 0)):
            now = datetime(2026, 11, 17, hour, minute, tzinfo=ET)
            self.assertIsNone(resolve_window_end(now), msg=f"{hour:02d}:{minute:02d} should be closed")

    def test_parse_now_accepts_z(self):
        parsed = parse_now("2026-11-18T02:41:56Z")
        self.assertEqual(parsed, datetime(2026, 11, 18, 2, 41, 56, tzinfo=timezone.utc))

    def test_window_has_games(self):
        end = datetime(2026, 11, 17, 21, 15, tzinfo=timezone.utc)  # 4:15 PM EST
        evening = {"games": [{"start_time_utc": "2026-11-18T00:00:00Z", "game_state": "FUT"}]}
        self.assertFalse(window_has_games(evening, end))

        matinee = {"games": [{"start_time_utc": "2026-11-17T17:00:00Z", "game_state": "FUT"}]}
        self.assertTrue(window_has_games(matinee, end))

        final = {"games": [{"start_time_utc": "2026-11-17T17:00:00Z", "game_state": "FINAL"}]}
        self.assertFalse(window_has_games(final, end))

        # Unknown start times stay conservative so a data gap cannot silently disable the loop.
        unknown = {"games": [{"date": "2026-11-17", "start_time_et": "TBD", "game_state": "FUT"}]}
        self.assertTrue(window_has_games(unknown, end))

        # Games from another ET date are ignored (stale payload).
        yesterday = {"games": [{"start_time_utc": "2026-11-17T00:00:00Z", "game_state": "FUT"}]}
        self.assertFalse(window_has_games(yesterday, end))

    def test_run_iteration_dry_run_when_due(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC"))
            calls: list[str] = []
            due = run_iteration(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
                dry_run=True,
                inference_runner=lambda: calls.append("infer") or 0,
                committer=lambda: calls.append("commit"),
            )
            self.assertTrue(due)
            self.assertEqual(calls, [])

    def test_run_iteration_runs_and_commits_when_due(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC"))
            calls: list[str] = []
            due = run_iteration(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
                inference_runner=lambda: calls.append("infer") or 0,
                committer=lambda: calls.append("commit"),
            )
            self.assertTrue(due)
            self.assertEqual(calls, ["infer", "commit"])

    def test_run_iteration_skips_after_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-18 00:10:00 UTC"))
            due = run_iteration(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
                inference_runner=lambda: 0,
                committer=lambda: None,
            )
            self.assertFalse(due)

    def test_run_iteration_skips_commit_when_inference_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC"))
            calls: list[str] = []
            due = run_iteration(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
                inference_runner=lambda: calls.append("infer") or 1,
                committer=lambda: calls.append("commit"),
            )
            self.assertTrue(due)
            self.assertEqual(calls, ["infer"])


if __name__ == "__main__":
    unittest.main()
