"""Unit tests for the live-data heartbeat."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.health_check import check_freshness, load_payload


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


class TestHealthCheck(unittest.TestCase):
    def test_overdue_game_alerts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC"))
            healthy, reason = check_freshness(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
            )
            self.assertFalse(healthy)
            self.assertIn("MTL@NYR", reason)

    def test_update_after_trigger_is_healthy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-18 00:10:00 UTC"))
            healthy, _ = check_freshness(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
            )
            self.assertTrue(healthy)

    def test_grace_period_suppresses_early_alert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC"))
            healthy, _ = check_freshness(
                path,
                now_utc=datetime(2026, 11, 18, 0, 15, tzinfo=timezone.utc),
                grace_minutes=20,
            )
            self.assertTrue(healthy)

    def test_final_game_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC", state="FINAL"))
            healthy, _ = check_freshness(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
            )
            self.assertTrue(healthy)

    def test_future_game_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write(tmpdir, _payload("2026-11-17 20:00:00 UTC", start_utc="2026-11-18T02:00:00Z"))
            healthy, _ = check_freshness(
                path,
                now_utc=datetime(2026, 11, 18, 0, 30, tzinfo=timezone.utc),
            )
            self.assertTrue(healthy)

    def test_nan_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"meta": {"status": "live", "updated_utc": "2026-11-17 20:00:00 UTC"}, "x": NaN}')
            with self.assertRaises(ValueError):
                load_payload(path)


if __name__ == "__main__":
    unittest.main()
