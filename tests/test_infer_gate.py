"""Unit tests for NHL infer gate and scheduled odds / matinee windows."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone

from src.infer_gate import (
    evaluate_gate,
    find_due_games,
    has_early_matinee_game,
    is_us_canada_holiday,
    parse_game_start_utc,
)


class TestInferGate(unittest.TestCase):
    def test_holiday_helper_coverage(self):
        # Fixed national holidays
        self.assertEqual(is_us_canada_holiday(date(2026, 1, 1)), (True, "New Year's Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 7, 1)), (True, "Canada Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 7, 4)), (True, "Independence Day (US)"))
        self.assertEqual(is_us_canada_holiday(date(2026, 11, 11)), (True, "Veterans Day / Remembrance Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 12, 25)), (True, "Christmas Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 12, 26)), (True, "Boxing Day (Canada)"))

        # Floating Holidays
        self.assertEqual(is_us_canada_holiday(date(2026, 1, 19)), (True, "Martin Luther King Jr. Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 2, 16)), (True, "Presidents' Day / Family Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 4, 3)), (True, "Good Friday"))
        self.assertEqual(is_us_canada_holiday(date(2026, 4, 6)), (True, "Easter Monday"))
        self.assertEqual(is_us_canada_holiday(date(2026, 5, 18)), (True, "Victoria Day (Canada)"))
        self.assertEqual(is_us_canada_holiday(date(2026, 5, 25)), (True, "Memorial Day (US)"))
        self.assertEqual(is_us_canada_holiday(date(2026, 8, 3)), (True, "Civic Holiday (Canada)"))
        self.assertEqual(is_us_canada_holiday(date(2026, 9, 7)), (True, "Labor Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 10, 12)), (True, "Canadian Thanksgiving / Columbus Day"))
        self.assertEqual(is_us_canada_holiday(date(2026, 11, 26)), (True, "US Thanksgiving"))
        self.assertEqual(is_us_canada_holiday(date(2026, 11, 27)), (True, "Day after Thanksgiving / Black Friday"))
        self.assertEqual(is_us_canada_holiday(date(2026, 11, 18)), (False, ""))

    def test_infer_gate_odds_windows_and_dst(self):
        evening_payload = {
            "meta": {"status": "live", "updated_utc": "2026-11-17 12:00:00 UTC"},
            "games": [
                {
                    "away": "MTL",
                    "home": "NYR",
                    "date": "2026-11-17",
                    "start_time_et": "7:00 PM ET",
                    "start_time_utc": "2026-11-18T00:00:00Z",
                    "game_state": "FUT",
                }
            ],
        }
        matinee_payload = {
            "meta": {"status": "live", "updated_utc": "2026-11-17 12:00:00 UTC"},
            "games": [
                {
                    "away": "BOS",
                    "home": "DET",
                    "date": "2026-11-17",
                    "start_time_et": "1:00 PM ET",
                    "start_time_utc": "2026-11-17T18:00:00Z",
                    "game_state": "FUT",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            evening_path = os.path.join(tmpdir, "evening.json")
            matinee_path = os.path.join(tmpdir, "matinee.json")
            with open(evening_path, "w", encoding="utf-8") as f:
                json.dump(evening_payload, f)
            with open(matinee_path, "w", encoding="utf-8") as f:
                json.dump(matinee_payload, f)

            # --- Fixed ET odds slots: one cron works in both EST and EDT ---
            # The workflow sends `timezone: America/New_York`, so the gate sees the
            # same wall-clock cron all year and GitHub handles the UTC conversion.
            slots = [
                (
                    "0 22 * * *",
                    datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc),   # 10:00 PM EST
                    datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc),    # 10:00 PM EDT
                    "10:00 PM EPT",
                ),
                (
                    "30 16 * * *",
                    datetime(2026, 11, 18, 21, 30, tzinfo=timezone.utc),  # 4:30 PM EST
                    datetime(2026, 6, 15, 20, 30, tzinfo=timezone.utc),   # 4:30 PM EDT
                    "4:30 PM EPT",
                ),
                (
                    "0 12 * * *",
                    datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc),   # 12:00 PM EST
                    datetime(2026, 6, 15, 16, 0, tzinfo=timezone.utc),    # 12:00 PM EDT
                    "12:00 PM EPT",
                ),
            ]
            for cron, est_now, edt_now, label in slots:
                for now in (est_now, edt_now):
                    ok, reason = evaluate_gate(evening_path, cron=cron, now_utc=now)
                    self.assertTrue(ok, msg=f"{label} cron {cron} at {now}: {reason}")
                    self.assertIn(label, reason)
                    self.assertNotIn("Seasonal offset mismatch", reason)

            # An unrecognised non-empty schedule still runs: hockeypicks_infer.yml
            # only contains fixed odds slots now, so skipping would lose a release.
            fallback_ok, fallback_reason = evaluate_gate(
                evening_path,
                cron="0 3 * * *",
                now_utc=datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(fallback_ok)
            self.assertIn("Scheduled odds window", fallback_reason)

            # --- 11:00 AM ET Weekend & Holiday / Early Matinee Windows ---
            # Weekend (Saturday 2026-11-21 in EST, 11:00 AM ET)
            ok_sat, reason_sat = evaluate_gate(
                evening_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 11, 21, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_sat)
            self.assertIn("weekend matinee", reason_sat)
            self.assertIn("Saturday", reason_sat)

            # Weekend (Sunday 2026-11-22 in EST, 11:00 AM ET)
            ok_sun, reason_sun = evaluate_gate(
                evening_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_sun)
            self.assertIn("Sunday", reason_sun)

            # Holiday (US Thanksgiving Thursday 2026-11-26 in EST, 11:00 AM ET)
            ok_hol, reason_hol = evaluate_gate(
                evening_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_hol)
            self.assertIn("holiday matinee", reason_hol)
            self.assertIn("US Thanksgiving", reason_hol)

            # Weekday with early matinee game (Tuesday 2026-11-17, 11:00 AM ET)
            ok_mat, reason_mat = evaluate_gate(
                matinee_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_mat)
            self.assertIn("early matinee", reason_mat)
            self.assertIn("BOS@DET", reason_mat)

            # Weekday without early game (Tuesday 2026-11-17, evening only games)
            skip_wd, reason_wd = evaluate_gate(
                evening_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_wd)
            self.assertIn("regular weekday (Tuesday) with no early games", reason_wd)

            # EDT 11:00 AM (Saturday 2026-06-20, 15:00 UTC)
            ok_edt_mat, _ = evaluate_gate(
                evening_path,
                cron="0 11 * * *",
                now_utc=datetime(2026, 6, 20, 15, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_edt_mat)

    def test_infer_gate_post_start_window(self):
        payload = {
            "meta": {
                "status": "live",
                "updated_utc": "2026-11-16 21:18:00 UTC",
            },
            "games": [
                {
                    "away": "MTL",
                    "home": "NYR",
                    "date": "2026-11-16",
                    "start_time_et": "7:30 PM ET",
                    "start_time_utc": "2026-11-17T00:30:00Z",
                    "game_state": "FUT",
                },
                {
                    "away": "VAN",
                    "home": "UTA",
                    "date": "2026-11-16",
                    "start_time_et": "9:30 PM ET",
                    "game_state": "FUT",
                },
            ],
        }
        fallback_dt = parse_game_start_utc(payload["games"][1])
        self.assertEqual(fallback_dt, datetime(2026, 11, 17, 2, 30, tzinfo=timezone.utc))

        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 32, tzinfo=timezone.utc)),
            [],
        )
        due = find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 33, tzinfo=timezone.utc))
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["away"], "MTL")

        payload["meta"]["updated_utc"] = "2026-11-17 00:34:10 UTC"
        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc)),
            [],
        )
        due2 = find_due_games(payload, now_utc=datetime(2026, 11, 17, 2, 33, tzinfo=timezone.utc))
        self.assertEqual(len(due2), 1)
        self.assertEqual(due2[0]["away"], "VAN")

        # A late trigger still catches the game while it is in progress (180m window)
        due_late = find_due_games(payload, now_utc=datetime(2026, 11, 17, 3, 25, tzinfo=timezone.utc))
        self.assertEqual(len(due_late), 1)
        self.assertEqual(due_late[0]["away"], "VAN")

        # After the full-game window, it stops re-triggering
        self.assertEqual(
            find_due_games(payload, now_utc=datetime(2026, 11, 17, 5, 34, tzinfo=timezone.utc)),
            [],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "data.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            ok, reason = evaluate_gate(
                json_path,
                event_name="schedule",
                now_utc=datetime(2026, 11, 17, 2, 35, tzinfo=timezone.utc),
            )
            self.assertTrue(ok)
            self.assertIn("VAN@UTA", reason)

            skip, _ = evaluate_gate(
                json_path,
                event_name="schedule",
                now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc),
            )
            self.assertFalse(skip)

            manual_ok, _ = evaluate_gate(json_path, event_name="workflow_dispatch")
            self.assertTrue(manual_ok)

            # The live loop evaluates post-start games with an empty cron.
            cron_ok, cron_reason = evaluate_gate(
                json_path,
                event_name="schedule",
                cron="",
                now_utc=datetime(2026, 11, 17, 2, 35, tzinfo=timezone.utc),
            )
            self.assertTrue(cron_ok)
            self.assertIn("VAN@UTA", cron_reason)

            cron_skip, _ = evaluate_gate(
                json_path,
                event_name="schedule",
                cron="",
                now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc),
            )
            self.assertFalse(cron_skip)


if __name__ == "__main__":
    unittest.main()
