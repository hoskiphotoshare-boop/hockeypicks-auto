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

            # --- 10:00 PM EPT (day before) Opening Odds Release ---
            # EST (Nov 18, 2026): 03:00 UTC should match (-0500); 02:00 UTC should skip
            ok, reason = evaluate_gate(
                evening_path,
                cron="0 3 * * *",
                now_utc=datetime(2026, 11, 18, 3, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok)
            self.assertIn("10:00 PM EPT", reason)
            self.assertIn("-0500", reason)

            skip, skip_reason = evaluate_gate(
                evening_path,
                cron="0 2 * * *",
                now_utc=datetime(2026, 11, 18, 2, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip)
            self.assertIn("Seasonal offset mismatch", skip_reason)

            # EDT (June 15, 2026): 02:00 UTC should match (-0400); 03:00 UTC should skip
            ok_edt, reason_edt = evaluate_gate(
                evening_path,
                cron="0 2 * * *",
                now_utc=datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_edt)
            self.assertIn("10:00 PM EPT", reason_edt)
            self.assertIn("-0400", reason_edt)

            skip_edt, _ = evaluate_gate(
                evening_path,
                cron="0 3 * * *",
                now_utc=datetime(2026, 6, 15, 3, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_edt)

            # --- 4:30 PM EPT Pre-Game Line Movements ---
            # EST (Nov 18, 2026): 21:30 UTC should match; 20:30 UTC should skip
            ok_430, reason_430 = evaluate_gate(
                evening_path,
                cron="30 21 * * *",
                now_utc=datetime(2026, 11, 18, 21, 30, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_430)
            self.assertIn("4:30 PM EPT", reason_430)

            skip_430, _ = evaluate_gate(
                evening_path,
                cron="30 20 * * *",
                now_utc=datetime(2026, 11, 18, 20, 30, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_430)

            # EDT (June 15, 2026): 20:30 UTC should match; 21:30 UTC should skip
            ok_430_edt, _ = evaluate_gate(
                evening_path,
                cron="30 20 * * *",
                now_utc=datetime(2026, 6, 15, 20, 30, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_430_edt)

            skip_430_edt, _ = evaluate_gate(
                evening_path,
                cron="30 21 * * *",
                now_utc=datetime(2026, 6, 15, 21, 30, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_430_edt)

            # --- 12:00 PM EPT Daily Noon Odds & Line Movements ---
            # EST (Nov 18, 2026): 17:00 UTC should match (-0500)
            ok_12, reason_12 = evaluate_gate(
                evening_path,
                cron="0 17 * * *",
                now_utc=datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_12)
            self.assertIn("12:00 PM EPT", reason_12)
            self.assertIn("-0500", reason_12)

            # EDT (June 15, 2026): 16:00 UTC should match (-0400)
            ok_12_edt, reason_12_edt = evaluate_gate(
                evening_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 6, 15, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_12_edt)
            self.assertIn("12:00 PM EPT", reason_12_edt)
            self.assertIn("-0400", reason_12_edt)

            # Seasonal mismatch for 12:00 PM (0 17 * * * during summer EDT)
            skip_12_mismatch, reason_12_mismatch = evaluate_gate(
                evening_path,
                cron="0 17 * * *",
                now_utc=datetime(2026, 6, 15, 17, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_12_mismatch)
            self.assertIn("Seasonal offset mismatch", reason_12_mismatch)

            # --- 11:00 AM EPT Weekend & Holiday / Early Matinee Windows ---
            # Weekend (Saturday 2026-11-21 in EST, 16:00 UTC)
            ok_sat, reason_sat = evaluate_gate(
                evening_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 11, 21, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_sat)
            self.assertIn("weekend matinee", reason_sat)
            self.assertIn("Saturday", reason_sat)

            # Weekend (Sunday 2026-11-22 in EST, 16:00 UTC)
            ok_sun, reason_sun = evaluate_gate(
                evening_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 11, 22, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_sun)
            self.assertIn("Sunday", reason_sun)

            # Holiday (US Thanksgiving Thursday 2026-11-26 in EST, 16:00 UTC)
            ok_hol, reason_hol = evaluate_gate(
                evening_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 11, 26, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_hol)
            self.assertIn("holiday matinee", reason_hol)
            self.assertIn("US Thanksgiving", reason_hol)

            # Weekday with early matinee game (Tuesday 2026-11-17, 16:00 UTC)
            ok_mat, reason_mat = evaluate_gate(
                matinee_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_mat)
            self.assertIn("early matinee", reason_mat)
            self.assertIn("BOS@DET", reason_mat)

            # Weekday without early game (Tuesday 2026-11-17, evening only games)
            skip_wd, reason_wd = evaluate_gate(
                evening_path,
                cron="0 16 * * *",
                now_utc=datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_wd)
            self.assertIn("regular weekday (Tuesday) with no early games", reason_wd)

            # EDT 11:00 AM (0 15 * * * during summer EDT Saturday 2026-06-20)
            ok_edt_mat, _ = evaluate_gate(
                evening_path,
                cron="0 15 * * *",
                now_utc=datetime(2026, 6, 20, 15, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(ok_edt_mat)

            # Seasonal mismatch for 11:00 AM (0 15 * * * during winter EST)
            skip_11_mismatch, _ = evaluate_gate(
                evening_path,
                cron="0 15 * * *",
                now_utc=datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc),
            )
            self.assertFalse(skip_11_mismatch)

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

            # Reactive puck-drop cron string passes through cleanly
            cron_ok, cron_reason = evaluate_gate(
                json_path,
                event_name="schedule",
                cron="3,18,33,48 15-23,0-4 * * *",
                now_utc=datetime(2026, 11, 17, 2, 35, tzinfo=timezone.utc),
            )
            self.assertTrue(cron_ok)
            self.assertIn("VAN@UTA", cron_reason)

            cron_skip, _ = evaluate_gate(
                json_path,
                event_name="schedule",
                cron="3,18,33,48 15-23,0-4 * * *",
                now_utc=datetime(2026, 11, 17, 0, 48, tzinfo=timezone.utc),
            )
            self.assertFalse(cron_skip)


if __name__ == "__main__":
    unittest.main()
