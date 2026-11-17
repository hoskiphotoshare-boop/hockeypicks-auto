"""Stdlib-only gate deciding whether live inference is due (+3 min after NHL game starts or odds windows)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

ET_ZONE = ZoneInfo("America/New_York")

# Dual-cron schedule mappings for Eastern Prevailing Time (EPT).
# GitHub Actions schedule triggers run on UTC. During EDT (summer/fall), offset is -0400;
# during EST (winter/spring), offset is -0500. Dual crons ensure exact alignment across DST.
ODDS_CRONS_10PM_ET: dict[str, str] = {
    "0 2 * * *": "-0400",  # EDT: 22:00 EDT = 02:00 UTC (day before / overnight)
    "0 3 * * *": "-0500",  # EST: 22:00 EST = 03:00 UTC (day before / overnight)
}

ODDS_CRONS_430PM_ET: dict[str, str] = {
    "30 20 * * *": "-0400",  # EDT: 16:30 EDT = 20:30 UTC
    "30 21 * * *": "-0500",  # EST: 16:30 EST = 21:30 UTC
}

ODDS_CRONS_11AM_ET: dict[str, str] = {
    "0 15 * * *": "-0400",  # EDT: 11:00 EDT = 15:00 UTC
    "0 16 * * *": "-0500",  # EST: 11:00 EST = 16:00 UTC
}

ODDS_CRONS_12PM_ET: dict[str, str] = {
    "0 16 * * *": "-0400",  # EDT: 12:00 EDT = 16:00 UTC
    "0 17 * * *": "-0500",  # EST: 12:00 EST = 17:00 UTC
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-based) of weekday (0=Mon, 6=Sun) in year/month."""
    first_day = date(year, month, 1)
    first_occurrence = 1 + (weekday - first_day.weekday()) % 7
    return date(year, month, first_occurrence + (n - 1) * 7)


def _last_weekday_on_or_before(year: int, month: int, day: int, target_weekday: int) -> date:
    """Return the last occurrence of target_weekday on or before year/month/day."""
    ref = date(year, month, day)
    diff = (ref.weekday() - target_weekday) % 7
    return ref - timedelta(days=diff)


def _easter_sunday(year: int) -> date:
    """Calculate Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_us_canada_holiday(target_date: date) -> tuple[bool, str]:
    """Check if target_date is a recognized US or Canadian holiday associated with NHL matinees."""
    y = target_date.year
    m = target_date.month
    d = target_date.day

    # Fixed-date national holidays
    fixed = {
        (1, 1): "New Year's Day",
        (7, 1): "Canada Day",
        (7, 4): "Independence Day (US)",
        (9, 30): "National Day for Truth and Reconciliation (Canada)",
        (11, 11): "Veterans Day / Remembrance Day",
        (12, 24): "Christmas Eve",
        (12, 25): "Christmas Day",
        (12, 26): "Boxing Day (Canada)",
        (12, 31): "New Year's Eve",
    }
    if (m, d) in fixed:
        return True, fixed[(m, d)]

    # Floating holidays
    # MLK Day: 3rd Monday in January
    if m == 1 and target_date == _nth_weekday(y, 1, 0, 3):
        return True, "Martin Luther King Jr. Day"

    # Presidents' Day (US) / Family Day (Canada): 3rd Monday in February
    if m == 2 and target_date == _nth_weekday(y, 2, 0, 3):
        return True, "Presidents' Day / Family Day"

    # Easter, Good Friday, Easter Monday
    easter = _easter_sunday(y)
    if target_date == easter - timedelta(days=2):
        return True, "Good Friday"
    if target_date == easter + timedelta(days=1):
        return True, "Easter Monday"

    # Victoria Day (Canada): Monday on or before May 24
    if m == 5 and target_date == _last_weekday_on_or_before(y, 5, 24, 0):
        return True, "Victoria Day (Canada)"

    # Memorial Day (US): Last Monday in May
    if m == 5 and target_date == _last_weekday_on_or_before(y, 5, 31, 0):
        return True, "Memorial Day (US)"

    # Civic Holiday (Canada): 1st Monday in August
    if m == 8 and target_date == _nth_weekday(y, 8, 0, 1):
        return True, "Civic Holiday (Canada)"

    # Labor Day / Labour Day (US & Canada): 1st Monday in September
    if m == 9 and target_date == _nth_weekday(y, 9, 0, 1):
        return True, "Labor Day"

    # Canadian Thanksgiving / Columbus Day: 2nd Monday in October
    if m == 10 and target_date == _nth_weekday(y, 10, 0, 2):
        return True, "Canadian Thanksgiving / Columbus Day"

    # US Thanksgiving: 4th Thursday in November & Black Friday (day after)
    if m == 11:
        us_thanksgiving = _nth_weekday(y, 11, 3, 4)
        if target_date == us_thanksgiving:
            return True, "US Thanksgiving"
        if target_date == us_thanksgiving + timedelta(days=1):
            return True, "Day after Thanksgiving / Black Friday"

    return False, ""


def has_early_matinee_game(payload: dict[str, Any], now_et: datetime) -> tuple[bool, str]:
    """Check if today's slate contains any games scheduled before 4:00 PM ET."""
    today_et_str = now_et.strftime("%Y-%m-%d")
    for game in payload.get("games") or []:
        if game.get("game_state") in ("FINAL", "OFF"):
            continue
        start_utc = parse_game_start_utc(game)
        if start_utc is not None:
            game_et = start_utc.astimezone(ET_ZONE)
            if game_et.strftime("%Y-%m-%d") == today_et_str and game_et.hour < 16:
                time_str = game_et.strftime("%-I:%M %p ET")
                return True, f"{game.get('away', '?')}@{game.get('home', '?')} at {time_str}"
        else:
            if str(game.get("date") or "").strip() == today_et_str:
                et_str = str(game.get("start_time_et") or "").strip()
                if et_str and et_str.upper() != "TBD":
                    if "AM" in et_str.upper():
                        return True, f"{game.get('away', '?')}@{game.get('home', '?')} at {et_str}"
                    for early_h in ("12:", "1:", "2:", "3:"):
                        if et_str.startswith(early_h) and "PM" in et_str.upper():
                            return True, f"{game.get('away', '?')}@{game.get('home', '?')} at {et_str}"
    return False, ""


def parse_game_start_utc(game: dict[str, Any]) -> datetime | None:
    """Resolve a game's scheduled puck-drop UTC datetime from start_time_utc or date + ET fallback."""
    dt = parse_utc_timestamp(game.get("start_time_utc"))
    if dt is not None:
        return dt

    date_str = str(game.get("date") or "").strip()
    et_str = str(game.get("start_time_et") or "").strip()
    if not date_str or not et_str or et_str.upper() == "TBD":
        return None
    et_clean = et_str[:-3].strip() if et_str.upper().endswith(" ET") else et_str
    try:
        naive = datetime.strptime(f"{date_str} {et_clean}", "%Y-%m-%d %I:%M %p")
        return naive.replace(tzinfo=ET_ZONE).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse ISO-8601 UTC timestamps or report_builder 'UTC' strings into aware UTC datetimes."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%d %H:%M UTC"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    try:
        iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def find_due_games(
    payload: dict[str, Any],
    *,
    now_utc: datetime | None = None,
    delay_minutes: int = 3,
    max_lag_minutes: int = 50,
) -> list[dict[str, Any]]:
    """Return scheduled games whose (start_time + delay_minutes) passed after the last run."""
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    meta = payload.get("meta") or {}
    last_updated = parse_utc_timestamp(meta.get("updated_utc"))
    max_lag = timedelta(minutes=max_lag_minutes)
    delay = timedelta(minutes=delay_minutes)

    due: list[dict[str, Any]] = []
    for game in payload.get("games") or []:
        if game.get("game_state") in ("FINAL", "OFF"):
            continue
        start_dt = parse_game_start_utc(game)
        if start_dt is None:
            continue
        trigger_dt = start_dt + delay
        if trigger_dt > now:
            continue
        if (now - trigger_dt) > max_lag:
            continue
        if last_updated is not None and last_updated >= trigger_dt:
            continue
        due.append(game)
    return due


def evaluate_gate(
    data_json_path: str,
    *,
    event_name: str = "schedule",
    cron: str | None = None,
    now_utc: datetime | None = None,
    delay_minutes: int = 3,
    max_lag_minutes: int = 50,
) -> tuple[bool, str]:
    """Evaluate whether scheduled odds release or post-puck-drop inference should proceed."""
    if event_name == "workflow_dispatch":
        return True, "Manual workflow_dispatch trigger."

    if not os.path.exists(data_json_path) or os.path.getsize(data_json_path) == 0:
        return False, f"No payload at {data_json_path}; skipping scheduled inference."

    try:
        with open(data_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return False, f"Failed to parse {data_json_path} ({exc}); skipping."

    if (payload.get("meta") or {}).get("status") == "bootstrap":
        return False, "Dashboard still in bootstrap state; waiting for initial training run."

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    now_et = now.astimezone(ET_ZONE)
    active_offset = now_et.strftime("%z")

    cron_clean = str(cron or "").strip()

    # 1) 10:00 PM EPT (day before): Daily opening odds release & next-day line movements
    if cron_clean in ODDS_CRONS_10PM_ET:
        expected_offset = ODDS_CRONS_10PM_ET[cron_clean]
        if active_offset != expected_offset:
            return (
                False,
                f"Seasonal offset mismatch for 10:00 PM EPT cron '{cron_clean}' "
                f"(active ET offset is {active_offset}, expected {expected_offset}).",
            )
        return True, f"10:00 PM EPT opening odds & next-day line movements window (offset {active_offset})."

    # 2) 4:30 PM EPT: Daily pre-game line movements & starting goalie confirmations
    if cron_clean in ODDS_CRONS_430PM_ET:
        expected_offset = ODDS_CRONS_430PM_ET[cron_clean]
        if active_offset != expected_offset:
            return (
                False,
                f"Seasonal offset mismatch for 4:30 PM EPT cron '{cron_clean}' "
                f"(active ET offset is {active_offset}, expected {expected_offset}).",
            )
        return True, f"4:30 PM EPT pre-game odds & line movements window (offset {active_offset})."

    # 3) 12:00 PM EPT: Daily noon odds release & mid-day line movements
    if cron_clean in ODDS_CRONS_12PM_ET and active_offset == ODDS_CRONS_12PM_ET[cron_clean]:
        return True, f"12:00 PM EPT daily noon odds & line movements window (offset {active_offset})."

    # 4) 11:00 AM EPT: Weekend & holiday early matinee odds release & line movements
    if cron_clean in ODDS_CRONS_11AM_ET and active_offset == ODDS_CRONS_11AM_ET[cron_clean]:
        # Check weekend (5 = Saturday, 6 = Sunday)
        weekday = now_et.weekday()
        if weekday == 5:
            return True, f"11:00 AM EPT weekend matinee odds window (Saturday, offset {active_offset})."
        if weekday == 6:
            return True, f"11:00 AM EPT weekend matinee odds window (Sunday, offset {active_offset})."

        # Check US / Canadian national holidays
        is_hol, hol_name = is_us_canada_holiday(now_et.date())
        if is_hol:
            return True, f"11:00 AM EPT holiday matinee odds window ({hol_name}, offset {active_offset})."

        # Check if today's slate has an early matinee game starting before 4:00 PM ET
        has_mat, mat_detail = has_early_matinee_game(payload, now_et)
        if has_mat:
            return True, f"11:00 AM EPT early matinee odds window ({mat_detail}, offset {active_offset})."

        day_name = now_et.strftime("%A")
        return (
            False,
            f"11:00 AM EPT matinee window skipped: regular weekday ({day_name}) with no early games scheduled before 4:00 PM ET.",
        )

    # 5) Handle seasonal offset mismatch for 11:00 AM or 12:00 PM crons that did not match active offset
    if cron_clean in ODDS_CRONS_12PM_ET or cron_clean in ODDS_CRONS_11AM_ET:
        target_label = (
            "12:00 PM EPT"
            if cron_clean == "0 17 * * *"
            else ("11:00 AM EPT" if cron_clean == "0 15 * * *" else "11:00 AM / 12:00 PM EPT")
        )
        return (
            False,
            f"Seasonal offset mismatch for {target_label} cron '{cron_clean}' "
            f"(active ET offset is {active_offset}).",
        )

    # 6) Post-puck-drop reactive schedule (+3 min after game start)
    due = find_due_games(
        payload,
        now_utc=now,
        delay_minutes=delay_minutes,
        max_lag_minutes=max_lag_minutes,
    )
    if not due:
        return False, "No scheduled games passed start_time + 3m since last update."

    labels = ", ".join(f"{g.get('away', '?')}@{g.get('home', '?')} ({g.get('start_time_et', '?')})" for g in due[:5])
    extra = f" (+{len(due) - 5} more)" if len(due) > 5 else ""
    return True, f"Post-start inference due for: {labels}{extra}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if post-puck-drop or scheduled odds NHL inference is due.")
    parser.add_argument("--data-json", default="docs/data.json", help="Path to exported dashboard JSON.")
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "schedule"))
    parser.add_argument("--cron", default=os.environ.get("GITHUB_SCHEDULE", ""), help="Triggering cron schedule string.")
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    parser.add_argument("--delay-minutes", type=int, default=3)
    parser.add_argument("--max-lag-minutes", type=int, default=50)
    args = parser.parse_args()

    should_run, reason = evaluate_gate(
        args.data_json,
        event_name=args.event_name,
        cron=args.cron,
        delay_minutes=args.delay_minutes,
        max_lag_minutes=args.max_lag_minutes,
    )
    print(f"Infer Gate -> should_run={str(should_run).lower()} | {reason}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as out:
            out.write(f"should_run={'true' if should_run else 'false'}\n")
            out.write(f"trigger_reason={reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
