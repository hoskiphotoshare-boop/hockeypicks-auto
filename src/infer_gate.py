"""Stdlib-only gate deciding whether live inference is due (+3 min after NHL game starts)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

ET_ZONE = ZoneInfo("America/New_York")


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
    now_utc: datetime | None = None,
    delay_minutes: int = 3,
    max_lag_minutes: int = 50,
) -> tuple[bool, str]:
    """Evaluate whether scheduled post-game-start inference should proceed."""
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

    due = find_due_games(
        payload,
        now_utc=now_utc,
        delay_minutes=delay_minutes,
        max_lag_minutes=max_lag_minutes,
    )
    if not due:
        return False, "No scheduled games passed start_time + 3m since last update."

    labels = ", ".join(f"{g.get('away', '?')}@{g.get('home', '?')} ({g.get('start_time_et', '?')})" for g in due[:5])
    extra = f" (+{len(due) - 5} more)" if len(due) > 5 else ""
    return True, f"Post-start inference due for: {labels}{extra}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if post-puck-drop NHL inference is due.")
    parser.add_argument("--data-json", default="docs/data.json", help="Path to exported dashboard JSON.")
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "schedule"))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    parser.add_argument("--delay-minutes", type=int, default=3)
    parser.add_argument("--max-lag-minutes", type=int, default=50)
    args = parser.parse_args()

    should_run, reason = evaluate_gate(
        args.data_json,
        event_name=args.event_name,
        delay_minutes=args.delay_minutes,
        max_lag_minutes=args.max_lag_minutes,
    )
    print(f"Infer Gate -> should_run={str(should_run).lower()} | {reason}")

    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as out:
            out.write(f"should_run={'true' if should_run else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
