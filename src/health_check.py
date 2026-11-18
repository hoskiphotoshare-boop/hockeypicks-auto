"""Detect games that started without a dashboard refresh, and reject broken JSON.

This is the heartbeat for the live-inference path. Instead of alerting on a blind
timer ("data is older than N minutes"), it reuses the gate's due-game logic: a game
that started 20+ minutes ago and has not been followed by an update means the
scheduler, the loop, or a manual run failed while it mattered.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infer_gate import DEFAULT_MAX_LAG_MINUTES, ET_ZONE, find_due_games, parse_game_start_utc

DEFAULT_GRACE_MINUTES = 20


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def load_payload(path: str) -> dict[str, Any]:
    """Load dashboard JSON and reject NaN/Infinity tokens that browsers cannot parse."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.loads(handle.read(), parse_constant=_reject_constant)


def check_freshness(
    data_json: str,
    *,
    now_utc: datetime | None = None,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
    max_lag_minutes: int = DEFAULT_MAX_LAG_MINUTES,
    delay_minutes: int = 3,
) -> tuple[bool, str]:
    """Return (healthy, reason).

    A game is overdue when its start + delay has passed, the dashboard has not been
    updated since that trigger, and at least ``grace_minutes`` have elapsed so a run
    that is already in flight does not trip the alert.
    """
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    payload = load_payload(data_json)
    meta = payload.get("meta") or {}
    if meta.get("status") == "bootstrap":
        return True, "Dashboard is still in bootstrap state."

    due = find_due_games(
        payload,
        now_utc=now,
        delay_minutes=delay_minutes,
        max_lag_minutes=max_lag_minutes,
    )

    grace = timedelta(minutes=grace_minutes)
    overdue: list[dict[str, Any]] = []
    for game in due:
        start = parse_game_start_utc(game)
        if start is None:
            continue
        trigger = start + timedelta(minutes=delay_minutes)
        if now - trigger >= grace:
            overdue.append(game)

    if not overdue:
        return True, "OK - no game is waiting for an update."

    labels = ", ".join(
        f"{game.get('away', '?')}@{game.get('home', '?')} ({game.get('start_time_et', '?')})"
        for game in overdue[:5]
    )
    extra = f" (+{len(overdue) - 5} more)" if len(overdue) > 5 else ""
    return (
        False,
        f"{len(overdue)} game(s) started without a data update: {labels}{extra}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that live games have fresh dashboard data.")
    parser.add_argument("--data-json", default="docs/data.json")
    parser.add_argument("--grace-minutes", type=int, default=DEFAULT_GRACE_MINUTES)
    parser.add_argument("--max-lag-minutes", type=int, default=DEFAULT_MAX_LAG_MINUTES)
    parser.add_argument("--now", default="", help="ISO-8601 override for tests.")
    args = parser.parse_args(argv)

    now = None
    if args.now:
        from src.live_loop import parse_now

        now = parse_now(args.now)

    try:
        healthy, reason = check_freshness(
            args.data_json,
            now_utc=now,
            grace_minutes=args.grace_minutes,
            max_lag_minutes=args.max_lag_minutes,
        )
    except Exception as exc:  # noqa: BLE001 - any read/parse failure is a real alert
        print(f"Health Check -> STALE | payload error: {exc}")
        return 1

    stamp = (now or datetime.now(timezone.utc)).astimezone(ET_ZONE).strftime("%Y-%m-%d %I:%M %p ET")
    print(f"[{stamp}] Health Check -> {'OK' if healthy else 'STALE'} | {reason}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
