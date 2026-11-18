"""Detect games that need a dashboard refresh, and reject broken JSON.

This is the heartbeat for the live-inference path. It stays game-aware instead of
alerting on a blind timer:

1. A game's start + delay has passed, the payload predates that trigger, and the
   grace period has elapsed (a missed first update).
2. A game is currently LIVE/CRIT and the payload is older than
   ``live_stale_minutes`` (a live snapshot that stopped advancing).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from src.infer_gate import (
    DEFAULT_MAX_LAG_MINUTES,
    ET_ZONE,
    find_due_games,
    parse_game_start_utc,
    parse_utc_timestamp,
)

DEFAULT_GRACE_MINUTES = 20
DEFAULT_LIVE_STALE_MINUTES = 45
LIVE_GAME_STATES = {"LIVE", "CRIT"}


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def load_payload(path: str) -> dict[str, Any]:
    """Load dashboard JSON and reject NaN/Infinity tokens that browsers cannot parse."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.loads(handle.read(), parse_constant=_reject_constant)


def _game_key(game: dict[str, Any]) -> Any:
    """Stable identity for de-duplicating games across the two stale checks."""
    return game.get("id") or (
        game.get("away"),
        game.get("home"),
        game.get("date"),
        game.get("start_time_utc"),
    )


def _game_labels(games: list[dict[str, Any]]) -> str:
    labels = ", ".join(
        f"{game.get('away', '?')}@{game.get('home', '?')} ({game.get('start_time_et', '?')})"
        for game in games[:5]
    )
    extra = f" (+{len(games) - 5} more)" if len(games) > 5 else ""
    return f"{labels}{extra}"


def check_freshness(
    data_json: str,
    *,
    now_utc: datetime | None = None,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
    max_lag_minutes: int = DEFAULT_MAX_LAG_MINUTES,
    delay_minutes: int = 3,
    live_stale_minutes: int = DEFAULT_LIVE_STALE_MINUTES,
) -> tuple[bool, str]:
    """Return (healthy, reason).

    A game is overdue when its start + delay has passed, the dashboard has not been
    updated since that trigger, and at least ``grace_minutes`` have elapsed so a run
    that is already in flight does not trip the alert.

    A live game is also stale when the payload itself is older than
    ``live_stale_minutes``, even if the update happened after puck drop. This catches
    a live snapshot that froze midway through a game.
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

    # Live games whose data has stopped advancing. The max-lag cap does not apply
    # here: a game can run long, and stale data is still stale after 180 minutes.
    updated = parse_utc_timestamp(meta.get("updated_utc"))
    live_stale: list[dict[str, Any]] = []
    if updated is not None and live_stale_minutes > 0:
        age = now - updated
        if age > timedelta(minutes=live_stale_minutes):
            overdue_keys = {_game_key(g) for g in overdue}
            for game in payload.get("games") or []:
                if str(game.get("game_state") or "").upper() not in LIVE_GAME_STATES:
                    continue
                start = parse_game_start_utc(game)
                if start is not None and start > now:
                    continue
                if _game_key(game) in overdue_keys:
                    continue  # already reported as a missed first update
                live_stale.append(game)

    if not overdue and not live_stale:
        return True, "OK - no game is waiting for an update."

    parts: list[str] = []
    if overdue:
        parts.append(f"{len(overdue)} game(s) started without a data update: {_game_labels(overdue)}")
    if live_stale:
        age_min = int((now - updated).total_seconds() // 60)
        parts.append(
            f"{len(live_stale)} live game(s) have not been updated in {age_min} min "
            f"(threshold {live_stale_minutes} min): {_game_labels(live_stale)}"
        )
    return False, " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that live games have fresh dashboard data.")
    parser.add_argument("--data-json", default="docs/data.json")
    parser.add_argument("--grace-minutes", type=int, default=DEFAULT_GRACE_MINUTES)
    parser.add_argument("--max-lag-minutes", type=int, default=DEFAULT_MAX_LAG_MINUTES)
    parser.add_argument(
        "--live-stale-minutes",
        type=int,
        default=DEFAULT_LIVE_STALE_MINUTES,
        help="Flag LIVE/CRIT games when the payload is older than this (0 disables).",
    )
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
            live_stale_minutes=args.live_stale_minutes,
        )
    except Exception as exc:  # noqa: BLE001 - any read/parse failure is a real alert
        print(f"Health Check -> STALE | payload error: {exc}")
        return 1

    stamp = (now or datetime.now(timezone.utc)).astimezone(ET_ZONE).strftime("%Y-%m-%d %I:%M %p ET")
    print(f"[{stamp}] Health Check -> {'OK' if healthy else 'STALE'} | {reason}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
