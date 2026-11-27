"""Drive live inference on a fixed cadence inside the daily game windows.

GitHub's shared scheduler can delay a cron by hours or drop it entirely. Instead of
relying on a 15-minute grid of one-shot triggers (which was delivering ~1 of 47
ticks), the live-loop workflow starts one long-lived job near a game window and
this module ticks the gate locally every few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, time as dt_time, timezone
from typing import Any, Callable

from src.infer_gate import DEFAULT_MAX_LAG_MINUTES, ET_ZONE, evaluate_gate, parse_game_start_utc

MATINEE_START = dt_time(12, 15)
MATINEE_END = dt_time(16, 15)
EVENING_START = dt_time(18, 40)
EVENING_END = dt_time(21, 45)
LATE_START = dt_time(22, 15)
LATE_END = dt_time(23, 40)

COMMIT_PATHS = ("docs/data.json", "docs/tims_history.json", "docs/tims_boards.json", "docs/players.json", "docs/injury_history/", "docs/roster_history/", "docs/transaction_history.json", "docs/calls_history.json", "docs/calls_scoreboard.json", "models/")
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_LIVE_ODDS_CAP = 3


def parse_now(value: str | None) -> datetime:
    """Parse an ISO-8601 timestamp (Z accepted) or return the current UTC time."""
    if not value:
        return datetime.now(timezone.utc)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_window_end(now_et: datetime) -> datetime | None:
    """Return the UTC end of the active live window, or None outside both windows.

    Windows are evaluated in ET so they follow daylight saving automatically:
    matinee 12:15-4:15 PM, evening 6:40-9:45 PM, late 10:15-11:40 PM.  The gaps
    leave room for the fixed noon / 4:30 PM / 10 PM odds runs, which share the
    repo-write lock with this workflow.
    """
    local = now_et.time()
    if MATINEE_START <= local < MATINEE_END:
        end_time = MATINEE_END
    elif EVENING_START <= local < EVENING_END:
        end_time = EVENING_END
    elif LATE_START <= local < LATE_END:
        end_time = LATE_END
    else:
        return None
    end_et = datetime.combine(now_et.date(), end_time, tzinfo=ET_ZONE)
    return end_et.astimezone(timezone.utc)


def load_payload(path: str) -> dict[str, Any]:
    """Load the dashboard payload, rejecting non-finite JSON constants."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    def _reject(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(text, parse_constant=_reject)


def window_has_games(payload: dict[str, Any], end_utc: datetime) -> bool:
    """Return True when a non-final game for the window's ET date starts before it ends.

    Used to release the shared repo-write lock early on days with no matinee slate,
    instead of holding it for hours with nothing to do.  Games from other dates are
    ignored so a stale payload cannot keep the loop alive all afternoon.
    """
    games = payload.get("games") or []
    if not games:
        return False
    window_date = end_utc.astimezone(ET_ZONE).date()
    for game in games:
        if game.get("game_state") in ("FINAL", "OFF"):
            continue
        start = parse_game_start_utc(game)
        if start is None:
            return True
        if start.astimezone(ET_ZONE).date() != window_date:
            continue
        if start <= end_utc:
            return True
    return False


def _default_inference(live_odds_cap: int = DEFAULT_LIVE_ODDS_CAP) -> int:
    return subprocess.run(
        [
            sys.executable,
            "main_pipeline.py",
            "--mode",
            "infer",
            "--odds-policy",
            "live_loop",
            "--live-odds-cap",
            str(live_odds_cap),
        ],
        check=False,
    ).returncode


def _git(*args: str) -> int:
    return subprocess.run(["git", *args], check=False).returncode


def commit_and_push() -> None:
    """Commit the regenerated artifacts and rebase-push them to main."""
    _git("config", "user.name", "github-actions[bot]")
    _git("config", "user.email", "github-actions[bot]@users.noreply.github.com")
    os.makedirs("docs/injury_history", exist_ok=True)
    os.makedirs("docs/roster_history", exist_ok=True)
    _git("add", *COMMIT_PATHS)

    if _git("diff", "--staged", "--quiet") == 0:
        print("No changes to commit.")
        return

    stamp = datetime.now(timezone.utc).astimezone(ET_ZONE).strftime("%Y-%m-%d %I:%M %p ET")
    if _git("commit", "-m", f"Live Inference Update ({stamp})") != 0:
        print("git commit failed; leaving changes for the next tick.", file=sys.stderr)
        return

    if _git("pull", "--rebase", "origin", "main") != 0:
        _git("rebase", "--abort")
        print("git pull --rebase failed; will retry on the next tick.", file=sys.stderr)
        return

    if _git("push", "origin", "main") != 0:
        print("git push failed; will retry on the next tick.", file=sys.stderr)


def run_iteration(
    data_json: str,
    *,
    max_lag_minutes: int = DEFAULT_MAX_LAG_MINUTES,
    live_odds_cap: int = DEFAULT_LIVE_ODDS_CAP,
    dry_run: bool = False,
    now_utc: datetime | None = None,
    inference_runner: Callable[[], int] | None = None,
    committer: Callable[[], None] | None = None,
) -> bool:
    """Evaluate the gate once and run inference + commit when it is due."""
    should_run, reason = evaluate_gate(
        data_json,
        event_name="schedule",
        cron="",
        now_utc=now_utc,
        max_lag_minutes=max_lag_minutes,
    )
    stamp_dt = now_utc or datetime.now(timezone.utc)
    stamp = stamp_dt.astimezone(ET_ZONE).strftime("%Y-%m-%d %I:%M %p ET")
    print(f"[{stamp}] gate: should_run={str(should_run).lower()} | {reason}", flush=True)

    if not should_run:
        return False
    if dry_run:
        print("[dry-run] inference and commit skipped", flush=True)
        return True

    runner = inference_runner or (lambda: _default_inference(live_odds_cap=live_odds_cap))
    code = runner()
    if code != 0:
        print(f"inference exited {code}; skipping commit", file=sys.stderr, flush=True)
        return True

    (committer or commit_and_push)()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-json", default="docs/data.json")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-lag-minutes", type=int, default=DEFAULT_MAX_LAG_MINUTES)
    parser.add_argument("--live-odds-cap", type=int, default=DEFAULT_LIVE_ODDS_CAP)
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate the gate without running inference.")
    parser.add_argument("--now", default="", help="ISO-8601 override for tests; implies --once.")
    args = parser.parse_args(argv)

    now = parse_now(args.now)
    if args.now:
        args.once = True

    end = resolve_window_end(now.astimezone(ET_ZONE))
    if end is None and not args.once:
        print("Live loop: outside the 12:15-4:15 / 6:40-9:45 / 10:15-11:40 PM ET windows; exiting.", flush=True)
        return 0

    if not args.once and end is not None:
        try:
            payload = load_payload(args.data_json)
        except Exception as exc:  # noqa: BLE001 - surface the exact file problem in the log
            print(f"Live loop: cannot read {args.data_json} ({exc}); exiting.", file=sys.stderr, flush=True)
            return 1
        if not window_has_games(payload, end):
            print("Live loop: no games scheduled inside this window; exiting.", flush=True)
            return 0

    max_iterations = 0
    if not args.once and end is not None:
        window_minutes = max(0.0, (end - now).total_seconds() / 60.0)
        max_iterations = int(window_minutes * 60 // max(1, args.interval_seconds)) + 5

    iterations = 0
    while True:
        current = parse_now(args.now) if args.now else datetime.now(timezone.utc)
        if not args.once and end is not None and current >= end:
            print("Live loop: window closed; exiting.", flush=True)
            return 0
        if max_iterations and iterations >= max_iterations:
            print("Live loop: iteration cap reached; exiting.", flush=True)
            return 0

        run_iteration(
            args.data_json,
            max_lag_minutes=args.max_lag_minutes,
            live_odds_cap=args.live_odds_cap,
            dry_run=args.dry_run,
            now_utc=parse_now(args.now) if args.now else None,
        )
        iterations += 1

        if args.once:
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
