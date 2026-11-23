"""Persist the union of Tim Hortons board players seen across the day's waves.

The official groups are fixed, but the pipeline drops players once their game
starts. That means a scorer can be missing from the current board by the time
the goal is recorded. This module keeps every player who appeared on any wave's
board for the day so the scoreboard can still recognise them.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.data_sync import clean_name, normalize_team_abbrev

DEFAULT_KEEP_DAYS = 14


def _board_key(player: Any, team: Any) -> str:
    return f"{clean_name(player)}|{normalize_team_abbrev(team)}"


def load_tims_boards(path: str) -> dict[str, Any]:
    """Load the persisted board union, returning an empty structure on failure."""
    default: dict[str, Any] = {"days": {}}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return default
    if not isinstance(data, dict) or not isinstance(data.get("days"), dict):
        return default
    for day in data["days"].values():
        if not isinstance(day, dict):
            continue
        if not isinstance(day.get("players"), dict):
            day["players"] = {}
    return data


def update_board_day(
    boards: dict[str, Any],
    date_str: str,
    tims_groups: dict[str, list[dict[str, Any]]],
    captured_et: str,
    *,
    keep_days: int = DEFAULT_KEEP_DAYS,
) -> dict[str, dict[str, Any]]:
    """Merge this wave's board into the day's union and prune old days."""
    days = boards.setdefault("days", {})
    day = days.setdefault(date_str, {"players": {}})
    players = day.setdefault("players", {})

    for group_key, group_players in (tims_groups or {}).items():
        for candidate in group_players or []:
            name = candidate.get("player")
            team = normalize_team_abbrev(candidate.get("team"))
            if not name or not team:
                continue
            key = _board_key(name, team)
            record = players.get(key)
            if record is None:
                players[key] = {
                    "player": str(name),
                    "team": team,
                    "group": str(group_key),
                    "first_seen_et": captured_et,
                    "last_seen_et": captured_et,
                }
            else:
                record["last_seen_et"] = captured_et
                record["group"] = str(group_key)

    if keep_days > 0:
        for stale_date in sorted(days)[:-keep_days]:
            days.pop(stale_date, None)
    return players


def save_tims_boards(boards: dict[str, Any], path: str) -> None:
    """Persist the board union atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(boards, handle, indent=2)
    os.replace(tmp_path, path)


def board_players_for(boards: dict[str, Any], date_str: str) -> list[dict[str, Any]]:
    """Return the day's board union sorted by group then player name."""
    day = (boards.get("days") or {}).get(date_str) or {}
    players = (day.get("players") or {}).values()
    return sorted(
        (dict(p) for p in players if isinstance(p, dict) and p.get("player") and p.get("team")),
        key=lambda p: (int(str(p.get("group", "9")) or 9), str(p.get("player", ""))),
    )
