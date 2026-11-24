"""Live scoreboard for Tim Hortons-listed skaters.

Counts today's goals from NHL boxscores and keeps only players who appeared on
the Tims board during any wave of the day. Works for in-progress and final
games, so a scorer stays visible after the board has dropped their team.
"""

from __future__ import annotations

from typing import Any

from src.data_sync import (
    PlayerMatcher,
    _fetch_with_retries,
    clean_name,
    create_http_session,
    normalize_team_abbrev,
)
from src.tims_tracker import _extract_boxscore_goal_entries

SCORING_STATES = ("LIVE", "CRIT", "FINAL", "OFF")  # NHL uses OFF for completed games
MAX_BOXSCORES = 20


def _wave_map(today_games: list[dict[str, Any]]) -> dict[str, tuple[int, str]]:
    """Map each team to (wave number, wave start label) from today's puck drops."""
    starts = sorted(
        {
            str(game.get("start_time_utc"))
            for game in today_games
            if game.get("start_time_utc")
        }
    )
    labels = {
        str(game.get("start_time_utc")): str(game.get("start_time_et") or "TBD")
        for game in today_games
        if game.get("start_time_utc")
    }
    team_wave: dict[str, tuple[int, str]] = {}
    for wave_idx, start_utc in enumerate(starts):
        label = labels.get(start_utc, "TBD")
        for game in today_games:
            if str(game.get("start_time_utc")) != start_utc:
                continue
            for team in (game.get("away"), game.get("home")):
                team_norm = normalize_team_abbrev(team)
                if team_norm:
                    team_wave[team_norm] = (wave_idx + 1, label)
    return team_wave


def choose_scoreboard_slate(
    today_games: list[dict[str, Any]] | None,
    previous_games: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Choose which day's slate the Tims scoreboard should display.

    Returns ``(slate_games, is_previous_day)``. If any of today's games is
    already in a scoring state (LIVE/CRIT/FINAL/OFF) we stay on today. Otherwise,
    if the previous day still has a game in a scoring state, we keep showing the
    previous day's slate. On an off-day with no scoring games either way we fall
    back to today's (possibly empty) slate.
    """
    today_list = list(today_games or [])
    previous_list = list(previous_games or [])

    today_scoring = any(
        str(g.get("game_state") or "").upper() in SCORING_STATES for g in today_list
    )
    if today_scoring:
        return today_list, False

    previous_scoring = any(
        str(g.get("game_state") or "").upper() in SCORING_STATES for g in previous_list
    )
    if previous_scoring:
        return previous_list, True

    return today_list, False


def build_tims_scoreboard(
    schedule_games: list[dict[str, Any]],
    board_players: list[dict[str, Any]],
    *,
    session: Any | None = None,
    allow_network: bool = True,
    date_str: str | None = None,
) -> list[dict[str, Any]]:
    """Return goal scorers who were on any Tims wave board for the chosen slate."""
    if not allow_network or not board_players:
        return []

    http = session or create_http_session()
    if date_str is None:
        selected_games = [g for g in (schedule_games or []) if g.get("is_today", True)]
    else:
        selected_games = [
            g for g in (schedule_games or []) if str(g.get("date")) == str(date_str)
        ]
    team_wave = _wave_map(selected_games)

    board_entries: list[dict[str, Any]] = []
    for player in board_players:
        team = normalize_team_abbrev(player.get("team"))
        name = str(player.get("player") or "")
        if not team or not name:
            continue
        board_entries.append(
            {
                "player_id": 0,
                "name": name,
                "name_clean": clean_name(name),
                "team": team,
                "position": "",
                "payload": player,
            }
        )
    matcher = PlayerMatcher(board_entries)

    scorers: dict[str, dict[str, Any]] = {}
    scoring_games = [
        g
        for g in selected_games
        if str(g.get("game_state") or "").upper() in SCORING_STATES and int(g.get("id") or 0) > 0
    ]
    for game in scoring_games[:MAX_BOXSCORES]:
        game_id = int(game.get("id") or 0)
        away = normalize_team_abbrev(game.get("away"))
        home = normalize_team_abbrev(game.get("home"))
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
        response = _fetch_with_retries(http, url, timeout=12, attempts=2)
        if response is None or response.status_code != 200:
            continue
        for entry in _extract_boxscore_goal_entries(response.json(), away_team=away, home_team=home):
            goals = int((entry.get("payload") or {}).get("goals", 0) or 0)
            if goals <= 0:
                continue
            team = normalize_team_abbrev(entry.get("team"))
            matched = matcher.match(
                name=str(entry.get("name") or ""),
                team=team,
                position=str((entry.get("payload") or {}).get("position") or ""),
                allow_initial_fallback=True,
                require_same_team=True,
                require_compatible_pos=False,
            )
            if matched is None:
                continue
            # PlayerMatcher.match returns the stored payload (the board player).
            board_player = matched if isinstance(matched, dict) else {}
            key = f"{clean_name(board_player.get('player'))}|{team}"
            wave, wave_et = team_wave.get(team, (0, "--"))
            opponent = home if team == away else away
            scorers[key] = {
                "player": str(board_player.get("player") or entry.get("name") or ""),
                "team": team,
                "opponent": opponent,
                "group": str(board_player.get("group") or ""),
                "wave": wave,
                "wave_et": wave_et,
                "goals": goals,
                "game_state": str(game.get("game_state") or "").upper(),
            }

    return sorted(
        scorers.values(),
        key=lambda row: (-int(row.get("goals", 0)), int(row.get("wave", 0)), str(row.get("player", ""))),
    )
