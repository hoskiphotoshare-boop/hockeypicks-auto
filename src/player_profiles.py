"""Build the public players.json payload and fetch NHL game logs."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from src.data_sync import _fetch_with_retries, clean_name


def _load_cache(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def _get(row: Any, col: str, default: Any = 0) -> Any:
    try:
        if isinstance(row, dict):
            val = row.get(col, default)
        else:
            val = row[col] if col in row else default
    except Exception:
        return default
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    return val


def read_season_skater_stats(archive_dir: str, year: Any) -> dict[str, dict[str, Any]]:
    """Read MoneyPuck skater stats for a season into a clean-name-keyed dict."""
    path = os.path.join(archive_dir, f"{year}_regular_skaters.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}

    has_situation = "situation" in df.columns
    if has_situation:
        sit = df["situation"].astype(str).str.lower()
        all_df = df[sit == "all"]
        if all_df.empty:
            all_df = df
        # MoneyPuck encodes power play as 5on4/5on3 (not "pp").
        pp_df = df[sit.isin({"pp", "powerplay", "5on4", "5on3", "4on3"})]
    else:
        all_df = df
        pp_df = pd.DataFrame()

    # pp points lookup by clean name
    pp_points_by_name: dict[str, float] = {}
    if not pp_df.empty:
        for _, row in pp_df.iterrows():
            name = str(_get(row, "name", "") or _get(row, "player", "") or "")
            if not name or name.lower() == "nan":
                continue
            key = clean_name(name)
            val = _get(row, "I_F_points", None)
            if val is None:
                val = _get(row, "points", 0)
            try:
                pp_points_by_name[key] = pp_points_by_name.get(key, 0.0) + float(val or 0)
            except (TypeError, ValueError):
                pp_points_by_name.setdefault(key, 0.0)

    result: dict[str, dict[str, Any]] = {}
    for _, row in all_df.iterrows():
        name = str(_get(row, "name", "") or _get(row, "player", "") or "")
        if not name or name.lower() == "nan":
            continue
        key = clean_name(name)
        try:
            gp = int(float(_get(row, "games_played", _get(row, "gp", 0)) or 0))
        except (TypeError, ValueError):
            gp = 0
        try:
            goals = int(float(_get(row, "I_F_goals", _get(row, "goals", 0)) or 0))
        except (TypeError, ValueError):
            goals = 0
        try:
            if "I_F_assists" in row:
                assists = int(float(_get(row, "I_F_assists", 0) or 0))
            elif "I_F_primaryAssists" in row or "I_F_secondaryAssists" in row:
                assists = int(float(_get(row, "I_F_primaryAssists", 0) or 0)) + int(
                    float(_get(row, "I_F_secondaryAssists", 0) or 0)
                )
            else:
                assists = int(float(_get(row, "assists", 0) or 0))
        except (TypeError, ValueError):
            assists = 0
        try:
            points = int(float(_get(row, "I_F_points", _get(row, "points", 0)) or 0))
        except (TypeError, ValueError):
            points = 0
        try:
            sog = int(float(_get(row, "I_F_shotsOnGoal", _get(row, "shotsOnGoal", _get(row, "sog", 0))) or 0))
        except (TypeError, ValueError):
            sog = 0
        icetime = _get(row, "icetime", None)
        try:
            icetime_f = float(icetime) if icetime is not None else 0.0
        except (TypeError, ValueError):
            icetime_f = 0.0
        if gp > 0 and icetime_f > 0:
            toi_pg = icetime_f / gp / 60.0
        else:
            toi_val = _get(row, "toi", _get(row, "toi_pg", 0))
            try:
                toi_pg = float(toi_val or 0)
            except (TypeError, ValueError):
                toi_pg = 0.0
        pp_points = int(pp_points_by_name.get(key, 0))
        result[key] = {
            "gp": gp,
            "goals": goals,
            "assists": assists,
            "points": points,
            "sog": sog,
            "toi_pg": round(toi_pg, 3) if isinstance(toi_pg, float) else toi_pg,
            "pp_points": pp_points,
        }
    return result


def parse_game_log_payload(payload: Any) -> list[dict[str, Any]]:
    """Parse an NHL player game-log payload into the last-6 rows (newest-last)."""
    if not isinstance(payload, dict):
        return []
    games = payload.get("gameLog") or payload.get("games") or payload.get("gameLogs") or []
    if not isinstance(games, list):
        return []
    rows: list[dict[str, Any]] = []
    for g in games:
        if not isinstance(g, dict):
            continue
        date = g.get("gameDate") or g.get("date")
        opp = g.get("opponentAbbrev") or g.get("opponentAbbreviation") or g.get("opponentTeamAbbrev") or g.get("opponent")
        goals = g.get("goals", 0)
        assists = g.get("assists", 0)
        points = g.get("points", None)
        if points is None:
            try:
                points = int(goals or 0) + int(assists or 0)
            except Exception:
                points = 0
        sog = g.get("shots", None)
        if sog is None:
            sog = g.get("shotsOnGoal", 0)
        toi = g.get("toi", None)
        if toi is None:
            toi = g.get("timeOnIce", "")
        try:
            goals = int(goals) if goals is not None else 0
        except (TypeError, ValueError):
            goals = 0
        try:
            assists = int(assists) if assists is not None else 0
        except (TypeError, ValueError):
            assists = 0
        try:
            points = int(points) if points is not None else 0
        except (TypeError, ValueError):
            points = 0
        try:
            sog = int(sog) if sog is not None else 0
        except (TypeError, ValueError):
            sog = 0
        rows.append({
            "date": date,
            "opp": opp,
            "g": goals,
            "a": assists,
            "p": points,
            "sog": sog,
            "toi": toi,
        })
    rows.sort(key=lambda r: str(r.get("date") or ""))
    return rows[-6:]


# aliases for tests
parse_game_log = parse_game_log_payload
parse_game_logs = parse_game_log_payload
parse_player_game_log = parse_game_log_payload
parse_game_log_rows = parse_game_log_payload
parse_game_log_row = parse_game_log_payload
parse_nhl_game_log = parse_game_log_payload
parse_game_logs_payload = parse_game_log_payload


def fetch_player_game_logs(
    player_ids: Any,
    season: Any,
    game_type: Any,
    session: Any,
    cache_path: str,
    max_age_hours: int = 12,
    concurrency: int = 8,
) -> dict[Any, list[dict[str, Any]]]:
    """Fetch (with caching) NHL game logs for the given player ids."""
    ids = [pid for pid in (player_ids or []) if pid is not None]
    cache = _load_cache(cache_path)
    now = datetime.now(timezone.utc)
    stale = True
    if cache and isinstance(cache.get("logs"), dict):
        updated = cache.get("updated_utc")
        if updated:
            try:
                dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_hours = (now - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
                if age_hours < max_age_hours and cache.get("season") == season:
                    stale = False
            except Exception:
                stale = False if cache.get("season") == season else True
    logs: dict[Any, list[dict[str, Any]]] = {}
    if isinstance(cache.get("logs"), dict):
        logs = dict(cache["logs"])

    if not stale:
        return logs

    # refresh
    new_logs = dict(logs)

    def _fetch_one(pid: Any) -> tuple[Any, list[dict[str, Any]] | None]:
        url = f"https://api-web.nhle.com/v1/player/{pid}/game-log/{season}/{game_type}"
        try:
            res = _fetch_with_retries(session, url, timeout=12, attempts=2) if session is not None else None
        except Exception:
            res = None
        if res is None or getattr(res, "status_code", None) != 200:
            return pid, None
        try:
            payload = res.json()
        except Exception:
            return pid, None
        return pid, parse_game_log_payload(payload)

    if session is not None and ids:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency or 1))) as ex:
                futures = [ex.submit(_fetch_one, pid) for pid in ids]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        pid, rows = fut.result()
                    except Exception:
                        continue
                    if rows is not None:
                        new_logs[pid] = rows
                        new_logs[str(pid)] = rows
        except Exception:
            pass

    try:
        _save_cache(cache_path, {"updated_utc": now.isoformat(), "season": season, "logs": new_logs})
    except Exception:
        pass
    return new_logs


def build_players_payload(
    slate_scorers: list[dict[str, Any]] | None = None,
    season_stats: dict[str, dict[str, Any]] | None = None,
    game_logs: dict[Any, list[dict[str, Any]]] | None = None,
    tims_groups: dict[Any, Any] | None = None,
    schedule_games: list[dict[str, Any]] | None = None,
    starting_goalies: dict[str, Any] | None = None,
    updated_utc: Any = None,
) -> dict[str, Any]:
    season_stats = season_stats or {}
    game_logs = game_logs or {}
    tims_groups = tims_groups or {}
    schedule_games = schedule_games or []
    starting_goalies = starting_goalies or {}

    # index schedule by team
    game_by_team: dict[str, dict[str, Any]] = {}
    for g in schedule_games:
        if not isinstance(g, dict):
            continue
        for key in ("away", "home"):
            t = g.get(key)
            if t:
                game_by_team[t] = g

    players: list[dict[str, Any]] = []
    for s in slate_scorers or []:
        if not isinstance(s, dict):
            continue
        try:
            live_gp = int(s.get("live_gp", 0) or 0)
        except (TypeError, ValueError):
            live_gp = 0
        if live_gp < 3:
            continue

        name = s.get("player")
        team = s.get("team")
        position = s.get("position") or "F"
        stats = season_stats.get(clean_name(name), {}) if name else {}

        # player id
        pid = s.get("playerId")
        if pid is None:
            pid = s.get("id")
        try:
            pid = int(pid) if pid is not None and str(pid).strip() != "" else 0
        except (TypeError, ValueError):
            pid = 0

        # tonight
        game = game_by_team.get(team) if team else None
        if game:
            if team == game.get("home"):
                home = True
                opp = game.get("away")
            else:
                home = False
                opp = game.get("home")
            state = game.get("game_state")
        else:
            home = False
            opp = s.get("opponent")
            state = None

        goal_rate = s.get("final_score", 0)
        try:
            goal_rate_f = float(goal_rate) if goal_rate is not None else 0.0
        except (TypeError, ValueError):
            goal_rate_f = 0.0
        # guard overflow
        try:
            goal_prob = 1.0 - math.exp(-goal_rate_f)
        except (OverflowError, ValueError):
            goal_prob = 1.0 if goal_rate_f > 0 else 0.0
        if math.isnan(goal_prob) or math.isinf(goal_prob):
            goal_prob = 1.0 if goal_rate_f > 0 else 0.0

        def_mult = s.get("def_multiplier")
        try:
            def_mult_f = float(def_mult) if def_mult is not None else 0.0
        except (TypeError, ValueError):
            def_mult_f = 0.0

        # goalie: opponent first, fallback own team
        opp_goalie = starting_goalies.get(opp) if opp is not None else None
        own_goalie = starting_goalies.get(team) if team is not None else None
        goalie = opp_goalie if opp_goalie is not None else own_goalie
        if isinstance(goalie, dict):
            goalie = goalie.get("goalie") or goalie.get("name") or goalie.get("player") or goalie

        injury = {
            "status": s.get("injury_status"),
            "out": bool(s.get("injury_out")),
            "dtd": bool(s.get("injury_dtd")),
            "note": s.get("injury_note") or "",
        }

        # last6 from game_logs
        rows = None
        if pid:
            rows = game_logs.get(pid)
            if rows is None:
                rows = game_logs.get(str(pid))
        if rows is None and name:
            rows = game_logs.get(name)
            if rows is None:
                rows = game_logs.get(clean_name(name))
        if not isinstance(rows, list):
            rows = []
        sorted_rows = sorted(rows, key=lambda r: str((r or {}).get("date") or ""))
        last6 = [dict(r) for r in sorted_rows[-6:]]

        gp = int(stats.get("gp", 0) or 0) if isinstance(stats, dict) else 0
        goals = int(stats.get("goals", 0) or 0) if isinstance(stats, dict) else 0
        assists = int(stats.get("assists", 0) or 0) if isinstance(stats, dict) else 0
        points = int(stats.get("points", 0) or 0) if isinstance(stats, dict) else 0
        sog = int(stats.get("sog", 0) or 0) if isinstance(stats, dict) else 0
        toi_pg = float(stats.get("toi_pg", 0) or 0) if isinstance(stats, dict) else 0.0
        pp_points = int(stats.get("pp_points", 0) or 0) if isinstance(stats, dict) else 0
        gpg = (goals / gp) if gp else 0.0
        ppg = (points / gp) if gp else 0.0
        sog_pg = (sog / gp) if gp else 0.0

        tims_group = tims_groups.get(name)
        if tims_group is None and name:
            tims_group = tims_groups.get(clean_name(name))

        player_obj = {
            "id": pid,
            "name": name,
            "team": team,
            "position": position,
            "gp": gp,
            "goals": goals,
            "assists": assists,
            "points": points,
            "sog": sog,
            "toi_pg": toi_pg,
            "pp_points": pp_points,
            "gpg": gpg,
            "ppg": ppg,
            "sog_pg": sog_pg,
            "last6": last6,
            "tonight": {
                "opp": opp,
                "home": home,
                "state": state,
                "goal_rate": goal_rate_f,
                "goal_prob": goal_prob,
                "def_multiplier": def_mult_f,
                "goalie": goalie,
            },
            "injury": injury,
            "tims_group": tims_group,
        }
        players.append(player_obj)

    players.sort(key=lambda p: (-float((p.get("tonight") or {}).get("goal_rate") or 0.0), str(p.get("name") or "")))
    players = players[:600]

    if updated_utc is None:
        updated_utc = datetime.now(timezone.utc).isoformat()
    elif hasattr(updated_utc, "isoformat"):
        updated_utc = updated_utc.isoformat()
    else:
        updated_utc = str(updated_utc)

    return {"updated_utc": updated_utc, "players": players}
