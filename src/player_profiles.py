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
        if not math.isfinite(float(toi_pg or 0.0)):
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
    return rows[-10:]


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
    if cache and isinstance(cache.get("logs"), dict) and cache.get("schema") == 2:
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
        _save_cache(cache_path, {"schema": 2, "updated_utc": now.isoformat(), "season": season, "logs": new_logs})
    except Exception:
        pass
    return new_logs


def read_prior_player_rates(archive_dir: str, year: Any) -> dict[str, dict[str, float]]:
    """Per-game prior-season rates keyed by clean name (goals/assists/points/sog)."""
    stats = read_season_skater_stats(archive_dir, year)
    out: dict[str, dict[str, float]] = {}
    for key, row in stats.items():
        gp = float(row.get("gp", 0) or 0)
        if gp <= 0:
            continue
        out[key] = {
            "g_pg": float(row.get("goals", 0) or 0) / gp,
            "a_pg": float(row.get("assists", 0) or 0) / gp,
            "p_pg": float(row.get("points", 0) or 0) / gp,
            "sog_pg": float(row.get("sog", 0) or 0) / gp,
        }
    return out


def read_prior_team_rates(archive_dir: str, year: Any) -> dict[str, dict[str, float]]:
    """Prior-season team rates keyed by abbreviation."""
    path = os.path.join(archive_dir, f"{year}_regular_teams.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "situation" in df.columns:
        df = df[df["situation"].astype(str).str.lower() == "all"]
    if "position" in df.columns:
        df = df[df["position"].astype(str).str.lower().isin(["all", "team", "team level"])]
    out: dict[str, dict[str, float]] = {}
    for row in df.drop_duplicates(subset=["team"]).to_dict("records"):
        gp = float(row.get("games_played", 0) or 0)
        if gp <= 0:
            continue
        ga = float(row.get("goalsAgainst", 0) or 0)
        sa = float(row.get("shotsOnGoalAgainst", 0) or 0)
        out[str(row.get("team", "")).upper()] = {
            "gf_pg": float(row.get("goalsFor", 0) or 0) / gp,
            "ga_pg": ga / gp,
            "save_pct": (1.0 - ga / sa) if sa > 0 else 0.9,
            "xgf_pg": float(row.get("xGoalsFor", 0) or 0) / gp,
            "xga_pg": float(row.get("xGoalsAgainst", 0) or 0) / gp,
        }
    return out


def build_team_rolling_context(logs_dir: str, season: str, games: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Last-10 GF/GA/win% and rest days before each team's next scheduled game."""
    upcoming: dict[str, str] = {}
    for g in games or []:
        date = str(g.get("date") or "")[:10]
        for team in (g.get("away"), g.get("home")):
            if team and (team not in upcoming or date < upcoming[team]):
                upcoming[team] = date
    out: dict[str, dict[str, float]] = {}
    for team, game_date in upcoming.items():
        path = os.path.join(logs_dir, f"{team}_{season}_gamelog.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        try:
            cutoff = pd.Timestamp(game_date)
        except Exception:
            cutoff = df["Date"].max() + pd.Timedelta(days=1)
        prior = df[df["Date"] < cutoff]
        if prior.empty:
            continue
        last = prior.tail(10)
        out[team] = {
            "gf_pg": float(pd.to_numeric(last.get("Team_Score", 0), errors="coerce").fillna(0).mean()),
            "ga_pg": float(pd.to_numeric(last.get("Opp_Score", 0), errors="coerce").fillna(0).mean()),
            "win_pct": float(pd.to_numeric(last.get("Won", 0), errors="coerce").fillna(0).mean()),
            "rest_days": float(max(0, min(10, (cutoff - prior["Date"].max()).days))),
        }
    return out


def load_stat_bundle(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        import joblib
        bundle = joblib.load(path)
        return bundle if isinstance(bundle, dict) and bundle.get("models") else None
    except Exception:
        return None


def _toi_minutes(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw or ":" not in raw:
        return 0.0
    try:
        mm, ss = raw.split(":")[:2]
        return float(mm) + float(ss) / 60.0
    except ValueError:
        return 0.0


def _poisson_interval(lam: float) -> tuple[int, int]:
    l = max(0.0, float(lam or 0.0))
    if l <= 0:
        return 0, 0
    def cdf(k: int) -> float:
        term = math.exp(-l)
        total = term
        for i in range(1, k + 1):
            term *= l / i
            total += term
        return total
    lo = 0
    while lo < 30 and cdf(lo) < 0.10:
        lo += 1
    hi = lo
    while hi < 30 and cdf(hi) < 0.90:
        hi += 1
    return lo, hi


def predict_expected_stats(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    prior: dict[str, float] | None,
    team_ctx: dict[str, float] | None,
    opp_ctx: dict[str, float] | None,
    prior_team: dict[str, float] | None,
    prior_opp: dict[str, float] | None,
    *,
    gp_season: int,
    is_home: bool,
    is_defense: bool,
    bundle: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not bundle:
        return None
    rows = sorted([r for r in (rows or []) if isinstance(r, dict)], key=lambda r: str(r.get("date") or ""))
    n = len(rows)
    def rate(key: str, w: int) -> float:
        chunk = rows[-w:] if n else []
        if not chunk:
            return 0.0
        return float(sum(float((r or {}).get(key, 0) or 0) for r in chunk) / len(chunk))
    prior = prior or {}
    team_ctx = team_ctx or {}
    opp_ctx = opp_ctx or {}
    prior_team = prior_team or {}
    prior_opp = prior_opp or {}
    gp = max(1, int(gp_season or 0))
    def season_rate(key: str) -> float:
        try:
            return float(stats.get(key, 0) or 0) / gp
        except Exception:
            return 0.0
    prior_g = float(prior.get("g_pg", 0) or 0)
    prior_a = float(prior.get("a_pg", 0) or 0)
    prior_p = float(prior.get("p_pg", 0) or 0)
    prior_sog = float(prior.get("sog_pg", 0) or 0)
    feats = {
        "roll5_goals": rate("g", 5), "roll10_goals": rate("g", 10),
        "roll5_assists": rate("a", 5), "roll10_assists": rate("a", 10),
        "roll5_points": rate("p", 5), "roll10_points": rate("p", 10),
        "roll5_sog": rate("sog", 5), "roll10_sog": rate("sog", 10),
        "roll5_toi_min": sum(_toi_minutes(r.get("toi")) for r in rows[-5:]) / max(1, len(rows[-5:])),
        "roll10_toi_min": sum(_toi_minutes(r.get("toi")) for r in rows[-10:]) / max(1, len(rows[-10:])),
        "season_goals_pg": season_rate("goals"), "season_assists_pg": season_rate("assists"),
        "season_points_pg": season_rate("points"), "season_sog_pg": season_rate("sog"),
        "season_toi_min_pg": float(stats.get("toi_pg", 0) or 0),
        "prior_g_pg": prior_g, "prior_a_pg": prior_a, "prior_p_pg": prior_p, "prior_sog_pg": prior_sog,
        "prior_sh_pct": min(0.25, max(0.02, prior_g / max(0.05, prior_sog))) if prior_sog > 0 else 0.08,
        "team_gf_pg": float(team_ctx.get("gf_pg", 3.0) or 3.0),
        "team_ga_pg": float(team_ctx.get("ga_pg", 3.0) or 3.0),
        "team_win_pct": float(team_ctx.get("win_pct", 0.5) or 0.5),
        "opp_gf_pg": float(opp_ctx.get("gf_pg", 3.0) or 3.0),
        "opp_ga_pg": float(opp_ctx.get("ga_pg", 3.0) or 3.0),
        "opp_win_pct": float(opp_ctx.get("win_pct", 0.5) or 0.5),
        "prior_team_gf_pg": float(prior_team.get("gf_pg", 3.0) or 3.0),
        "prior_team_ga_pg": float(prior_team.get("ga_pg", 3.0) or 3.0),
        "prior_team_save_pct": float(prior_team.get("save_pct", 0.9) or 0.9),
        "prior_opp_gf_pg": float(prior_opp.get("gf_pg", 3.0) or 3.0),
        "prior_opp_ga_pg": float(prior_opp.get("ga_pg", 3.0) or 3.0),
        "prior_opp_save_pct": float(prior_opp.get("save_pct", 0.9) or 0.9),
        "gp_season": float(gp),
        "rest_days": float(team_ctx.get("rest_days", 3.0) or 3.0),
        "is_home": 1.0 if is_home else 0.0,
        "is_defense": 1.0 if is_defense else 0.0,
        "is_b2b": 1.0 if float(team_ctx.get("rest_days", 3.0) or 3.0) <= 1 else 0.0,
    }
    cols = bundle.get("features") or []
    try:
        x = [float(feats.get(c, 0.0) or 0.0) for c in cols]
    except Exception:
        return None
    group = "D" if is_defense else "F"
    prior_map = {"goals": prior_g, "assists": prior_a, "points": prior_p, "sog": prior_sog}
    cal = bundle.get("calibration") or {}
    scales = bundle.get("interval_scale") or {}
    out: dict[str, Any] = {}
    for target in ("goals", "assists", "sog"):
        models = (bundle.get("models") or {}).get(target) or {}
        model = models.get(group) or models.get("F")
        if model is None:
            return None
        bm = math.log(min(5.0, max(0.02, float(prior_map.get(target, 0.05) or 0.05))))
        try:
            raw = float(model["mean"].predict([x], base_margin=[bm])[0])
            lo = float(model["q10"].predict([x], base_margin=[bm])[0])
            hi = float(model["q90"].predict([x], base_margin=[bm])[0])
        except Exception:
            return None
        c = float(cal.get(target, 1.0) or 1.0)
        k = float(scales.get(target, 1.0) or 1.0)
        pred = max(0.0, raw * c)
        lo_i = max(0.0, pred - k * max(0.0, raw - lo))
        hi_i = max(pred, pred + k * max(0.0, hi - raw))
        if hi_i <= pred + 1e-9:
            hi_i = pred + max(0.5, pred * 0.8)
        if lo_i >= pred - 1e-9:
            lo_i = max(0.0, pred - max(0.5, pred * 0.8))
        out[target] = round(pred, 3)
        out[f"{target}_lo"] = float(math.floor(lo_i))
        out[f"{target}_hi"] = float(math.ceil(hi_i))
    # Points is exactly G + A for consistency; use a Poisson interval for the sum.
    pts = out["goals"] + out["assists"]
    plo, phi = _poisson_interval(pts)
    out["points"] = round(pts, 3)
    out["points_lo"] = float(plo)
    out["points_hi"] = float(phi)
    out["model"] = "xgboost-poisson-v1"
    return out


def build_players_payload(
    slate_scorers: list[dict[str, Any]] | None = None,
    season_stats: dict[str, dict[str, Any]] | None = None,
    game_logs: dict[Any, list[dict[str, Any]]] | None = None,
    tims_groups: dict[Any, Any] | None = None,
    schedule_games: list[dict[str, Any]] | None = None,
    starting_goalies: dict[str, Any] | None = None,
    updated_utc: Any = None,
    stat_bundle: dict[str, Any] | None = None,
    prior_stats: dict[str, dict[str, float]] | None = None,
    team_context: dict[str, dict[str, float]] | None = None,
    prior_team: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    season_stats = season_stats or {}
    prior_stats = prior_stats or {}
    team_context = team_context or {}
    prior_team = prior_team or {}
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
        if not math.isfinite(goal_rate_f):
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
        if not math.isfinite(def_mult_f):
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

        expected = None
        if stat_bundle:
            try:
                expected = predict_expected_stats(
                    sorted_rows,
                    stats if isinstance(stats, dict) else {},
                    prior_stats.get(clean_name(name), {}) if name else {},
                    team_context.get(team, {}) if team else {},
                    team_context.get(opp, {}) if opp else {},
                    prior_team.get(team, {}) if team else {},
                    prior_team.get(opp, {}) if opp else {},
                    gp_season=int((stats or {}).get("gp", 0) or live_gp or 0),
                    is_home=bool(home),
                    is_defense=str(position).upper().startswith("D"),
                    bundle=stat_bundle,
                )
            except Exception:
                expected = None

        gp = int(stats.get("gp", 0) or 0) if isinstance(stats, dict) else 0
        goals = int(stats.get("goals", 0) or 0) if isinstance(stats, dict) else 0
        assists = int(stats.get("assists", 0) or 0) if isinstance(stats, dict) else 0
        points = int(stats.get("points", 0) or 0) if isinstance(stats, dict) else 0
        sog = int(stats.get("sog", 0) or 0) if isinstance(stats, dict) else 0
        toi_pg = float(stats.get("toi_pg", 0) or 0) if isinstance(stats, dict) else 0.0
        if not math.isfinite(toi_pg):
            toi_pg = 0.0
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
        if expected:
            player_obj["expected"] = expected
        players.append(player_obj)

    players.sort(key=lambda p: (-float((p.get("tonight") or {}).get("goal_rate") or 0.0), str(p.get("name") or "")))
    players = players[:600]

    if updated_utc is None:
        updated_utc = datetime.now(timezone.utc).isoformat()
    elif hasattr(updated_utc, "isoformat"):
        updated_utc = updated_utc.isoformat()
    else:
        updated_utc = str(updated_utc)

    return {
        "updated_utc": updated_utc,
        "model_version": {
            "name": "xgboost-poisson-v1",
            "trained_utc": (stat_bundle or {}).get("trained_utc"),
            "train_through": (stat_bundle or {}).get("train_through"),
        } if stat_bundle else None,
        "players": players,
    }
