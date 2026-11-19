"""Forward logger and automatic next-day NHL boxscore grader for Tim Hortons picks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.data_sync import (
    PlayerMatcher,
    _fetch_with_retries,
    clean_name,
    create_http_session,
    normalize_team_abbrev,
)


def load_tims_history(path: str) -> dict[str, Any]:
    """Load persisted Tim Hortons Challenge history JSON safely."""
    default = {"days": [], "summary": {}}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("days"), list):
            return default
        data.setdefault("summary", {})
        return data
    except Exception:
        return default


def save_tims_history(history: dict[str, Any], path: str) -> None:
    """Persist Tim Hortons Challenge history JSON atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "player": str(item.get("player", "")),
        "team": str(item.get("team", "UNK")),
        "opponent": str(item.get("opponent", "UNK")),
        "position": str(item.get("position", "F")),
        "base_pred": round(float(item.get("base_pred", 0.0)), 3),
        "live_gp": int(item.get("live_gp", 0)),
        "live_goals": int(item.get("live_goals", 0)),
        "def_multiplier": round(float(item.get("def_multiplier", 1.0)), 3),
        "final_score": round(float(item.get("final_score", 0.0)), 3),
    }


def _resolve_pick(groups_raw: dict[str, Any], group_key: str, pick: dict[str, Any]) -> dict[str, Any]:
    grp = []
    if isinstance(groups_raw, dict):
        grp = groups_raw.get(group_key) or groups_raw.get(str(group_key)) or []
    target_team = normalize_team_abbrev(pick.get("team"))
    for cand in grp:
        if str(cand.get("player", "")) == str(pick.get("player", "")) and normalize_team_abbrev(cand.get("team")) == target_team:
            return _compact_candidate(cand)
    return {
        "player": str(pick.get("player", "")),
        "team": normalize_team_abbrev(pick.get("team")),
        "opponent": normalize_team_abbrev(pick.get("opponent")),
        "position": str(pick.get("position", "F")),
        "base_pred": round(float(pick.get("final_score", pick.get("base_pred", 0.0))), 3),
        "live_gp": 0,
        "live_goals": 0,
        "def_multiplier": 1.0,
        "final_score": round(float(pick.get("final_score", 0.0)), 3),
    }


def upsert_tims_snapshot(
    history: dict[str, Any],
    *,
    date_str: str,
    context: dict[str, Any],
    schedule_info: dict[str, Any],
    skater_bundle: dict[str, Any],
    tims_groups: dict[str, list[dict[str, Any]]],
    captured_et: str,
    timing_advice: dict[str, Any] | None = None,
    now_utc: datetime | None = None,
) -> bool:
    """Record or refresh today's unsettled Tim Hortons Group 1-3 AI rankings."""
    if not tims_groups:
        return False

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    days: list[dict[str, Any]] = history.setdefault("days", [])
    existing_idx = next((i for i, d in enumerate(days) if d.get("date") == date_str), None)
    if existing_idx is not None and days[existing_idx].get("settled"):
        return False

    # Freeze rule for already-locked records.
    if existing_idx is not None:
        existing = days[existing_idx]
        if existing.get("picks_locked") is True:
            return False
        existing_timing = existing.get("timing") or {}
        lock_str = existing_timing.get("lock_utc")
        if lock_str:
            try:
                lock_dt = datetime.fromisoformat(str(lock_str).replace("Z", "+00:00"))
            except ValueError:
                lock_dt = None
            if lock_dt is not None and now_utc >= lock_dt:
                existing["picks_locked"] = True
                if not existing.get("locked_et"):
                    existing["locked_et"] = str(existing.get("captured_et") or "")
                return False

    groups_compact = {
        str(k): [_compact_candidate(c) for c in grp]
        for k, grp in sorted(tims_groups.items(), key=lambda kv: int(kv[0]))
        if grp
    }
    if not groups_compact:
        return False

    selection: dict[str, Any] = {}
    if isinstance(timing_advice, dict):
        selection = timing_advice.get("selection") or {}

    if selection:
        ai_picks: dict[str, Any] = {}
        for key in groups_compact:
            sel_pick = selection.get(key) if isinstance(selection, dict) else None
            if sel_pick:
                ai_picks[key] = _resolve_pick(tims_groups, key, sel_pick)
            else:
                ai_picks[key] = groups_compact[key][0]
    else:
        ai_picks = {k: grp[0] for k, grp in groups_compact.items()}

    # Parse advice lock time.
    lock_str = ""
    if isinstance(timing_advice, dict):
        lock_str = timing_advice.get("selection_lock_utc") or ""
    lock_dt = None
    if lock_str:
        try:
            lock_dt = datetime.fromisoformat(str(lock_str).replace("Z", "+00:00"))
        except ValueError:
            lock_dt = None
    if lock_str:
        picks_locked = bool(lock_dt is not None and now_utc >= lock_dt)
    else:
        # No future wave is left to wait for (e.g. every game has started), so the
        # ticket is final at capture. Records without any advice keep refreshing.
        picks_locked = bool(isinstance(timing_advice, dict) and selection)
    locked_et = captured_et if picks_locked else ""

    timing_obj: dict[str, Any] = {}
    if isinstance(timing_advice, dict):
        timing_obj = {
            "decision": timing_advice.get("decision"),
            "basis": timing_advice.get("selection_basis"),
            "lock_utc": timing_advice.get("selection_lock_utc"),
            "lock_et": timing_advice.get("selection_lock_et"),
            "recommended_wave_et": timing_advice.get("recommended_wave_et"),
            "v_now": timing_advice.get("v_now"),
            "v_defer_best": timing_advice.get("v_defer_best"),
            "ev_delta": timing_advice.get("ev_delta"),
        }

    record = {
        "date": date_str,
        "season": str(context.get("nhl_season", "")),
        "game_type": int(schedule_info.get("active_game_type", 2)),
        "mode": str(schedule_info.get("active_mode", "Regular Season")),
        "captured_et": captured_et,
        "model_n_half": float(skater_bundle.get("n_half", 0.0)),
        "groups": groups_compact,
        "ai_picks": ai_picks,
        "settled": False,
        "results": None,
        "timing": timing_obj,
        "picks_locked": picks_locked,
        "locked_et": locked_et,
    }
    if existing_idx is not None:
        days[existing_idx] = record
    else:
        days.append(record)
    return True


def _extract_boxscore_goal_entries(
    payload: dict[str, Any],
    *,
    away_team: str = "",
    home_team: str = "",
) -> list[dict[str, Any]]:
    """Extract normalized skater goal records (with team/position/playerId) from an NHL boxscore."""
    entries: list[dict[str, Any]] = []
    pstats = payload.get("playerByGameStats") or {}
    resolved_away = normalize_team_abbrev(away_team or (payload.get("awayTeam") or {}).get("abbrev"))
    resolved_home = normalize_team_abbrev(home_team or (payload.get("homeTeam") or {}).get("abbrev"))
    for side_key, side_team in (("awayTeam", resolved_away), ("homeTeam", resolved_home)):
        side = pstats.get(side_key) or {}
        for pos_group in ("forwards", "defense"):
            for player in side.get(pos_group, []):
                name_obj = player.get("name") or {}
                raw_name = name_obj.get("default", "") if isinstance(name_obj, dict) else str(name_obj)
                key = clean_name(raw_name)
                if not key:
                    continue
                goals = int(player.get("goals", 0) or 0)
                pos_code = str(player.get("position", "") or ("D" if pos_group == "defense" else "F"))
                entries.append(
                    {
                        "player_id": int(player.get("playerId", 0) or 0),
                        "name": raw_name,
                        "name_clean": key,
                        "team": side_team,
                        "position": pos_code,
                        "payload": {"goals": goals, "team": side_team, "position": pos_code},
                    }
                )
    return entries


def _extract_boxscore_goal_map(payload: dict[str, Any]) -> dict[str, int]:
    """Extract {clean_name: goals} across dressed skaters from an NHL boxscore payload."""
    goals_by_name: dict[str, int] = {}
    for entry in _extract_boxscore_goal_entries(payload):
        key = entry["name_clean"]
        goals = int(entry["payload"]["goals"])
        goals_by_name[key] = max(goals_by_name.get(key, 0), goals)
    return goals_by_name


def _fetch_finished_date_goals(
    entry_date: str,
    entry_teams: set[str],
    session: Any,
) -> PlayerMatcher | None:
    """Return a PlayerMatcher of boxscore goals if all relevant NHL games on entry_date are finished, else None."""
    sched_url = f"https://api-web.nhle.com/v1/schedule/{entry_date}"
    res = _fetch_with_retries(session, sched_url, timeout=12, attempts=2)
    if res is None or res.status_code != 200:
        return None

    relevant_games = []
    for week in res.json().get("gameWeek", []):
        if week.get("date") != entry_date:
            continue
        for game in week.get("games", []):
            away = normalize_team_abbrev(game.get("awayTeam", {}).get("abbrev"))
            home = normalize_team_abbrev(game.get("homeTeam", {}).get("abbrev"))
            if not entry_teams or away in entry_teams or home in entry_teams:
                relevant_games.append((game, away, home))

    if not relevant_games:
        return None
    if any(game.get("gameState") in ("FUT", "PRE", "LIVE", "CRIT") for game, _, _ in relevant_games):
        return None

    finished_games = [
        (int(game.get("id", 0) or 0), away, home)
        for game, away, home in relevant_games
        if game.get("gameState") in ("FINAL", "OFF")
    ]
    if not finished_games:
        return None

    all_entries: list[dict[str, Any]] = []
    for gid, away, home in finished_games:
        if gid <= 0:
            continue
        box_url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore"
        bres = _fetch_with_retries(session, box_url, timeout=12, attempts=2)
        if bres is None or bres.status_code != 200:
            return None
        all_entries.extend(_extract_boxscore_goal_entries(bres.json(), away_team=away, home_team=home))
    return PlayerMatcher(all_entries)


def settle_pending_tims_days(
    history: dict[str, Any],
    *,
    today_str: str,
    settled_et: str,
    session: Any | None = None,
    allow_network: bool = True,
) -> int:
    """Grade unsettled Tim Hortons snapshot entries once all relevant games finish."""
    if not allow_network:
        return 0
    http = session or create_http_session()
    settled_count = 0

    for entry in history.get("days", []):
        if entry.get("settled"):
            continue
        entry_date = str(entry.get("date", ""))
        if not entry_date or entry_date > today_str:
            continue

        groups: dict[str, list[dict[str, Any]]] = entry.get("groups") or {}
        entry_teams = {
            normalize_team_abbrev(cand.get("team"))
            for grp in groups.values()
            for cand in grp
            if cand.get("team") and cand.get("team") != "UNK"
        }
        goal_matcher = _fetch_finished_date_goals(entry_date, entry_teams, http)
        if goal_matcher is None:
            continue

        ai_picks: dict[str, dict[str, Any]] = entry.get("ai_picks") or {}
        group_hits: dict[str, dict[str, Any]] = {}
        group_random_probs: dict[str, float] = {}
        player_goals: dict[str, int] = {}

        for g_key, grp in groups.items():
            grp_scored = 0
            for cand in grp:
                p_name = str(cand.get("player", ""))
                p_team = str(cand.get("team", "") or "")
                p_pos = str(cand.get("position", "") or "")
                matched = goal_matcher.match(
                    name=p_name,
                    team=p_team,
                    position=p_pos,
                    allow_initial_fallback=True,
                    require_same_team=True,
                    require_compatible_pos=True,
                )
                goals = int(matched.get("goals", 0)) if matched is not None else 0
                player_goals[p_name] = goals
                if goals >= 1:
                    grp_scored += 1
            group_random_probs[g_key] = round(grp_scored / len(grp), 4) if grp else 0.0

            pick = ai_picks.get(g_key) or (grp[0] if grp else None)
            if pick is not None:
                p_name = str(pick.get("player", ""))
                p_team = str(pick.get("team", "") or "")
                p_pos = str(pick.get("position", "") or "")
                matched_pick = goal_matcher.match(
                    name=p_name,
                    team=p_team,
                    position=p_pos,
                    allow_initial_fallback=True,
                    require_same_team=True,
                    require_compatible_pos=True,
                )
                goals = int(matched_pick.get("goals", 0)) if matched_pick is not None else 0
                group_hits[g_key] = {
                    "player": p_name,
                    "team": str(pick.get("team", "UNK")),
                    "goals": goals,
                    "scored": goals >= 1,
                }

        hits_of_3 = sum(1 for res in group_hits.values() if res["scored"])
        entry["settled"] = True
        entry["settled_et"] = settled_et
        entry["results"] = {
            "hits_of_3": hits_of_3,
            "total_picks": len(group_hits),
            "group_hits": group_hits,
            "group_random_probs": group_random_probs,
            "player_goals": player_goals,
        }
        settled_count += 1

    return settled_count


def recompute_tims_summary(history: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate live forward-tracking stats across logged Tim Hortons days."""
    days = sorted(history.get("days", []), key=lambda d: str(d.get("date", "")), reverse=True)
    history["days"] = days

    settled = [d for d in days if d.get("settled") and isinstance(d.get("results"), dict)]
    pending = [d for d in days if not d.get("settled")]

    ai_hits = 0
    ai_possible = 0
    rand_expected_sum = 0.0
    hit_1plus = 0
    hit_2plus = 0
    sweep_3 = 0
    grp_totals: dict[str, list[int]] = {"1": [], "2": [], "3": []}

    for d in settled:
        res = d["results"]
        gh = res.get("group_hits") or {}
        gr = res.get("group_random_probs") or {}
        day_hits = int(res.get("hits_of_3", 0))
        day_total = int(res.get("total_picks", len(gh)))

        ai_hits += day_hits
        ai_possible += day_total
        rand_expected_sum += sum(float(v) for v in gr.values())
        if day_hits >= 1:
            hit_1plus += 1
        if day_hits >= 2:
            hit_2plus += 1
        if day_total >= 3 and day_hits == day_total:
            sweep_3 += 1

        for g_key in ("1", "2", "3"):
            if g_key in gh:
                grp_totals[g_key].append(1 if gh[g_key].get("scored") else 0)

    n_settled = len(settled)
    summary = {
        "total_days_logged": len(days),
        "settled_days": n_settled,
        "pending_days": len(pending),
        "ai_hits": ai_hits,
        "ai_possible": ai_possible,
        "ai_hit_rate_pct": round((ai_hits / ai_possible) * 100.0, 1) if ai_possible > 0 else 0.0,
        "random_expected_pct": round((rand_expected_sum / ai_possible) * 100.0, 1) if ai_possible > 0 else 0.0,
        "hit_1plus_days": hit_1plus,
        "hit_2plus_days": hit_2plus,
        "sweep_3_days": sweep_3,
        "group_hit_rates": {
            g_key: round((sum(vals) / len(vals)) * 100.0, 1) if vals else 0.0
            for g_key, vals in grp_totals.items()
        },
        "recent_days": [
            {
                "date": d.get("date"),
                "mode": d.get("mode"),
                "settled": bool(d.get("settled")),
                "timing": d.get("timing") or {},
                "picks_locked": bool(d.get("picks_locked")),
                "hits_of_3": d["results"]["hits_of_3"] if d.get("settled") and d.get("results") else None,
                "picks": {
                    g_key: {
                        "player": p.get("player"),
                        "team": p.get("team"),
                        "score": p.get("final_score"),
                        "scored": (
                            d["results"]["group_hits"].get(g_key, {}).get("scored")
                            if d.get("settled") and d.get("results")
                            else None
                        ),
                    }
                    for g_key, p in sorted((d.get("ai_picks") or {}).items(), key=lambda kv: int(kv[0]))
                },
            }
            for d in days[:7]
        ],
    }
    history["summary"] = summary
    return summary
