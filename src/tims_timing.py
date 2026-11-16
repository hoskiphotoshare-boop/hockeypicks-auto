"""Backward-induction optimal-stopping advisor for Tim Hortons Challenge pick timing."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import numpy as np

from src.data_sync import PlayerMatcher, clean_name, normalize_team_abbrev


def poisson_goal_prob(rate: float) -> float:
    """Convert expected goals-per-game rate to anytime 1+ goal probability."""
    return 1.0 - math.exp(-max(0.0, float(rate)))


def _parse_utc_ts(utc_str: str | None) -> float | None:
    if not utc_str or not isinstance(utc_str, str):
        return None
    try:
        return datetime.fromisoformat(utc_str.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _split_groups_proportional(pool: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Slice an unstarted pool into Groups 1, 2, 3 preserving Tims 10/12/14 ratios."""
    n = len(pool)
    if n >= 36:
        return {1: pool[:10], 2: pool[10:22], 3: pool[22:36]}
    n1 = max(2, round(n * (10.0 / 36.0)))
    n2 = max(2, round(n * (12.0 / 36.0)))
    if n1 + n2 >= n - 1:
        n1 = max(1, n // 3)
        n2 = max(1, (2 * n) // 3 - n1)
    return {1: pool[:n1], 2: pool[n1 : n1 + n2], 3: pool[n1 + n2 :]}


def compute_tims_deferral_advice(
    *,
    tims_groups: dict[str, list[dict[str, Any]]],
    slate_candidates: list[dict[str, Any]],
    schedule_games: list[dict[str, Any]],
    defer_threshold: float = 0.020,
    mc_trials: int = 300,
) -> dict[str, Any]:
    """
    Evaluate whether to select current Tims picks now or selectively defer to a future wave.

    Uses full-information backward induction (Monte Carlo refill over unstarted hidden skaters)
    and always returns the best immediately available Group 1-3 picks alongside the advice.
    """
    ordered_groups = {
        str(k): sorted(
            list(v),
            key=lambda x: (not bool(x.get("injury_out")), float(x.get("final_score", 0.0))),
            reverse=True,
        )
        for k, v in sorted(tims_groups.items(), key=lambda kv: int(kv[0]))
        if v
    }
    if not ordered_groups:
        return {
            "status": "no_tims_pool",
            "decision": "UNAVAILABLE",
            "headline": "Awaiting Tim Hortons Player Lists",
            "detail": "Upstream Tim Hortons groups are not posted yet.",
            "current_best": {},
            "v_now": 0.0,
            "v_defer_best": 0.0,
            "ev_delta": 0.0,
            "hidden_upside_players": [],
            "waves": [],
        }

    # Build chronological waves from unstarted today games (or tomorrow's slate after today's finish)
    tims_teams = {
        normalize_team_abbrev(p.get("team"))
        for lst in ordered_groups.values()
        for p in lst
        if p.get("team")
    }
    unstarted = [
        g
        for g in (schedule_games or [])
        if g.get("is_today", True) and str(g.get("game_state", "FUT")) in ("FUT", "PRE")
    ]
    if not unstarted and tims_teams:
        unstarted = [
            g
            for g in (schedule_games or [])
            if str(g.get("game_state", "FUT")) in ("FUT", "PRE")
            and (
                normalize_team_abbrev(g.get("away")) in tims_teams
                or normalize_team_abbrev(g.get("home")) in tims_teams
            )
        ]
    waves_map: dict[tuple[float, str], dict[str, Any]] = {}
    for g in unstarted:
        ts = _parse_utc_ts(g.get("start_time_utc"))
        if ts is None:
            continue
        et_lbl = str(g.get("start_time_et") or "TBD")
        key = (ts, et_lbl)
        w = waves_map.setdefault(
            key,
            {"start_ts": ts, "start_time_et": et_lbl, "matchups": [], "teams": set()},
        )
        away = normalize_team_abbrev(g.get("away"))
        home = normalize_team_abbrev(g.get("home"))
        w["matchups"].append(f"{away} @ {home}")
        w["teams"].add(away)
        w["teams"].add(home)

    sorted_waves = [waves_map[k] for k in sorted(waves_map.keys(), key=lambda kv: kv[0])]
    if not sorted_waves:
        cur_best = {
            g_k: {
                "player": lst[0]["player"],
                "team": lst[0]["team"],
                "final_score": round(float(lst[0]["final_score"]), 3),
                "p_goal_pct": round(poisson_goal_prob(lst[0]["final_score"]) * 100.0, 1),
            }
            for g_k, lst in ordered_groups.items()
        }
        v_now = round(sum(poisson_goal_prob(lst[0]["final_score"]) for lst in ordered_groups.values()), 4)
        return {
            "status": "all_started",
            "decision": "SELECT_NOW",
            "headline": "All Today's Games Started or Final",
            "detail": "No future puck-drop waves remain on today's schedule.",
            "current_best": cur_best,
            "v_now": v_now,
            "v_defer_best": v_now,
            "ev_delta": 0.0,
            "hidden_upside_players": [],
            "waves": [],
        }

    team_wave_idx = {}
    for idx, w in enumerate(sorted_waves):
        for t in w["teams"]:
            team_wave_idx[t] = idx

    # Best immediately available pick in each group (filtered to unstarted teams when matched)
    unstarted_teams = {t for w in sorted_waves for t in w["teams"]}
    cur_best: dict[str, dict[str, Any]] = {}
    v_now = 0.0
    locked_if_defer_1 = 0
    for g_k, lst in ordered_groups.items():
        elig_grp = [p for p in lst if normalize_team_abbrev(p.get("team")) in unstarted_teams] or lst
        healthy_grp = [p for p in elig_grp if not p.get("injury_out")] or elig_grp
        best = max(healthy_grp, key=lambda x: float(x.get("final_score", 0.0)))
        w_idx = team_wave_idx.get(normalize_team_abbrev(best.get("team")), 0)
        if w_idx == 0:
            locked_if_defer_1 += 1
        p_g = poisson_goal_prob(float(best.get("final_score", 0.0)))
        v_now += p_g
        cur_best[g_k] = {
            "player": str(best.get("player", "")),
            "team": normalize_team_abbrev(best.get("team")),
            "opponent": normalize_team_abbrev(best.get("opponent")),
            "final_score": round(float(best.get("final_score", 0.0)), 3),
            "p_goal_pct": round(p_g * 100.0, 1),
            "wave_idx": w_idx + 1,
            "wave_et": sorted_waves[w_idx]["start_time_et"] if w_idx < len(sorted_waves) else "--",
            "injury_status": best.get("injury_status"),
        }

    if len(sorted_waves) == 1:
        w0 = sorted_waves[0]
        return {
            "status": "single_wave",
            "decision": "SELECT_NOW",
            "headline": f"Select Current Best Now (Single Wave at {w0['start_time_et']})",
            "detail": "All remaining games start together; waiting adds no new players.",
            "current_best": cur_best,
            "v_now": round(v_now, 4),
            "v_defer_best": round(v_now, 4),
            "ev_delta": 0.0,
            "hidden_upside_players": [],
            "waves": [
                {
                    "wave": 1,
                    "start_time_et": w0["start_time_et"],
                    "matchups": w0["matchups"],
                    "est_ev": round(v_now, 4),
                    "ev_delta": 0.0,
                    "beat_now_pct": 0.0,
                    "is_current": True,
                }
            ],
        }

    # Build PlayerMatcher over currently visible Tims players to identify genuinely unlisted late skaters
    tims_flat: list[dict[str, Any]] = []
    tims_entries: list[dict[str, Any]] = []
    for g_k, lst in ordered_groups.items():
        for p in lst:
            if p.get("injury_out"):
                continue
            t_norm = normalize_team_abbrev(p.get("team"))
            rec = {
                "player": str(p.get("player", "")),
                "team": t_norm,
                "opponent": normalize_team_abbrev(p.get("opponent")),
                "position": str(p.get("position", "F")),
                "final_score": float(p.get("final_score", 0.0)),
                "group": int(g_k),
                "wave_idx": team_wave_idx.get(t_norm, 0),
                "injury_dtd": bool(p.get("injury_dtd")),
            }
            tims_flat.append(rec)
            tims_entries.append(
                {
                    "player_id": 0,
                    "name": rec["player"],
                    "name_clean": clean_name(rec["player"]),
                    "team": t_norm,
                    "position": rec["position"],
                    "payload": rec,
                }
            )
    tims_matcher = PlayerMatcher(tims_entries)

    hidden_pool: list[dict[str, Any]] = []
    seen_hidden: set[tuple[str, str]] = set()
    for s in slate_candidates or []:
        if s.get("injury_out"):
            continue
        t_norm = normalize_team_abbrev(s.get("team"))
        if t_norm not in unstarted_teams:
            continue
        p_name = str(s.get("player", ""))
        if not p_name:
            continue
        matched = tims_matcher.match(
            name=p_name,
            team=t_norm,
            position=str(s.get("position", "")),
            allow_initial_fallback=True,
            require_same_team=True,
            require_compatible_pos=True,
        )
        if matched is not None:
            continue
        key = (clean_name(p_name), t_norm)
        if key in seen_hidden:
            continue
        seen_hidden.add(key)
        w_idx = team_wave_idx.get(t_norm, 0)
        hidden_pool.append(
            {
                "player": p_name,
                "team": t_norm,
                "opponent": normalize_team_abbrev(s.get("opponent")),
                "position": str(s.get("position", "F")),
                "final_score": round(float(s.get("final_score", 0.0)), 3),
                "p_goal_pct": round(poisson_goal_prob(float(s.get("final_score", 0.0))) * 100.0, 1),
                "wave_idx": w_idx,
                "wave_et": sorted_waves[w_idx]["start_time_et"] if w_idx < len(sorted_waves) else "--",
                "injury_status": s.get("injury_status"),
                "injury_dtd": bool(s.get("injury_dtd")),
            }
        )
    hidden_pool.sort(key=lambda x: x["final_score"], reverse=True)

    # Identify high-upside hidden skaters in waves > 0 (beating Group 2 best or top-5 later hidden)
    g1_score = float(cur_best.get("1", {}).get("final_score", 0.0))
    g2_score = float(cur_best.get("2", {}).get("final_score", 0.0))
    hidden_later_upside = [
        h for h in hidden_pool if h["wave_idx"] >= 1 and h["final_score"] >= max(0.18, g2_score * 0.90)
    ][:6]

    # Backward-induction continuation EV estimate at each future wave k >= 1
    rng = np.random.default_rng(42)
    wave_summaries: list[dict[str, Any]] = [
        {
            "wave": 1,
            "start_time_et": sorted_waves[0]["start_time_et"],
            "matchups": sorted_waves[0]["matchups"],
            "est_ev": round(v_now, 4),
            "ev_delta": 0.0,
            "beat_now_pct": 0.0,
            "visible_carried": sum(1 for p in tims_flat if p["wave_idx"] >= 0),
            "hidden_available": len(hidden_pool),
            "is_current": True,
        }
    ]

    best_future_wave: dict[str, Any] | None = None
    for k in range(1, len(sorted_waves)):
        carried = [p for p in tims_flat if p["wave_idx"] >= k]
        rem_hidden = [h for h in hidden_pool if h["wave_idx"] >= k]
        total_rem = len(carried) + len(rem_hidden)
        if total_rem < 15:
            continue

        # Realistic refill target: preserve carried slots and fill vacated slots from remaining rosters
        max_fill = min(36, len(carried) + len(rem_hidden))
        sim_evs: list[float] = []
        if not rem_hidden:
            if len(carried) >= 12:
                ordered_c = sorted(carried, key=lambda x: x["final_score"], reverse=True)
                grps = _split_groups_proportional(ordered_c[:36])
                ev_k = sum(
                    poisson_goal_prob(max(grp, key=lambda x: x["final_score"])["final_score"])
                    for grp in grps.values()
                )
                sim_evs = [ev_k]
        else:
            min_fill = min(max_fill, len(carried) + min(len(rem_hidden), max(1, round(0.55 * (max_fill - len(carried))))))
            for _ in range(mc_trials):
                target_n = int(rng.integers(min_fill, max_fill + 1)) if max_fill > min_fill else max_fill
                active_hidden = [h for h in rem_hidden if (not h.get("injury_dtd") or rng.random() < 0.65)]
                need = max(0, target_n - len(carried))
                added = sorted(
                    active_hidden,
                    key=lambda x: x["final_score"] + float(rng.normal(0.0, 0.08)),
                    reverse=True,
                )[:need]
                active_carried = [c for c in carried if (not c.get("injury_dtd") or rng.random() < 0.65)]
                pool_src = active_carried + added
                if len(pool_src) < 12:
                    continue
                combined = sorted(
                    pool_src,
                    key=lambda x: x["final_score"] + float(rng.normal(0.0, 0.08)),
                    reverse=True,
                )
                grps = _split_groups_proportional(combined)
                ev_k = sum(
                    poisson_goal_prob(max(grp, key=lambda x: x["final_score"])["final_score"])
                    for grp in grps.values()
                )
                sim_evs.append(ev_k)

        if not sim_evs:
            continue
        arr = np.asarray(sim_evs, dtype=float)
        est_ev = float(np.mean(arr))
        delta = est_ev - v_now
        beat_pct = float(np.mean(arr > v_now) * 100.0)
        w_rec = {
            "wave": k + 1,
            "start_time_et": sorted_waves[k]["start_time_et"],
            "matchups": sorted_waves[k]["matchups"],
            "est_ev": round(est_ev, 4),
            "ev_delta": round(delta, 4),
            "beat_now_pct": round(beat_pct, 1),
            "visible_carried": len(carried),
            "hidden_available": len(rem_hidden),
            "is_current": False,
        }
        wave_summaries.append(w_rec)
        if best_future_wave is None or est_ev > best_future_wave["est_ev"]:
            best_future_wave = w_rec

    v_defer_best = best_future_wave["est_ev"] if best_future_wave else v_now
    ev_delta = round(v_defer_best - v_now, 4)

    if best_future_wave is not None and ev_delta > defer_threshold:
        decision = "SELECT_DEFER"
        target_et = best_future_wave["start_time_et"]
        headline = f"Selective Defer — Wait Until Before {target_et}"
        detail = (
            f"Waiting past {sorted_waves[0]['start_time_et']} projects +{ev_delta:.3f} xG "
            f"({best_future_wave['beat_now_pct']:.0f}% win chance) as unlisted late-wave skaters backfill {len(tims_flat) - best_future_wave['visible_carried']} vacated spots."
        )
    elif best_future_wave is not None and ev_delta > 0.0:
        decision = "BORDERLINE_WATCH"
        target_et = best_future_wave["start_time_et"]
        headline = f"Select Now (Borderline Defer Watch Before {target_et})"
        detail = (
            f"Current best ticket ({v_now:.3f} xG) is near-optimal, though waiting past {sorted_waves[0]['start_time_et']} "
            f"offers slight +{ev_delta:.3f} xG ({best_future_wave['beat_now_pct']:.0f}% chance) upside if unlisted late scorers enter."
        )
    else:
        decision = "SELECT_NOW"
        headline = f"Select Current Best Now (Before {sorted_waves[0]['start_time_et']})"
        detail = (
            f"Deferring past {sorted_waves[0]['start_time_et']} loses {abs(ev_delta):.3f} xG by shrinking the top-tier pool"
            if best_future_wave is not None
            else "Current pool captures maximum available scoring value."
        )

    return {
        "status": "evaluated",
        "decision": decision,
        "headline": headline,
        "detail": detail,
        "defer_threshold": defer_threshold,
        "recommended_wave_et": best_future_wave["start_time_et"] if best_future_wave else sorted_waves[0]["start_time_et"],
        "v_now": round(v_now, 4),
        "v_defer_best": round(v_defer_best, 4),
        "ev_delta": ev_delta,
        "locked_picks_if_defer_1": locked_if_defer_1,
        "current_best": cur_best,
        "hidden_upside_players": hidden_later_upside,
        "g1_best_score": round(g1_score, 3),
        "g2_best_score": round(g2_score, 3),
        "waves": wave_summaries,
    }
