"""Backward-induction optimal-stopping advisor for Tim Hortons Challenge pick timing."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np

from src.data_sync import PlayerMatcher, clean_name, normalize_team_abbrev

DTD_PLAY_PROB = 0.65


def poisson_goal_prob(rate: float) -> float:
    """Convert expected goals-per-game rate to anytime 1+ goal probability."""
    return 1.0 - math.exp(-max(0.0, float(rate)))


def effective_goal_prob(player: dict[str, Any]) -> float:
    """Anytime 1+ goal probability discounted by DTD play probability when applicable."""
    prob = poisson_goal_prob(float(player.get("final_score", 0.0)))
    return prob * (DTD_PLAY_PROB if bool(player.get("injury_dtd")) else 1.0)


def _parse_utc_ts(utc_str: str | None) -> float | None:
    if not utc_str or not isinstance(utc_str, str):
        return None
    try:
        return datetime.fromisoformat(utc_str.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _group_medians(groups: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Median projected score of each contest group, used to tier late entrants."""
    medians: dict[str, float] = {}
    for g_key, lst in groups.items():
        scores = sorted(float(p.get("final_score", 0.0)) for p in lst)
        medians[str(g_key)] = float(np.median(scores)) if scores else 0.0
    return medians


def _group_for_score(score: float, medians: dict[str, float]) -> str:
    """Map a prospective late entrant to the group whose typical score it matches.

    The contest never moves a skater between groups, so a simulated Group 2 or 3
    pick must not be allowed to take a Group 1 calibre player.
    """
    return min(medians, key=lambda g: abs(score - medians[g]))


def build_group_ticket(
    groups: dict[str, list[dict[str, Any]]],
    *,
    team_wave_idx: dict[str, int],
    wave_labels: list[str],
    allowed_teams: set[str],
) -> dict[str, dict[str, Any]]:
    """Best pickable player in each group right now.

    Only players on teams that have not started are eligible. Nothing here
    predicts who might appear in a later wave: the ticket reflects the pool as
    it exists at the moment of evaluation.
    """
    allowed_norm = {normalize_team_abbrev(t) for t in (allowed_teams or set())}
    result: dict[str, dict[str, Any]] = {}
    for g_key in sorted((groups or {}).keys(), key=lambda x: int(x)):
        lst = groups.get(g_key) or []
        if not lst:
            continue

        def _norm_team(p: dict[str, Any]) -> str:
            return normalize_team_abbrev(p.get("team"))

        eligible = [
            p
            for p in lst
            if _norm_team(p) in allowed_norm and not bool(p.get("injury_out"))
        ]
        if not eligible:
            # No healthy player is still available; fall back to any unstarted
            # player, then to the full group once every game has started.
            eligible = [p for p in lst if _norm_team(p) in allowed_norm]
        if not eligible:
            eligible = [p for p in lst if not bool(p.get("injury_out"))] or list(lst)

        pick = max(eligible, key=effective_goal_prob)
        team = _norm_team(pick)
        w_idx = int(team_wave_idx.get(team, 0))
        p_g = effective_goal_prob(pick)
        wave_et = wave_labels[w_idx] if 0 <= w_idx < len(wave_labels) else "--"
        result[str(g_key)] = {
            "player": str(pick.get("player", "")),
            "team": team,
            "opponent": normalize_team_abbrev(pick.get("opponent")),
            "position": str(pick.get("position", "F")),
            "final_score": round(float(pick.get("final_score", 0.0)), 3),
            "p_goal_pct": round(p_g * 100.0, 1),
            "wave_idx": w_idx + 1,
            "wave_et": wave_et if wave_et is not None else "--",
            "injury_status": pick.get("injury_status"),
            "injury_dtd": bool(pick.get("injury_dtd")),
        }
    return result


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
            "selection": {},
            "lock_basis": "unavailable",
            "first_wave_utc": "",
            "selection_lock_utc": "",
            "selection_lock_et": "",
            "selection_wave": 0,
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
        cur_best = build_group_ticket(
            ordered_groups,
            team_wave_idx={},
            wave_labels=[],
            allowed_teams=set(),
        )
        v_now = round(sum(effective_goal_prob(p) for p in cur_best.values()), 4)
        return {
            "status": "all_started",
            "decision": "SELECT_NOW",
            "headline": "All Today's Games Started or Final",
            "detail": "No future puck-drop waves remain on today's schedule.",
            "current_best": cur_best,
            "selection": cur_best,
            "lock_basis": "now",
            "first_wave_utc": "",
            "selection_lock_utc": "",
            "selection_lock_et": "",
            "selection_wave": 1,
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
    wave_labels = [str(w.get("start_time_et", "--")) for w in sorted_waves]
    cur_best: dict[str, dict[str, Any]] = build_group_ticket(
        ordered_groups,
        team_wave_idx=team_wave_idx,
        wave_labels=wave_labels,
        allowed_teams=unstarted_teams,
    )
    first_wave_utc = datetime.fromtimestamp(
        float(sorted_waves[0]["start_ts"]), tz=timezone.utc
    ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    v_now = sum(effective_goal_prob(p) for p in cur_best.values())
    locked_if_defer_1 = sum(1 for p in cur_best.values() if int(p.get("wave_idx", 0)) == 1)

    if len(sorted_waves) == 1:
        w0 = sorted_waves[0]
        total_visible = sum(len(lst) for lst in ordered_groups.values())
        return {
            "status": "single_wave",
            "decision": "SELECT_NOW",
            "headline": f"Select Current Best Now (Single Wave at {w0['start_time_et']})",
            "detail": "All remaining games start together; waiting adds no new players.",
            "current_best": cur_best,
            "selection": cur_best,
            "lock_basis": "now",
            "first_wave_utc": first_wave_utc,
            "selection_lock_et": w0["start_time_et"],
            "selection_lock_utc": first_wave_utc,
            "selection_wave": 1,
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
                    "visible_carried": total_visible,
                    "hidden_available": 0,
                    "unlisted_scorers": [],
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

    # Prospective late entrants inherit the contest's tier, so bucket them by
    # the group whose typical score they match. They never migrate between
    # groups in the real contest, and neither should the simulation.
    group_medians = _group_medians(ordered_groups)
    hidden_by_group: dict[str, list[dict[str, Any]]] = {g: [] for g in ordered_groups}
    for h in hidden_pool:
        hidden_by_group[_group_for_score(h["final_score"], group_medians)].append(h)
    for lst in hidden_by_group.values():
        lst.sort(key=lambda x: x["final_score"], reverse=True)

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
            "unlisted_scorers": [
                {
                    "player": h["player"],
                    "team": h["team"],
                    "final_score": h["final_score"],
                    "p_goal_pct": h["p_goal_pct"],
                    "injury_status": h.get("injury_status"),
                }
                for h in hidden_pool
                if h["wave_idx"] == 0
            ][:4],
            "is_current": True,
        }
    ]

    best_future_wave: dict[str, Any] | None = None
    for k in range(1, len(sorted_waves)):
        carried = [p for p in tims_flat if p["wave_idx"] >= k]
        rem_hidden = [h for h in hidden_pool if h["wave_idx"] >= k]
        wave_unlisted = [
            {
                "player": h["player"],
                "team": h["team"],
                "final_score": h["final_score"],
                "p_goal_pct": h["p_goal_pct"],
                "injury_status": h.get("injury_status"),
            }
            for h in hidden_pool
            if h["wave_idx"] == k
        ][:4]
        if len(carried) + len(rem_hidden) < 3:
            continue

        # Stay inside each group: the ticket at wave k is one skater per group,
        # picked from the players still listed then plus a few possible late
        # additions in that same group. Re-sorting the whole pool and re-slicing
        # it into equal groups would let one ticket take two Group 1 stars.
        carried_by_group = {
            g: [p for p in carried if str(p["group"]) == str(g)] for g in ordered_groups
        }
        hidden_by_group_k = {
            g: [h for h in hidden_by_group.get(g, []) if h["wave_idx"] >= k] for g in ordered_groups
        }
        sim_evs: list[float] = []
        for _ in range(mc_trials):
            ev_k = 0.0
            for g in ordered_groups:
                pool_g = [
                    p
                    for p in carried_by_group[g]
                    if (not p.get("injury_dtd") or rng.random() < DTD_PLAY_PROB)
                ]
                # A wave rebuild tends to surface a small number of skaters
                # from the remaining games. History shows 0-4 per wave, so give
                # each group 0-2 and keep them in their own tier.
                n_new = int(rng.integers(0, 3))
                candidates = [
                    h
                    for h in hidden_by_group_k[g]
                    if (not h.get("injury_dtd") or rng.random() < DTD_PLAY_PROB)
                ]
                if n_new and candidates:
                    ranked = sorted(
                        candidates,
                        key=lambda x: x["final_score"] + float(rng.normal(0.0, 0.08)),
                        reverse=True,
                    )
                    pool_g.extend(ranked[:n_new])
                if not pool_g:
                    continue
                ev_k += poisson_goal_prob(max(pool_g, key=lambda x: x["final_score"])["final_score"])
            sim_evs.append(ev_k)

        if not sim_evs:
            continue
        arr = np.asarray(sim_evs, dtype=float)
        est_ev = float(np.mean(arr))
        delta = est_ev - v_now
        beat_pct = float(np.mean(arr > v_now) * 100.0)
        projected_ticket: dict[str, dict[str, Any]] = {}
        for g in ordered_groups:
            healthy = [p for p in carried_by_group[g] if not p.get("injury_dtd")]
            pool_proj = healthy or carried_by_group[g]
            if not pool_proj:
                continue
            top = max(pool_proj, key=effective_goal_prob)
            projected_ticket[g] = {
                "player": top["player"],
                "team": top["team"],
                "group": g,
                "final_score": round(float(top["final_score"]), 3),
            }

        w_rec = {
            "wave": k + 1,
            "start_time_et": sorted_waves[k]["start_time_et"],
            "matchups": sorted_waves[k]["matchups"],
            "est_ev": round(est_ev, 4),
            "ev_delta": round(delta, 4),
            "beat_now_pct": round(beat_pct, 1),
            "visible_carried": len(carried),
            "hidden_available": len(rem_hidden),
            "projected_ticket": projected_ticket,
            "unlisted_scorers": wave_unlisted,
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
            f"({best_future_wave['beat_now_pct']:.0f}% of simulations) if late-game additions "
            f"upgrade the remaining group ticket."
        )
    elif best_future_wave is not None and ev_delta > 0.0:
        decision = "BORDERLINE_WATCH"
        target_et = best_future_wave["start_time_et"]
        headline = f"Select Now — Lock Current Ticket Before {sorted_waves[0]['start_time_et']}"
        detail = (
            f"Current best ticket ({v_now:.3f} xG) is near-optimal; deferring to {target_et} projects only "
            f"+{ev_delta:.3f} xG ({best_future_wave['beat_now_pct']:.0f}% of sims) even if unlisted late "
            f"scorers enter, so lock the current ticket."
        )
    else:
        decision = "SELECT_NOW"
        headline = f"Select Current Best Now (Before {sorted_waves[0]['start_time_et']})"
        detail = (
            f"Deferring past {sorted_waves[0]['start_time_et']} loses {abs(ev_delta):.3f} xG by shrinking the top-tier pool"
            if best_future_wave is not None
            else "Current pool captures maximum available scoring value."
        )

    # Never predict a player who is not on the board yet. The ticket is always
    # the best selection available right now; the lock time only says when the
    # tracker should stop refreshing it. On a deferral, the wave-1 games start
    # before the lock, so the last refresh naturally drops those players and
    # records the ticket that is actually in effect at the advised time.
    deferred = best_future_wave is not None and ev_delta > defer_threshold
    selection = cur_best
    lock_idx = (int(best_future_wave["wave"]) - 1) if deferred else 0
    selection_lock_et = sorted_waves[lock_idx]["start_time_et"]
    selection_lock_utc = datetime.fromtimestamp(
        float(sorted_waves[lock_idx]["start_ts"]), tz=timezone.utc
    ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    selection_wave = lock_idx + 1
    lock_basis = "deferred" if deferred else "now"

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
        "selection": selection,
        "lock_basis": lock_basis,
        "first_wave_utc": first_wave_utc,
        "selection_lock_utc": selection_lock_utc,
        "selection_lock_et": selection_lock_et,
        "selection_wave": selection_wave,
        "hidden_upside_players": hidden_later_upside,
        "g1_best_score": round(g1_score, 3),
        "g2_best_score": round(g2_score, 3),
        "waves": wave_summaries,
    }
