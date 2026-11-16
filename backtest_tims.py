"""Leak-free walk-forward backtest for Hockeypicks AI & Tim Hortons Challenge scoring."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data_sync import (
    NHL_TEAMS,
    _fetch_with_retries,
    clean_name,
    create_http_session,
    get_nhl_season_context,
    normalize_team_abbrev,
    sync_game_logs,
    sync_moneypuck_data,
)
from src.skater_model import (
    _read_optional_csv,
    combine_skater_frames,
    normalize_skater_frame,
    prep_dynamic_features,
    train_skater_goal_model,
)
from src.team_game_model import (
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_power_rankings_and_defense,
    train_pairwise_game_model,
)


def parse_toi_seconds(toi_str: Any) -> int:
    """Convert NHL boxscore MM:SS time-on-ice string into integer seconds."""
    if not isinstance(toi_str, str) or ":" not in toi_str:
        return 0
    parts = toi_str.strip().split(":", 1)
    try:
        return (int(parts[0]) * 60) + int(parts[1])
    except ValueError:
        return 0


def extract_boxscore_skaters(payload: dict[str, Any], game_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all dressed skaters (toi > 0) from an NHL v1 boxscore JSON response."""
    pstats = payload.get("playerByGameStats") or {}
    away_team = normalize_team_abbrev(game_meta.get("away_team"))
    home_team = normalize_team_abbrev(game_meta.get("home_team"))
    rows: list[dict[str, Any]] = []

    side_specs = (
        ("awayTeam", away_team, home_team, 0),
        ("homeTeam", home_team, away_team, 1),
    )
    for side_key, team, opp, is_home in side_specs:
        team_block = pstats.get(side_key) or {}
        for group_key in ("forwards", "defense"):
            for player in team_block.get(group_key, []):
                toi_sec = parse_toi_seconds(player.get("toi"))
                if toi_sec <= 0:
                    continue
                name_obj = player.get("name") or {}
                name_str = name_obj.get("default", "") if isinstance(name_obj, dict) else str(name_obj)
                rows.append(
                    {
                        "season": str(game_meta["season"]),
                        "game_type": int(game_meta["game_type"]),
                        "game_id": int(game_meta["game_id"]),
                        "game_date": str(game_meta["game_date"]),
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "team_goals_against": int(
                            game_meta["home_score"] if is_home == 0 else game_meta["away_score"]
                        ),
                        "player_id": int(player.get("playerId", 0) or 0),
                        "name": name_str,
                        "name_clean": clean_name(name_str),
                        "position": str(player.get("position", "F")),
                        "goals": int(player.get("goals", 0) or 0),
                        "shots": int(player.get("sog", player.get("shots", 0)) or 0),
                        "toi_sec": toi_sec,
                    }
                )
    return rows


def probe_tims_history_endpoints(session: Any) -> dict[str, str]:
    """Check whether HockeyChallengeHelper exposes historical past-date Tims picks endpoints."""
    headers = {
        "Referer": "https://hockeychallengehelper.com/",
        "Origin": "https://hockeychallengehelper.com",
        "Accept": "application/json, text/plain, */*",
    }
    probes = {
        "picks_today": "https://api.hockeychallengehelper.com/api/picks",
        "picks_date_param": "https://api.hockeychallengehelper.com/api/picks?date=2026-11-14",
        "history_endpoint": "https://api.hockeychallengehelper.com/api/history",
        "picks_history": "https://api.hockeychallengehelper.com/api/picks/history",
    }
    statuses: dict[str, str] = {}
    for label, url in probes.items():
        res = _fetch_with_retries(session, url, headers=headers, timeout=10, attempts=1)
        if res is None:
            statuses[label] = "error"
        else:
            snippet = res.text[:80].replace("\n", " ") if res.status_code == 200 else ""
            statuses[label] = f"http_{res.status_code}:{snippet}" if snippet else f"http_{res.status_code}"
    return statuses


def _collect_finished_games(season: str, game_type: int, session: Any) -> dict[int, dict[str, Any]]:
    """Collect unique finished NHL game metadata for a given season and game_type (2 or 3)."""
    games: dict[int, dict[str, Any]] = {}
    for team in NHL_TEAMS:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        res = _fetch_with_retries(session, url, timeout=12, attempts=2)
        if res is None or res.status_code != 200:
            continue
        for g in res.json().get("games", []):
            if int(g.get("gameType") or 0) != game_type:
                continue
            if g.get("gameState") not in ("FINAL", "OFF"):
                continue
            gid = int(g.get("id", 0) or 0)
            if gid <= 0 or gid in games:
                continue
            games[gid] = {
                "season": season,
                "game_type": game_type,
                "game_id": gid,
                "game_date": str(g.get("gameDate")),
                "away_team": normalize_team_abbrev(g.get("awayTeam", {}).get("abbrev")),
                "home_team": normalize_team_abbrev(g.get("homeTeam", {}).get("abbrev")),
                "away_score": int(g.get("awayTeam", {}).get("score", 0) or 0),
                "home_score": int(g.get("homeTeam", {}).get("score", 0) or 0),
            }
    return games


def sync_boxscore_skaters(
    season: str,
    game_type: int,
    backtest_dir: str,
    max_workers: int = 2,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Download/cache complete game-by-game skater boxscores for a season & gameType."""
    os.makedirs(backtest_dir, exist_ok=True)
    cache_path = os.path.join(backtest_dir, f"boxscores_{season}_{game_type}.csv")
    existing_df = pd.read_csv(cache_path) if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0 else pd.DataFrame()
    if not allow_network:
        return existing_df

    session = create_http_session()
    games_map = _collect_finished_games(season, game_type, session)
    cached_ids = set(existing_df["game_id"].astype(int).unique()) if not existing_df.empty else set()
    missing_games = [meta for gid, meta in sorted(games_map.items()) if gid not in cached_ids]

    if not missing_games:
        return existing_df

    print(f"Fetching {len(missing_games)} NHL boxscores for {season} (gameType={game_type}, cached={len(cached_ids)})...")
    thread_local = threading.local()

    def _get_session() -> requests.Session:
        if not hasattr(thread_local, "sess"):
            thread_local.sess = requests.Session()
        return thread_local.sess

    def _fetch_one(meta: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        url = f"https://api-web.nhle.com/v1/gamecenter/{meta['game_id']}/boxscore"
        time.sleep(0.15)
        res = _fetch_with_retries(_get_session(), url, timeout=12, attempts=5)
        if res is None or res.status_code != 200:
            return meta["game_id"], []
        return meta["game_id"], extract_boxscore_skaters(res.json(), meta)

    new_rows: list[dict[str, Any]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, meta) for meta in missing_games]
        for idx, fut in enumerate(as_completed(futures), 1):
            _, rows = fut.result()
            if not rows:
                failed += 1
            new_rows.extend(rows)
            if idx % 150 == 0 or idx == len(missing_games):
                print(f"  ...{idx}/{len(missing_games)} boxscores synced (failed={failed})")
                if new_rows:
                    part_df = pd.DataFrame(new_rows)
                    existing_df = (
                        pd.concat([existing_df, part_df], ignore_index=True)
                        if not existing_df.empty
                        else part_df
                    )
                    existing_df.drop_duplicates(subset=["game_id", "player_id"]).sort_values(
                        ["game_date", "game_id", "team", "player_id"]
                    ).reset_index(drop=True).to_csv(cache_path, index=False)
                    new_rows = []

    final_df = pd.read_csv(cache_path) if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0 else existing_df
    return final_df


def _build_prior_skater_lookup(
    archive_dir: str,
    baseline_frames: list[pd.DataFrame],
    skater_bundle: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], dict[str, dict[str, float]]]:
    """Pre-score strictly pre-period baseline skaters into fast id/name lookups."""
    valid_frames = [f for f in baseline_frames if not f.empty]
    if not valid_frames:
        return {}, {}
    df_comb = combine_skater_frames(valid_frames)
    df_prepped, _ = prep_dynamic_features(df_comb)
    df_prepped["name_clean"] = df_prepped["name"].apply(clean_name)

    features: list[str] = skater_bundle["features"]
    X_all = df_prepped.reindex(columns=features, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    X_imp = skater_bundle["imputer"].transform(X_all)
    preds = np.asarray(skater_bundle["model"].predict(X_imp), dtype=float).reshape(-1)
    df_prepped["base_pred"] = np.clip(preds, 0.0, 2.0)

    by_id: dict[int, dict[str, float]] = {}
    by_name: dict[str, dict[str, float]] = {}
    for _, row in df_prepped.sort_values("games_played", ascending=False).iterrows():
        gp = float(row.get("games_played", 0.0) or 0.0)
        goals = float(row.get("I_F_goals", 0.0) or 0.0)
        rec = {
            "base_pred": float(row["base_pred"]),
            "prior_gp": gp,
            "prior_goals": goals,
            "prior_rate": (goals / gp) if gp > 0 else 0.14,
        }
        pid = int(row.get("playerId", 0) or 0)
        if pid > 0 and pid not in by_id:
            by_id[pid] = rec
        nc = str(row.get("name_clean", ""))
        if nc and nc not in by_name:
            by_name[nc] = rec
    return by_id, by_name


def _build_causal_prior_defense(
    archive_dir: str,
    df_master_logs: pd.DataFrame,
    pairwise_years: list[int],
    profile_year: int,
    max_log_season: str,
    exclude_season_game_type: tuple[str, int] | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Train strictly pre-period pairwise model to get causal team ai_factor & prior base_mult."""
    df_th, df_st, df_go = build_team_star_goalie_history(archive_dir, pairwise_years)
    logs_filtered = df_master_logs[df_master_logs["Season"].astype(str) <= max_log_season].copy()
    if exclude_season_game_type is not None:
        ex_season, ex_gt = exclude_season_game_type
        mask_drop = (logs_filtered["Season"].astype(str) == ex_season) & (logs_filtered["GameType"] == ex_gt)
        logs_filtered = logs_filtered[~mask_drop]

    pairwise_bundle = train_pairwise_game_model(df_th, df_st, df_go, logs_filtered)

    # Extract profile year stats strictly before target period (without reading target-year playoffs)
    df_th_prof, df_st_prof, df_go_prof = build_team_star_goalie_history(archive_dir, [profile_year])
    profiles, _ = build_live_team_profiles(
        df_team_hist=df_th_prof,
        df_stars=df_st_prof,
        df_goalies=df_go_prof,
        active_mp_year=profile_year,
        latest_team_context={},
        base_features=pairwise_bundle["base_features"],
        playoff_teams_df=None,
    )
    rankings, _ = compute_power_rankings_and_defense(profiles, pairwise_bundle)

    raw_ai = {str(r["team"]): 1.0 / max(3.0, min(15.0, float(r["ai_power"]))) for r in rankings}
    mean_ai = max(1e-6, float(np.mean(list(raw_ai.values())))) if raw_ai else 1.0
    ai_factors = {t: val / mean_ai for t, val in raw_ai.items()}

    # Normalized prior season defense component (20% GA/GP, 40% xGA/GP, 40% HDxGA/GP)
    rates = {}
    for team, st in profiles.items():
        gp = max(1.0, float(st.get("games_played", 1.0)))
        rates[team] = (
            float(st.get("goalsAgainst", 0.0)) / gp,
            float(st.get("xGoalsAgainst", 0.0)) / gp,
            float(st.get("highDangerxGoalsAgainst", 0.0)) / gp,
        )
    lg_ga = max(1e-6, float(np.mean([v[0] for v in rates.values()]))) if rates else 3.0
    lg_xga = max(1e-6, float(np.mean([v[1] for v in rates.values()]))) if rates else 3.0
    lg_hdxga = max(1e-6, float(np.mean([v[2] for v in rates.values()]))) if rates else 1.0

    raw_base = {
        t: (0.2 * (v[0] / lg_ga)) + (0.4 * (v[1] / lg_xga)) + (0.4 * (v[2] / lg_hdxga))
        for t, v in rates.items()
    }
    mean_base = max(1e-6, float(np.mean(list(raw_base.values())))) if raw_base else 1.0
    prior_base_mults = {t: val / mean_base for t, val in raw_base.items()}
    prior_lg_ga = max(1.5, lg_ga)

    return ai_factors, prior_base_mults, prior_lg_ga


def evaluate_walk_forward_backtest(
    label: str,
    df_box: pd.DataFrame,
    archive_dir: str,
    df_master_logs: pd.DataFrame,
    skater_mode: str,
    skater_target_years: list[int],
    baseline_files: list[str],
    pairwise_years: list[int],
    profile_year: int,
    max_log_season: str,
    exclude_season_game_type: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Execute zero-leakage chronological walk-forward backtest over real NHL boxscores."""
    if df_box.empty:
        return {"label": label, "error": "empty_boxscores"}

    skater_bundle = train_skater_goal_model(archive_dir, skater_target_years, mode=skater_mode)
    n_half = float(skater_bundle["n_half"])
    baseline_frames = [
        _read_optional_csv(os.path.join(archive_dir, fname)) for fname in baseline_files
    ]
    prior_by_id, prior_by_name = _build_prior_skater_lookup(archive_dir, baseline_frames, skater_bundle)
    ai_factors, prior_base_mults, prior_lg_ga = _build_causal_prior_defense(
        archive_dir=archive_dir,
        df_master_logs=df_master_logs,
        pairwise_years=pairwise_years,
        profile_year=profile_year,
        max_log_season=max_log_season,
        exclude_season_game_type=exclude_season_game_type,
    )

    live_player_state: dict[int, dict[str, float]] = {}
    team_ga_state: dict[str, dict[str, float]] = {}

    strat_names = ("ai_full", "xgb_eb", "xgb_only", "emp_blend", "naive_live", "naive_prior", "random")
    tier_hits: dict[int, dict[str, list[int | float]]] = {
        1: {s: [] for s in strat_names},
        2: {s: [] for s in strat_names},
        3: {s: [] for s in strat_names},
    }
    ticket_outcomes: dict[str, list[int]] = {s: [] for s in strat_names if s != "random"}
    random_expected_tickets = {"hit_1plus": [], "hit_2plus": [], "sweep_3": [], "mean_hits": []}

    slate_top1_hits: dict[str, list[int]] = {s: [] for s in ("ai_full", "xgb_only", "emp_blend", "naive_live", "naive_prior")}
    slate_top3_hits: dict[str, list[int]] = {s: [] for s in ("ai_full", "xgb_only", "emp_blend", "naive_live", "naive_prior")}

    all_preds_prob: list[float] = []
    all_preds_rate: list[float] = []
    all_targets: list[int] = []
    min_games_for_3tier = 2 if skater_mode == "playoffs" else 3

    for date_str, day_df in df_box.groupby("game_date", sort=True):
        # Compute causal league average GA/GP strictly from games < date_str
        tot_ga = sum(v["ga"] for v in team_ga_state.values())
        tot_gp = sum(v["gp"] for v in team_ga_state.values())
        live_lg_ga = (tot_ga / tot_gp) if tot_gp >= 32 else prior_lg_ga

        candidates: list[dict[str, Any]] = []
        for row in day_df.itertuples(index=False):
            pid = int(row.player_id)
            nc = str(row.name_clean)
            prior = prior_by_id.get(pid) or prior_by_name.get(nc)
            base_pred = prior["base_pred"] if prior else 0.14
            prior_gp = prior["prior_gp"] if prior else 0.0
            prior_goals = prior["prior_goals"] if prior else 0.0
            prior_rate = prior["prior_rate"] if prior else 0.14

            st = live_player_state.get(pid, {"gp": 0.0, "goals": 0.0, "toi_sum": 0.0})
            live_gp = st["gp"]
            live_goals = st["goals"]
            live_rate = (live_goals / live_gp) if live_gp > 0 else prior_rate

            w_live = live_gp / (live_gp + n_half) if (live_gp + n_half) > 0 else 0.0
            blended = (base_pred * (1.0 - w_live)) + (live_rate * w_live)

            opp = str(row.opponent)
            p_base_m = prior_base_mults.get(opp, 1.0)
            opp_st = team_ga_state.get(opp, {"gp": 0.0, "ga": 0.0})
            if opp_st["gp"] > 0:
                live_ga_mult = (opp_st["ga"] / opp_st["gp"]) / max(1.5, live_lg_ga)
                w_team = opp_st["gp"] / (opp_st["gp"] + 15.0)
                eff_base_m = (p_base_m * (1.0 - w_team)) + (live_ga_mult * w_team)
            else:
                eff_base_m = p_base_m
            raw_def = (0.7 * eff_base_m) + (0.3 * ai_factors.get(opp, 1.0))
            def_mult = max(0.85, min(1.15, 1.0 + 0.35 * (raw_def - 1.0)))
            ai_full = max(0.0, blended * def_mult)

            scored = 1 if int(row.goals) >= 1 else 0
            p_goal = 1.0 - math.exp(-ai_full)
            all_preds_rate.append(ai_full)
            all_preds_prob.append(p_goal)
            all_targets.append(scored)

            avg_pre_toi = (st["toi_sum"] / live_gp) if live_gp > 0 else float(row.toi_sec)
            neutral_rate = (prior_goals + live_goals + 1.4) / (prior_gp + live_gp + 10.0)
            candidates.append(
                {
                    "pid": pid,
                    "scored": scored,
                    "goals": int(row.goals),
                    "prior_gp": prior_gp,
                    "live_gp": live_gp,
                    "avg_pre_toi": avg_pre_toi,
                    "neutral_rate": neutral_rate,
                    "emp_blend": neutral_rate,
                    "ai_full": ai_full,
                    "xgb_eb": blended,
                    "xgb_only": base_pred,
                    "naive_live": (live_goals / live_gp) if live_gp >= 3 else prior_rate,
                    "naive_prior": prior_rate,
                }
            )

        # Evaluate overall slate top picks
        regulars = [c for c in candidates if (c["prior_gp"] + c["live_gp"] >= 10) or c["avg_pre_toi"] >= 720]
        pool = regulars if len(regulars) >= 10 else candidates
        if pool:
            for s in ("ai_full", "xgb_only", "emp_blend", "naive_live", "naive_prior"):
                ranked_s = sorted(pool, key=lambda x: (x[s], x["avg_pre_toi"]), reverse=True)
                slate_top1_hits[s].append(ranked_s[0]["scored"])
                slate_top3_hits[s].extend(x["scored"] for x in ranked_s[:3])

        # Simulated 3-Group Tim Hortons Challenge evaluation
        num_games = int(day_df["game_id"].nunique())
        eligible = [c for c in candidates if (c["prior_gp"] + c["live_gp"] >= 10) and c["avg_pre_toi"] >= 660]
        if num_games >= min_games_for_3tier and len(eligible) >= 30:
            ranked_neutral = sorted(eligible, key=lambda x: (x["neutral_rate"], x["avg_pre_toi"]), reverse=True)
            if len(ranked_neutral) >= 36:
                groups = {
                    1: ranked_neutral[0:10],
                    2: ranked_neutral[10:22],
                    3: ranked_neutral[22:36],
                }
            else:
                g1_end = max(6, len(ranked_neutral) // 3)
                g2_end = max(g1_end + 8, (2 * len(ranked_neutral)) // 3)
                groups = {
                    1: ranked_neutral[0:g1_end],
                    2: ranked_neutral[g1_end:g2_end],
                    3: ranked_neutral[g2_end:],
                }

            night_hits: dict[str, int] = {s: 0 for s in ticket_outcomes}
            grp_rand_probs: list[float] = []
            for g_idx in (1, 2, 3):
                bucket = groups[g_idx]
                p_rand = float(np.mean([c["scored"] for c in bucket]))
                tier_hits[g_idx]["random"].append(p_rand)
                grp_rand_probs.append(p_rand)
                for s in ticket_outcomes:
                    best = max(bucket, key=lambda x: (x[s], x["neutral_rate"]))
                    tier_hits[g_idx][s].append(best["scored"])
                    night_hits[s] += best["scored"]

            for s, hits_cnt in night_hits.items():
                ticket_outcomes[s].append(hits_cnt)

            p1, p2, p3 = grp_rand_probs
            p_sweep = p1 * p2 * p3
            p_0 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
            p_2plus = (p1 * p2 * (1.0 - p3)) + (p1 * (1.0 - p2) * p3) + ((1.0 - p1) * p2 * p3) + p_sweep
            random_expected_tickets["hit_1plus"].append(1.0 - p_0)
            random_expected_tickets["hit_2plus"].append(p_2plus)
            random_expected_tickets["sweep_3"].append(p_sweep)
            random_expected_tickets["mean_hits"].append(p1 + p2 + p3)

        # Strictly update causal state AFTER scoring date_str
        seen_teams: set[tuple[str, int]] = set()
        for row in day_df.itertuples(index=False):
            pid = int(row.player_id)
            st = live_player_state.setdefault(pid, {"gp": 0.0, "goals": 0.0, "toi_sum": 0.0})
            st["gp"] += 1.0
            st["goals"] += float(row.goals)
            st["toi_sum"] += float(row.toi_sec)

            team = str(row.team)
            if (team, int(row.game_id)) not in seen_teams:
                seen_teams.add((team, int(row.game_id)))
                tst = team_ga_state.setdefault(team, {"gp": 0.0, "ga": 0.0})
                tst["gp"] += 1.0
                tst["ga"] += float(row.team_goals_against)

    # Aggregate results
    y_true = np.asarray(all_targets, dtype=int)
    y_prob = np.clip(np.asarray(all_preds_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    y_rate = np.asarray(all_preds_rate, dtype=float)

    calib_bins = [
        ("0.00-0.12", 0.0, 0.12),
        ("0.12-0.20", 0.12, 0.20),
        ("0.20-0.30", 0.20, 0.30),
        ("0.30-0.42", 0.30, 0.42),
        ("0.42+", 0.42, 10.0),
    ]
    calibration = []
    for b_label, lo, hi in calib_bins:
        mask = (y_rate >= lo) & (y_rate < hi)
        if int(np.sum(mask)) > 0:
            calibration.append(
                {
                    "score_bin": b_label,
                    "count": int(np.sum(mask)),
                    "pred_goal_prob_pct": round(float(np.mean(y_prob[mask])) * 100.0, 1),
                    "actual_hit_pct": round(float(np.mean(y_true[mask])) * 100.0, 1),
                }
            )

    num_slates = len(ticket_outcomes["ai_full"])
    tier_summary: dict[str, dict[str, float]] = {}
    for g_idx in (1, 2, 3):
        tier_summary[f"group_{g_idx}"] = {
            s: round(float(np.mean(tier_hits[g_idx][s])) * 100.0, 2) if tier_hits[g_idx][s] else 0.0
            for s in strat_names
        }

    ticket_summary: dict[str, dict[str, float]] = {}
    for s, vals in ticket_outcomes.items():
        arr = np.asarray(vals, dtype=int)
        ticket_summary[s] = {
            "hit_1plus_pct": round(float(np.mean(arr >= 1)) * 100.0, 2) if len(arr) else 0.0,
            "hit_2plus_pct": round(float(np.mean(arr >= 2)) * 100.0, 2) if len(arr) else 0.0,
            "sweep_3_pct": round(float(np.mean(arr == 3)) * 100.0, 2) if len(arr) else 0.0,
            "mean_hits_of_3": round(float(np.mean(arr)), 3) if len(arr) else 0.0,
        }
    ticket_summary["random"] = {
        k: round(float(np.mean(v)) * (100.0 if k != "mean_hits" else 1.0), 2 if k != "mean_hits" else 3)
        if v
        else 0.0
        for k, v in random_expected_tickets.items()
    }
    ticket_summary["random"]["mean_hits_of_3"] = ticket_summary["random"].pop("mean_hits")

    slate_summary = {
        s: {
            "top1_pct": round(float(np.mean(slate_top1_hits[s])) * 100.0, 2) if slate_top1_hits[s] else 0.0,
            "top3_pct": round(float(np.mean(slate_top3_hits[s])) * 100.0, 2) if slate_top3_hits[s] else 0.0,
        }
        for s in slate_top1_hits
    }

    return {
        "label": label,
        "mode": skater_mode,
        "skater_model_metrics": skater_bundle["metrics"],
        "n_half": n_half,
        "games": int(df_box["game_id"].nunique()),
        "game_nights": int(df_box["game_date"].nunique()),
        "tims_3tier_slates": num_slates,
        "skater_appearances": int(len(y_true)),
        "base_goal_rate_pct": round(float(np.mean(y_true)) * 100.0, 2),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "tiers": tier_summary,
        "tickets": ticket_summary,
        "slate_top_picks": slate_summary,
        "calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward NHL backtest for Tim Hortons Hockey Challenge.")
    parser.add_argument("--data-dir", default="./data", help="Data directory containing moneypuck/ and nhl_game_logs/.")
    parser.add_argument("--skip-network", action="store_true", help="Run strictly from cached CSVs without network.")
    parser.add_argument("--sync-only", action="store_true", help="Download and cache historical data without training.")
    args = parser.parse_args()

    archive_dir = os.path.join(args.data_dir, "moneypuck")
    logs_dir = os.path.join(args.data_dir, "nhl_game_logs")
    backtest_dir = os.path.join(args.data_dir, "backtest")
    os.makedirs(backtest_dir, exist_ok=True)
    probe_path = os.path.join(backtest_dir, "tims_history_probes.json")
    context = get_nhl_season_context()
    session = None if args.skip_network else create_http_session()

    if not args.skip_network:
        print("Syncing MoneyPuck CSVs & NHL game logs...")
        sync_moneypuck_data(archive_dir, context, session=session, max_active_age_hours=12.0)
        tims_probes = probe_tims_history_endpoints(session)
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(tims_probes, f, indent=2)
        print("Tims History Endpoint Probes:", json.dumps(tims_probes))
    elif os.path.exists(probe_path):
        with open(probe_path, "r", encoding="utf-8") as f:
            tims_probes = json.load(f)
    else:
        tims_probes = {"status": "skipped_network"}

    df_master_logs, _, _ = sync_game_logs(
        logs_dir,
        context,
        session=session,
        max_active_age_hours=12.0 if not args.skip_network else 1e9,
        allow_network=not args.skip_network,
    )

    df_box_2526_reg = sync_boxscore_skaters("20252026", 2, backtest_dir, allow_network=not args.skip_network)
    df_box_2526_play = sync_boxscore_skaters("20252026", 3, backtest_dir, allow_network=not args.skip_network)
    df_box_2627_reg = sync_boxscore_skaters("20262027", 2, backtest_dir, allow_network=not args.skip_network)

    if args.sync_only:
        print(
            f"Sync complete: 2025-26 reg={len(df_box_2526_reg)} rows, "
            f"2026 playoffs={len(df_box_2526_play)} rows, 2026-27 reg={len(df_box_2627_reg)} rows."
        )
        return 0

    specs = [
        {
            "label": "2025-26 Regular Season (Out-of-Sample Walk-Forward)",
            "df_box": df_box_2526_reg,
            "skater_mode": "regular",
            "skater_target_years": [2022, 2023, 2024],
            "baseline_files": ["2024_regular_skaters.csv", "2024_playoffs_skaters.csv"],
            "pairwise_years": [2021, 2022, 2023, 2024],
            "profile_year": 2024,
            "max_log_season": "20242025",
            "exclude_season_game_type": None,
        },
        {
            "label": "2026 Stanley Cup Playoffs (Out-of-Sample Walk-Forward)",
            "df_box": df_box_2526_play,
            "skater_mode": "playoffs",
            "skater_target_years": [2022, 2023, 2024],
            "baseline_files": ["2024_playoffs_skaters.csv", "2025_regular_skaters.csv"],
            "pairwise_years": [2021, 2022, 2023, 2024],
            "profile_year": 2025,
            "max_log_season": "20252026",
            "exclude_season_game_type": ("20252026", 3),
        },
        {
            "label": "2026-27 Season to Date (Out-of-Sample Walk-Forward)",
            "df_box": df_box_2627_reg,
            "skater_mode": "regular",
            "skater_target_years": [2022, 2023, 2024, 2025],
            "baseline_files": ["2025_regular_skaters.csv", "2025_playoffs_skaters.csv"],
            "pairwise_years": [2021, 2022, 2023, 2024, 2025],
            "profile_year": 2025,
            "max_log_season": "20252026",
            "exclude_season_game_type": None,
        },
    ]

    results = []
    for sp in specs:
        res = evaluate_walk_forward_backtest(
            label=sp["label"],
            df_box=sp["df_box"],
            archive_dir=archive_dir,
            df_master_logs=df_master_logs,
            skater_mode=sp["skater_mode"],
            skater_target_years=sp["skater_target_years"],
            baseline_files=sp["baseline_files"],
            pairwise_years=sp["pairwise_years"],
            profile_year=sp["profile_year"],
            max_log_season=sp["max_log_season"],
            exclude_season_game_type=sp["exclude_season_game_type"],
        )
        results.append(res)

    report = {"tims_history_probes": tims_probes, "results": results}
    out_path = os.path.join(backtest_dir, "tims_backtest_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
