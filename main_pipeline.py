"""Headless NHL Hockeypicks AI pipeline split cleanly into Training and Inference stages."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.data_sync import (
    build_player_value_table,
    build_roster_aware_star_table,
    create_http_session,
    fetch_live_club_skaters,
    fetch_nhl_schedule,
    fetch_tims_picks,
    get_nhl_season_context,
    sync_game_logs,
    sync_moneypuck_data,
)
from src.transactions import (
    compute_team_churn,
    diff_rosters,
    fetch_espn_transactions,
    fetch_team_rosters,
    load_latest_roster_history,
    load_roster_snapshot,
    write_roster_history,
)
from src.report_builder import (
    build_dashboard_payload,
    write_json_report,
    write_terminal_log_html,
)
from src.skater_model import (
    build_inference_baseline,
    compute_skater_base_predictions,
    load_skater_bundle,
    normalize_skater_frame,
    save_skater_bundle,
    score_skater_candidates,
    train_skater_goal_model,
)
from src.availability import (
    apply_live_availability_to_profiles,
    fetch_nhl_injuries,
    fetch_starting_goalies,
    write_injury_history,
)
from src.team_game_model import (
    apply_roster_stars_to_profiles,
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_power_rankings_and_defense,
    extract_cached_odds_lookup,
    extract_latest_team_form,
    load_active_goalie_lookup,
    load_pairwise_bundle,
    predict_upcoming_games,
    save_pairwise_bundle,
    train_pairwise_game_model,
)

from src.live_scoreboard import build_tims_scoreboard, choose_scoreboard_slate
from src.tims_boards import (
    board_players_for,
    load_tims_boards,
    save_tims_boards,
    update_board_day,
)
from src.tims_timing import compute_tims_deferral_advice
from src.tims_tracker import (
    load_tims_history,
    recompute_tims_summary,
    save_tims_history,
    settle_pending_tims_days,
    upsert_tims_snapshot,
)

ARTIFACT_SCHEMA_VERSION = 4


def _determine_skater_target_years(
    archive_dir: str,
    historical_years: list[int],
    current_mp_year: int,
    mode: str,
) -> list[int]:
    """Choose valid historical training target years, guarding tiny early-season samples."""
    years: list[int] = []
    for y in historical_years:
        if y < 2022:
            continue
        if mode == "playoffs":
            reg_ok = os.path.exists(os.path.join(archive_dir, f"{y}_regular_skaters.csv"))
            play_ok = os.path.exists(os.path.join(archive_dir, f"{y}_playoffs_skaters.csv"))
            if reg_ok and play_ok:
                years.append(y)
        else:
            prev_ok = os.path.exists(os.path.join(archive_dir, f"{y - 1}_regular_skaters.csv"))
            curr_path = os.path.join(archive_dir, f"{y}_regular_skaters.csv")
            if prev_ok and os.path.exists(curr_path):
                if y < current_mp_year:
                    years.append(y)
                else:
                    df_curr = normalize_skater_frame(pd.read_csv(curr_path))
                    if len(df_curr[df_curr["games_played"] >= 15]) >= 250:
                        years.append(y)
    return years


def _artifact_paths(models_dir: str) -> dict[str, str]:
    return {
        "pairwise": os.path.join(models_dir, "pairwise_game_bundle.joblib"),
        "skater_regular": os.path.join(models_dir, "skater_regular_bundle.joblib"),
        "skater_playoffs": os.path.join(models_dir, "skater_playoffs_bundle.joblib"),
        "team_profiles": os.path.join(models_dir, "team_profiles_snapshot.json"),
        "skater_pool": os.path.join(models_dir, "skater_pool.csv.gz"),
        "meta": os.path.join(models_dir, "inference_meta.json"),
    }


def _inference_artifacts_ready(models_dir: str) -> bool:
    paths = _artifact_paths(models_dir)
    required = ("pairwise", "team_profiles", "skater_pool", "meta")
    has_base = all(os.path.exists(paths[k]) and os.path.getsize(paths[k]) > 0 for k in required)
    has_skater = os.path.exists(paths["skater_regular"]) or os.path.exists(paths["skater_playoffs"])
    if not (has_base and has_skater):
        return False
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("artifact_schema_version", 1)) >= ARTIFACT_SCHEMA_VERSION
    except Exception:
        return False


def run_training(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    skip_network: bool = False,
) -> dict[str, Any]:
    """Training Stage: Sync MoneyPuck & historical NHL logs, train XGBoost models, save snapshots."""
    archive_dir = os.path.join(data_dir, "moneypuck")
    logs_dir = os.path.join(data_dir, "nhl_game_logs")
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    context = get_nhl_season_context()
    session = None if skip_network else create_http_session()
    paths = _artifact_paths(models_dir)

    print("=== HOCKEYPICKS AI — TRAINING STAGE ===")
    print(f"Date (ET): {context['today_str']} | Season: {context['nhl_season_label']} ({context['nhl_season']})")

    if skip_network:
        available = [
            y
            for y in sorted(context["historical_mp_years"], reverse=True)
            if os.path.exists(os.path.join(archive_dir, f"{y}_regular_skaters.csv"))
        ]
        active_mp_year = available[0] if available else context["mp_year"]
        sync_stats = {
            "downloaded": 0,
            "cached": len(available),
            "unavailable_optional": 0,
            "errors": 0,
            "active_mp_year": active_mp_year,
        }
        print(f"Network skipped (--skip-network). Active local MoneyPuck year: {active_mp_year}")
    else:
        print("\n[Train 1/4] Syncing MoneyPuck Historical & Active CSVs...")
        sync_stats = sync_moneypuck_data(archive_dir, context, session=session)
        print(
            f"MoneyPuck Sync -> Downloaded: {sync_stats['downloaded']} | Cached: {sync_stats['cached']} | "
            f"Optional N/A: {sync_stats['unavailable_optional']} | Errors: {sync_stats['errors']} | "
            f"Active Year: {sync_stats['active_mp_year']}"
        )

    print("\n[Train 2/4] Syncing Historical & Active NHL Game Logs...")
    max_age = 1e9 if skip_network else 6.0
    df_master_logs, _, log_stats = sync_game_logs(
        logs_dir, context, session=session, max_active_age_hours=max_age, allow_network=not skip_network
    )
    latest_team_context = extract_latest_team_form(df_master_logs)
    print(
        f"Game Logs -> {log_stats['total_rows']} team-games compiled "
        f"(New: {log_stats['downloaded']} | Cached: {log_stats['cached']})"
    )

    mp_years = [
        y for y in context.get("historical_mp_years", [])
        if os.path.exists(os.path.join(archive_dir, f"{y}_regular_skaters.csv"))
    ]
    player_values = build_player_value_table(archive_dir, mp_years) if mp_years else {}

    rosters = None
    recent_trades: list[dict[str, Any]] = []
    roster_moves: dict[str, Any] = {}
    if not skip_network and session is not None:
        _snap = None
        try:
            _snap = load_roster_snapshot(docs_dir, context["today_str"])
            rosters = (_snap or {}).get("teams") or None
        except Exception:
            rosters = None
        if rosters is None:
            try:
                rosters = fetch_team_rosters(context["nhl_season"], session=session)
            except Exception as exc:  # pragma: no cover
                print(f"Warning: roster fetch failed: {exc}")
                rosters = None
            if rosters and not _snap:
                try:
                    write_roster_history(rosters, docs_dir, context["today_str"], captured_utc=datetime.now(timezone.utc).isoformat())
                except Exception as exc:  # pragma: no cover
                    print(f"Warning: roster archive write failed: {exc}")
        try:
            all_tx = fetch_espn_transactions(pages=2, session=session)
            today_date = date.fromisoformat(context["today_str"])
            for tx in all_tx:
                if not tx.get("is_trade"):
                    continue
                try:
                    tx_date = date.fromisoformat(str(tx.get("date", ""))[:10])
                except ValueError:
                    continue
                if 0 <= (today_date - tx_date).days <= 7:
                    recent_trades.append(tx)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: transaction fetch failed: {exc}")

    roster_stars = build_roster_aware_star_table(rosters, player_values) if (rosters and player_values) else None

    print("\n[Train 3/4] Training Pairwise Team Matchup Model & Calibrating Temperature T...")
    df_team_hist, df_stars, df_goalies = build_team_star_goalie_history(
        archive_dir, context["historical_mp_years"]
    )
    pairwise_bundle = train_pairwise_game_model(df_team_hist, df_stars, df_goalies, df_master_logs)
    save_pairwise_bundle(pairwise_bundle, paths["pairwise"])
    print(
        f"Pairwise XGBoost -> {pairwise_bundle['metrics']['games_trained']} games | "
        f"Optimal T={pairwise_bundle['optimal_T']} | "
        f"OOF LogLoss: {pairwise_bundle['metrics']['oof_log_loss_raw']} -> "
        f"{pairwise_bundle['metrics']['oof_log_loss_calibrated']} (cal) | "
        f"Acc={pairwise_bundle['metrics']['oof_accuracy']:.1%}"
    )

    active_mp_year = int(sync_stats["active_mp_year"])
    profiles_regular, _ = build_live_team_profiles(
        df_team_hist=df_team_hist,
        df_stars=df_stars,
        df_goalies=df_goalies,
        active_mp_year=active_mp_year,
        latest_team_context=latest_team_context,
        base_features=pairwise_bundle["base_features"],
        playoff_teams_df=None,
        roster_stars=roster_stars,
    )
    profiles_playoffs = None
    pt_path = os.path.join(archive_dir, f"{active_mp_year}_playoffs_teams.csv")
    if os.path.exists(pt_path) and os.path.getsize(pt_path) > 0:
        profiles_playoffs, _ = build_live_team_profiles(
            df_team_hist=df_team_hist,
            df_stars=df_stars,
            df_goalies=df_goalies,
            active_mp_year=active_mp_year,
            latest_team_context=latest_team_context,
            base_features=pairwise_bundle["base_features"],
            playoff_teams_df=pd.read_csv(pt_path),
            roster_stars=roster_stars,
        )

    active_goalies_by_team = load_active_goalie_lookup(archive_dir, active_mp_year)
    with open(paths["team_profiles"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "active_mp_year": active_mp_year,
                "profiles_regular": profiles_regular,
                "profiles_playoffs": profiles_playoffs,
                "active_goalies": active_goalies_by_team,
            },
            f,
        )

    print("\n[Train 4/4] Training Skater Goal Models & Saving Compact Inference Snapshots...")
    skater_bundles: dict[str, dict[str, Any]] = {}
    for sk_mode in ("regular", "playoffs"):
        target_years = _determine_skater_target_years(
            archive_dir, context["historical_mp_years"], context["mp_year"], sk_mode
        )
        if not target_years:
            continue
        bundle = train_skater_goal_model(archive_dir, target_years, mode=sk_mode)
        save_skater_bundle(bundle, paths[f"skater_{sk_mode}"])
        skater_bundles[sk_mode] = bundle
        print(
            f"Skater XGBoost ({sk_mode}) -> {bundle['metrics']['samples']} rows × "
            f"{bundle['metrics']['feature_count']} features | "
            f"Optimal N_half={bundle['n_half']} | CV RMSE={bundle['metrics']['cv_rmse']}"
        )

    if not skater_bundles:
        raise RuntimeError(f"Failed to train either regular or playoffs skater model from {archive_dir}.")

    df_baseline, baseline_label = build_inference_baseline(archive_dir, active_mp_year)
    if not df_baseline.empty:
        reg_b = skater_bundles.get("regular") or skater_bundles["playoffs"]
        play_b = skater_bundles.get("playoffs") or skater_bundles["regular"]
        df_baseline["base_pred_regular"] = compute_skater_base_predictions(df_baseline, reg_b)
        df_baseline["base_pred_playoffs"] = compute_skater_base_predictions(df_baseline, play_b)
        keep_cols = [
            c
            for c in (
                "playerId",
                "name",
                "name_clean",
                "team",
                "position",
                "games_played",
                "I_F_shotsOnGoal_per_game",
                "base_pred_regular",
                "base_pred_playoffs",
            )
            if c in df_baseline.columns
        ]
        df_pool = df_baseline[keep_cols].copy()
    else:
        df_pool = pd.DataFrame(
            columns=[
                "playerId",
                "name",
                "name_clean",
                "team",
                "position",
                "games_played",
                "I_F_shotsOnGoal_per_game",
                "base_pred_regular",
                "base_pred_playoffs",
            ]
        )
    df_pool.to_csv(paths["skater_pool"], index=False, compression="gzip")

    now_et_str = context["now_et"].strftime("%Y-%m-%d %I:%M %p ET")
    inf_meta = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "trained_et": now_et_str,
        "active_mp_year": active_mp_year,
        "skater_baseline_source": baseline_label,
        "sync_stats": sync_stats,
        "log_stats": log_stats,
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(inf_meta, f, indent=2)

    print(f"Training artifacts saved cleanly under {models_dir}.")
    return {
        "meta": {
            "status": "trained",
            "run_mode": "train",
            "updated_et": now_et_str,
            "nhl_season_label": context["nhl_season_label"],
            "active_mode": "Training",
        },
        "context": context,
        "session": session,
        "sync_stats": sync_stats,
        "log_stats": log_stats,
        "latest_team_context": latest_team_context,
        "pairwise_bundle": pairwise_bundle,
        "skater_bundles": skater_bundles,
        "profiles_regular": profiles_regular,
        "profiles_playoffs": profiles_playoffs,
        "active_goalies": active_goalies_by_team,
        "df_pool": df_pool,
        "baseline_label": baseline_label,
        "roster_stars": roster_stars,
        "player_values": player_values,
        "roster_moves": roster_moves,
        "recent_trades": recent_trades,
    }


def _load_existing_payload(path: str) -> dict[str, Any]:
    """Load an existing docs/data.json payload safely for cached odds & quota state."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_inference(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    docs_dir: str = "./docs",
    skip_network: bool = False,
    odds_api_key: str | None = None,
    odds_policy: str = "always",
    live_odds_cap: int = 3,
    trained_state: dict[str, Any] | None = None,
    run_mode: str = "infer",
) -> dict[str, Any]:
    """Inference Stage: Load saved models, sync live slate/form/Tims/odds, export Pages payload."""
    logs_dir = os.path.join(data_dir, "nhl_game_logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    if trained_state is None and not _inference_artifacts_ready(models_dir):
        print(f"Saved inference artifacts missing in {models_dir}; running initial training stage first...")
        trained_state = run_training(data_dir=data_dir, models_dir=models_dir, skip_network=skip_network)
        run_mode = "full"

    odds_key = (odds_api_key if odds_api_key is not None else os.environ.get("ODDS_API_KEY", "")).strip()
    context = trained_state["context"] if trained_state else get_nhl_season_context()
    session = (
        trained_state["session"]
        if trained_state is not None
        else (None if skip_network else create_http_session())
    )
    paths = _artifact_paths(models_dir)

    print("\n=== HOCKEYPICKS AI — INFERENCE STAGE ===")
    print(f"Date (ET): {context['today_str']} | Mode: {run_mode.upper()}")

    print("\n[Infer 1/4] Fetching Live NHL Schedule, Tim Hortons Picks, Injuries & Goalies...")
    if skip_network:
        schedule_info = {
            "games": [],
            "today_games": [],
            "previous_games": [],
            "daily_matchups": {},
            "active_game_type": 2,
            "active_mode": "Regular Season",
        }
        df_tims, tims_status = pd.DataFrame(), "skipped_offline"
        injury_records, injury_status = [], "skipped_offline"
        starting_goalies, goalie_status = {}, "skipped_offline"
    else:
        schedule_info = fetch_nhl_schedule(context, session=session)
        df_tims, tims_status = fetch_tims_picks(session=session)
        injury_records, injury_status = fetch_nhl_injuries(session=session, allow_network=True)
        starting_goalies, goalie_status = fetch_starting_goalies(context["today_str"], session=session, allow_network=True)

    if injury_records:
        try:
            write_injury_history(injury_records, docs_dir, context["today_str"])
        except Exception as exc:  # pragma: no cover
            print(f"Warning: injury archive write failed: {exc}")

    archive_dir = os.path.join(data_dir, "moneypuck")
    mp_years = [
        y for y in context.get("historical_mp_years", [])
        if os.path.exists(os.path.join(archive_dir, f"{y}_regular_skaters.csv"))
    ]
    player_values = build_player_value_table(archive_dir, mp_years) if mp_years else {}

    rosters = None
    recent_trades: list[dict[str, Any]] = []
    roster_moves: dict[str, Any] = {}
    roster_stars = None
    team_churn: dict[str, float] = {}
    # Test doubles expose .urls; skip live roster/tx fetches for them.
    _live_session = (not skip_network) and not hasattr(session, "urls")
    if _live_session:
        _snap = None
        try:
            _snap = load_roster_snapshot(docs_dir, context["today_str"])
            rosters = (_snap or {}).get("teams") or None
        except Exception:
            rosters = None
        if rosters is None:
            try:
                rosters = fetch_team_rosters(context["nhl_season"], session=session)
            except Exception as exc:  # pragma: no cover
                print(f"Warning: roster fetch failed: {exc}")
                rosters = None
        try:
            all_tx = fetch_espn_transactions(pages=2, session=session)
            today_date = date.fromisoformat(context["today_str"])
            for tx in all_tx:
                if not tx.get("is_trade"):
                    continue
                try:
                    tx_date = date.fromisoformat(str(tx.get("date", ""))[:10])
                except ValueError:
                    continue
                if 0 <= (today_date - tx_date).days <= 7:
                    recent_trades.append(tx)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: transaction fetch failed: {exc}")

        if rosters:
            try:
                prev = load_latest_roster_history(docs_dir, before_date=context["today_str"])
                if prev is not None:
                    roster_moves = diff_rosters(prev, rosters)
            except Exception as exc:  # pragma: no cover
                print(f"Warning: roster diff failed: {exc}")
            try:
                if not _snap:
                    write_roster_history(rosters, docs_dir, context["today_str"], captured_utc=datetime.now(timezone.utc).isoformat())
            except Exception as exc:  # pragma: no cover
                print(f"Warning: roster archive write failed: {exc}")

        if rosters and player_values:
            roster_stars = build_roster_aware_star_table(rosters, player_values)

        if roster_moves and player_values:
            team_churn = compute_team_churn(
                roster_moves,
                player_values,
                recent_trades=recent_trades,
            )

    daily_matchups = dict(schedule_info["daily_matchups"])
    if df_tims is not None and not df_tims.empty:
        for _, row in df_tims.iterrows():
            t = str(row.get("team", ""))
            opp = str(row.get("opponentTeam", "UNK"))
            if t and opp != "UNK" and t not in daily_matchups:
                daily_matchups[t] = opp

    if trained_state is not None:
        sync_stats = trained_state["sync_stats"]
        log_stats = trained_state["log_stats"]
        latest_team_context = trained_state["latest_team_context"]
        pairwise_bundle = trained_state["pairwise_bundle"]
        skater_bundles = trained_state["skater_bundles"]
        profiles_regular = trained_state["profiles_regular"]
        profiles_playoffs = trained_state["profiles_playoffs"]
        active_goalies_by_team = trained_state.get("active_goalies") or {}
        df_pool = trained_state["df_pool"].copy()
        baseline_label = trained_state["baseline_label"]
    else:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            inf_meta = json.load(f)
        sync_stats = inf_meta.get("sync_stats", {"active_mp_year": inf_meta.get("active_mp_year")})
        baseline_label = str(inf_meta.get("skater_baseline_source", "Saved Snapshot"))

        df_curr_logs, _, current_log_stats = sync_game_logs(
            logs_dir,
            context,
            session=session,
            max_active_age_hours=1e9 if skip_network else 0.0,
            allow_network=not skip_network,
            seasons_filter=[context["nhl_season"]],
        )
        latest_team_context = extract_latest_team_form(df_curr_logs)
        prior_total = int(inf_meta.get("log_stats", {}).get("total_rows", 0))
        log_stats = {
            **current_log_stats,
            "total_rows": max(prior_total, int(current_log_stats.get("total_rows", 0))),
        }

        pairwise_bundle = load_pairwise_bundle(paths["pairwise"])
        skater_bundles = {}
        for sk_mode in ("regular", "playoffs"):
            if os.path.exists(paths[f"skater_{sk_mode}"]):
                skater_bundles[sk_mode] = load_skater_bundle(paths[f"skater_{sk_mode}"])
        with open(paths["team_profiles"], "r", encoding="utf-8") as f:
            prof_blob = json.load(f)
        profiles_regular = prof_blob.get("profiles_regular") or {}
        profiles_playoffs = prof_blob.get("profiles_playoffs")
        active_goalies_by_team = prof_blob.get("active_goalies") or load_active_goalie_lookup(
            os.path.join(data_dir, "moneypuck"),
            int(sync_stats.get("active_mp_year") or context["mp_year"]),
        )
        df_pool = pd.read_csv(paths["skater_pool"])

    print(
        f"Mode: {schedule_info['active_mode']} (gameType={schedule_info['active_game_type']}) | "
        f"Scheduled Games: {len(schedule_info['games'])} | Tims API: {tims_status} ({len(df_tims)} players) | "
        f"Injuries ({injury_status}): {len(injury_records)} | Starters ({goalie_status}): {len(starting_goalies)}"
    )

    print("\n[Infer 2/4] Scoring Team Matchups, Power Rankings & Sharp Betting Edges...")
    is_playoffs = schedule_info["active_game_type"] == 3
    active_skater_mode = "playoffs" if is_playoffs and "playoffs" in skater_bundles else "regular"
    if active_skater_mode not in skater_bundles:
        active_skater_mode = next(iter(skater_bundles.keys()))
    skater_bundle = skater_bundles[active_skater_mode]

    pred_col = f"base_pred_{active_skater_mode}"
    if pred_col in df_pool.columns:
        df_pool["base_pred"] = df_pool[pred_col]
    elif "base_pred_regular" in df_pool.columns:
        df_pool["base_pred"] = df_pool["base_pred_regular"]

    profiles = (
        {k: dict(v) for k, v in (profiles_playoffs or profiles_regular).items()}
        if is_playoffs
        else {k: dict(v) for k, v in profiles_regular.items()}
    )
    for team, ctx in latest_team_context.items():
        if team in profiles:
            profiles[team]["Last_Game_Won"] = int(ctx.get("Last_Game_Won", profiles[team].get("Last_Game_Won", 0)))
            profiles[team]["Streak"] = int(ctx.get("Current_Streak", profiles[team].get("Streak", 0)))
            if ctx.get("Last_Game_Date"):
                profiles[team]["Last_Game_Date"] = str(ctx["Last_Game_Date"])
            if ctx.get("Recent_Game_Dates"):
                profiles[team]["Recent_Game_Dates"] = list(ctx["Recent_Game_Dates"])
            for k in ("Rolling10_WinPct", "Rolling10_GoalDiff", "EWMA_WinPct", "EWMA_GD", "Causal_Elo"):
                if k in ctx:
                    profiles[team][k] = float(ctx[k])
            if "Causal_Elo" in ctx:
                profiles[team]["Elo"] = float(ctx["Causal_Elo"])

    roster_deltas = apply_roster_stars_to_profiles(profiles, roster_stars) or {}
    for _team, _delta in roster_deltas.items():
        if abs(float(_delta)) >= 0.2:
            team_churn[_team] = max(team_churn.get(_team, 0.0), min(1.0, abs(float(_delta))))

    profiles_unadjusted = {k: dict(v) for k, v in profiles.items()}
    team_availability = apply_live_availability_to_profiles(
        profiles=profiles,
        starting_goalies=starting_goalies,
        injury_records=injury_records,
        active_goalies_by_team=active_goalies_by_team,
        skater_baseline_df=df_pool,
    )
    power_rankings, defense_multipliers = compute_power_rankings_and_defense(profiles, pairwise_bundle)

    existing_payload = _load_existing_payload(os.path.join(docs_dir, "data.json"))
    cached_odds = extract_cached_odds_lookup(existing_payload)
    prev_usage = (existing_payload.get("meta") or {}).get("odds_usage") or {}
    prev_live_fetches = (
        int(prev_usage.get("live_loop_fetches", 0) or 0)
        if prev_usage.get("date") == context["today_str"]
        else 0
    )
    has_unstarted = any(
        str(g.get("game_state", "FUT")) in ("FUT", "PRE")
        for g in (schedule_info.get("games") or [])
    )

    skip_reason = ""
    effective_odds_key = odds_key
    if not odds_key or skip_network:
        effective_odds_key = ""
        skip_reason = "disabled"
    elif odds_policy == "cached":
        effective_odds_key = ""
        skip_reason = "cached_forced"
    elif odds_policy == "live_loop":
        if not has_unstarted:
            effective_odds_key = ""
            skip_reason = "cached_all_started"
        elif prev_live_fetches >= live_odds_cap:
            effective_odds_key = ""
            skip_reason = "cached_cap_reached"

    odds_meta_out: dict[str, Any] = {}
    game_predictions, playoff_series = predict_upcoming_games(
        schedule_games=schedule_info["games"],
        profiles=profiles,
        bundle=pairwise_bundle,
        odds_api_key=effective_odds_key,
        session=session,
        team_availability=team_availability,
        cached_odds=cached_odds,
        odds_meta_out=odds_meta_out,
        profiles_unadjusted=profiles_unadjusted,
        team_churn=team_churn,
    )

    if effective_odds_key and odds_meta_out.get("api_ok"):
        live_loop_fetches = prev_live_fetches + (1 if odds_policy == "live_loop" else 0)
        req_last = odds_meta_out.get("requests_last")
        req_used = odds_meta_out.get("requests_used")
        req_rem = odds_meta_out.get("requests_remaining")
        odds_usage = {
            "date": context["today_str"],
            "live_loop_fetches": live_loop_fetches,
            "live_loop_cap": live_odds_cap,
            "last_source": "api",
            "requests_last": req_last if req_last is not None else prev_usage.get("requests_last"),
            "requests_used": req_used if req_used is not None else prev_usage.get("requests_used"),
            "requests_remaining": req_rem if req_rem is not None else prev_usage.get("requests_remaining"),
        }
        loop_note = f" | Live-loop {live_loop_fetches}/{live_odds_cap}" if odds_policy == "live_loop" else ""
        print(
            f"Odds API -> {odds_meta_out.get('fetched_games', 0)} games fetched{loop_note} "
            f"(cost={odds_usage['requests_last']}, used={odds_usage['requests_used']}, remaining={odds_usage['requests_remaining']})"
        )
    else:
        live_loop_fetches = prev_live_fetches
        odds_usage = {
            "date": context["today_str"],
            "live_loop_fetches": live_loop_fetches,
            "live_loop_cap": live_odds_cap,
            "last_source": skip_reason or "cached_api_fallback",
            "requests_last": prev_usage.get("requests_last"),
            "requests_used": prev_usage.get("requests_used"),
            "requests_remaining": prev_usage.get("requests_remaining"),
        }
        if odds_key and not skip_network:
            print(
                f"Odds API -> Skipped ({odds_usage['last_source']}; live-loop {live_loop_fetches}/{live_odds_cap}, "
                f"reused {len(cached_odds)} cached games)"
            )

    print(f"Power Rankings computed for {len(power_rankings)} teams | Predicted {len(game_predictions)} games.")

    print("\n[Infer 3/4] Scoring Tim Hortons & Full Slate Skaters (Empirical Bayes + Matchup D-Mult)...")

    active_teams = set(daily_matchups.keys()) | set(daily_matchups.values())
    if skip_network or not active_teams:
        live_skaters: dict[str, dict[str, Any]] = {}
    else:
        live_skaters = fetch_live_club_skaters(
            active_teams,
            nhl_season=context["nhl_season"],
            game_type=schedule_info["active_game_type"],
            session=session,
        )

    skater_scores = score_skater_candidates(
        bundle=skater_bundle,
        df_baseline=df_pool,
        df_tims=df_tims,
        daily_matchups=daily_matchups,
        live_skaters=live_skaters,
        defense_multipliers=defense_multipliers,
        active_mode=schedule_info["active_mode"],
        injury_records=injury_records,
    )

    if skater_scores["tims_groups"]:
        for g_key in sorted(skater_scores["tims_groups"].keys()):
            top_p = skater_scores["tims_groups"][g_key][0]
            print(
                f"  Tims Group {g_key}: {top_p['player']} ({top_p['team']} vs {top_p['opponent']}) "
                f"-> Score {top_p['final_score']:.3f} (AI {top_p['base_pred']:.3f}, Live {top_p['live_goals']}G/{top_p['live_gp']}GP)"
            )
    else:
        print("  Tims Groups unavailable right now; generated full slate top projected scorers.")

    tims_timing_advice = compute_tims_deferral_advice(
        tims_groups=skater_scores["tims_groups"],
        slate_candidates=skater_scores.get("all_slate_scorers", skater_scores["slate_top_scorers"]),
        schedule_games=schedule_info.get("games", []),
        now_utc=context["now_et"].astimezone(timezone.utc),
    )
    print(
        f"  Tims Timing Advisor [{tims_timing_advice.get('decision', 'N/A')}]: "
        f"{tims_timing_advice.get('headline', '--')} "
        f"(V_now={tims_timing_advice.get('v_now', 0.0):.3f} vs V_defer={tims_timing_advice.get('v_defer_best', 0.0):.3f})"
    )

    print("\n[Infer 4/4] Settling Forward Tracker & Exporting JSON to docs/...")
    tims_history_path = os.path.join(docs_dir, "tims_history.json")
    tims_history = load_tims_history(tims_history_path)
    now_et_str = context["now_et"].strftime("%Y-%m-%d %I:%M %p ET")
    settled_now = settle_pending_tims_days(
        tims_history,
        today_str=context["today_str"],
        settled_et=now_et_str,
        session=session,
        allow_network=not skip_network,
    )
    logged_now = upsert_tims_snapshot(
        tims_history,
        date_str=context["today_str"],
        context=context,
        schedule_info=schedule_info,
        skater_bundle=skater_bundle,
        tims_groups=skater_scores["tims_groups"],
        captured_et=now_et_str,
        timing_advice=tims_timing_advice,
        now_utc=context["now_et"].astimezone(timezone.utc),
    )
    history_summary = recompute_tims_summary(tims_history)
    save_tims_history(tims_history, tims_history_path)
    print(
        f"Tims History Tracker -> Days: {history_summary['total_days_logged']} | "
        f"Settled: {history_summary['settled_days']} (Newly settled: {settled_now}, Snapshot saved: {logged_now}) | "
        f"AI Hit Rate: {history_summary['ai_hit_rate_pct']}% vs Random: {history_summary['random_expected_pct']}%"
    )

    # Keep every player who appeared on any wave's Tims board, then score the
    # day's boxscores against that union so late goal scorers stay visible.
    boards_path = os.path.join(docs_dir, "tims_boards.json")
    boards = load_tims_boards(boards_path)
    update_board_day(
        boards,
        context["today_str"],
        skater_scores["tims_groups"],
        now_et_str,
    )
    save_tims_boards(boards, boards_path)
    slate, is_previous_day = choose_scoreboard_slate(
        schedule_info.get("today_games") or [],
        schedule_info.get("previous_games") or [],
    )
    today_str = str(context["today_str"])
    yesterday_str = str(context.get("yesterday_str") or "")
    if not yesterday_str:
        yesterday_str = (date.fromisoformat(today_str) - timedelta(days=1)).isoformat()
    scoreboard_date = yesterday_str if is_previous_day else today_str
    tims_scoreboard = build_tims_scoreboard(
        slate,
        board_players_for(boards, scoreboard_date),
        session=session,
        allow_network=not skip_network,
        date_str=scoreboard_date,
    )
    tims_scoreboard_meta = {
        "date": scoreboard_date,
        "previous_day": bool(is_previous_day),
        "all_final": bool(tims_scoreboard)
        and all(
            str(row.get("game_state") or "").upper() in ("FINAL", "OFF")
            for row in tims_scoreboard
        ),
        "scorers": len(tims_scoreboard),
        "goals": sum(int(row.get("goals", 0) or 0) for row in tims_scoreboard),
    }
    if tims_scoreboard:
        print(
            "Tims Scoreboard -> "
            + ", ".join(f"{row['player']} ({row['goals']}G)" for row in tims_scoreboard[:5])
        )

    payload = build_dashboard_payload(
        context=context,
        sync_stats=sync_stats,
        log_stats=log_stats,
        schedule_info=schedule_info,
        tims_status=tims_status,
        skater_bundle=skater_bundle,
        skater_scores=skater_scores,
        skater_baseline_label=baseline_label,
        pairwise_bundle=pairwise_bundle,
        power_rankings=power_rankings,
        game_predictions=game_predictions,
        playoff_series=playoff_series,
        odds_enabled=bool(odds_key),
        odds_usage=odds_usage,
        tims_history=history_summary,
        tims_timing_advice=tims_timing_advice,
        tims_scoreboard=tims_scoreboard,
        tims_scoreboard_meta=tims_scoreboard_meta,
        availability_summary={
            "injury_source": injury_status,
            "injury_count": len(injury_records),
            "goalie_source": goalie_status,
            "goalie_count": len(starting_goalies),
        },
        run_mode=run_mode,
    )
    write_json_report(payload, os.path.join(docs_dir, "data.json"))
    print("Inference complete.")
    return payload


def run_pipeline(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    docs_dir: str = "./docs",
    skip_network: bool = False,
    odds_api_key: str | None = None,
    odds_policy: str = "always",
    live_odds_cap: int = 3,
    mode: str = "full",
) -> dict[str, Any]:
    """Orchestrate training, inference, or full end-to-end execution."""
    if mode == "train":
        return run_training(
            data_dir=data_dir,
            models_dir=models_dir,
            skip_network=skip_network,
        )
    if mode == "infer":
        return run_inference(
            data_dir=data_dir,
            models_dir=models_dir,
            docs_dir=docs_dir,
            skip_network=skip_network,
            odds_api_key=odds_api_key,
            odds_policy=odds_policy,
            live_odds_cap=live_odds_cap,
            run_mode="infer",
        )
    trained_state = run_training(
        data_dir=data_dir,
        models_dir=models_dir,
        skip_network=skip_network,
    )
    return run_inference(
        data_dir=data_dir,
        models_dir=models_dir,
        docs_dir=docs_dir,
        skip_network=skip_network,
        odds_api_key=odds_api_key,
        odds_policy=odds_policy,
        live_odds_cap=live_odds_cap,
        trained_state=trained_state,
        run_mode="full",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hockeypicks AI training and/or inference pipeline.")
    parser.add_argument(
        "--mode",
        choices=("full", "train", "infer"),
        default="full",
        help="Execution stage: 'train', 'infer' (fast live scoring from saved models), or 'full'.",
    )
    parser.add_argument("--data-dir", default="./data", help="Directory for cached MoneyPuck & NHL game logs.")
    parser.add_argument("--models-dir", default="./models", help="Directory for saved joblib bundles & snapshots.")
    parser.add_argument("--docs-dir", default="./docs", help="Directory for GitHub Pages HTML & JSON artifacts.")
    parser.add_argument("--skip-network", action="store_true", help="Run strictly from cached local data.")
    parser.add_argument(
        "--odds-policy",
        choices=("always", "live_loop", "cached"),
        default="always",
        help="Odds API fetch policy: 'always', 'live_loop' (daily capped + unstarted check), or 'cached'.",
    )
    parser.add_argument(
        "--live-odds-cap",
        type=int,
        default=3,
        help="Max live-loop Odds API fetches per ET day when --odds-policy=live_loop.",
    )
    args = parser.parse_args()

    buf = io.StringIO()
    orig_stdout = sys.stdout

    class _Tee(io.TextIOBase):
        def write(self, s: str) -> int:
            orig_stdout.write(s)
            return buf.write(s)

        def flush(self) -> None:
            orig_stdout.flush()

    tee = _Tee()
    with redirect_stdout(tee):
        payload = run_pipeline(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            docs_dir=args.docs_dir,
            skip_network=args.skip_network,
            odds_policy=args.odds_policy,
            live_odds_cap=args.live_odds_cap,
            mode=args.mode,
        )

    write_terminal_log_html(
        buf.getvalue(),
        payload["meta"],
        os.path.join(args.docs_dir, "latest_run.html"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
