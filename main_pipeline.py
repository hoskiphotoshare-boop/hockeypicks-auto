"""Headless NHL Hockeypicks AI pipeline split cleanly into Training and Inference stages."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from typing import Any

import pandas as pd

from src.data_sync import (
    create_http_session,
    fetch_live_club_skaters,
    fetch_nhl_schedule,
    fetch_tims_picks,
    get_nhl_season_context,
    sync_game_logs,
    sync_moneypuck_data,
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
from src.team_game_model import (
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_power_rankings_and_defense,
    load_pairwise_bundle,
    predict_upcoming_games,
    save_pairwise_bundle,
    train_pairwise_game_model,
)
from src.tims_tracker import (
    load_tims_history,
    recompute_tims_summary,
    save_tims_history,
    settle_pending_tims_days,
    upsert_tims_snapshot,
)


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
    return has_base and has_skater


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
    df_master_logs, latest_team_context, log_stats = sync_game_logs(
        logs_dir, context, session=session, max_active_age_hours=max_age, allow_network=not skip_network
    )
    print(
        f"Game Logs -> {log_stats['total_rows']} team-games compiled "
        f"(New: {log_stats['downloaded']} | Cached: {log_stats['cached']})"
    )

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
        )

    with open(paths["team_profiles"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "active_mp_year": active_mp_year,
                "profiles_regular": profiles_regular,
                "profiles_playoffs": profiles_playoffs,
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
                "base_pred_regular",
                "base_pred_playoffs",
            ]
        )
    df_pool.to_csv(paths["skater_pool"], index=False, compression="gzip")

    now_et_str = context["now_et"].strftime("%Y-%m-%d %I:%M %p ET")
    inf_meta = {
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
        "df_pool": df_pool,
        "baseline_label": baseline_label,
    }


def run_inference(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    docs_dir: str = "./docs",
    skip_network: bool = False,
    odds_api_key: str | None = None,
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

    print("\n[Infer 1/4] Fetching Live NHL Schedule, Tim Hortons Picks & Active Streaks...")
    if skip_network:
        schedule_info = {
            "games": [],
            "today_games": [],
            "daily_matchups": {},
            "active_game_type": 2,
            "active_mode": "Regular Season",
        }
        df_tims, tims_status = pd.DataFrame(), "skipped_offline"
    else:
        schedule_info = fetch_nhl_schedule(context, session=session)
        df_tims, tims_status = fetch_tims_picks(session=session)

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
        df_pool = trained_state["df_pool"].copy()
        baseline_label = trained_state["baseline_label"]
    else:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            inf_meta = json.load(f)
        sync_stats = inf_meta.get("sync_stats", {"active_mp_year": inf_meta.get("active_mp_year")})
        baseline_label = str(inf_meta.get("skater_baseline_source", "Saved Snapshot"))

        _, latest_team_context, current_log_stats = sync_game_logs(
            logs_dir,
            context,
            session=session,
            max_active_age_hours=1e9 if skip_network else 0.0,
            allow_network=not skip_network,
            seasons_filter=[context["nhl_season"]],
        )
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
        df_pool = pd.read_csv(paths["skater_pool"])

    print(
        f"Mode: {schedule_info['active_mode']} (gameType={schedule_info['active_game_type']}) | "
        f"Scheduled Games: {len(schedule_info['games'])} | Tims API: {tims_status} ({len(df_tims)} players)"
    )

    print("\n[Infer 2/4] Scoring Team Matchups, Power Rankings & Sharp Betting Edges...")
    is_playoffs = schedule_info["active_game_type"] == 3
    profiles = (
        {k: dict(v) for k, v in (profiles_playoffs or profiles_regular).items()}
        if is_playoffs
        else {k: dict(v) for k, v in profiles_regular.items()}
    )
    for team, ctx in latest_team_context.items():
        if team in profiles:
            profiles[team]["Last_Game_Won"] = int(ctx.get("Last_Game_Won", 0))
            profiles[team]["Streak"] = int(ctx.get("Current_Streak", 0))

    power_rankings, defense_multipliers = compute_power_rankings_and_defense(profiles, pairwise_bundle)
    game_predictions, playoff_series = predict_upcoming_games(
        schedule_games=schedule_info["games"],
        profiles=profiles,
        bundle=pairwise_bundle,
        odds_api_key=odds_key,
        session=session,
    )
    print(f"Power Rankings computed for {len(power_rankings)} teams | Predicted {len(game_predictions)} games.")

    print("\n[Infer 3/4] Scoring Tim Hortons & Full Slate Skaters (Empirical Bayes + Matchup D-Mult)...")
    active_skater_mode = "playoffs" if is_playoffs and "playoffs" in skater_bundles else "regular"
    if active_skater_mode not in skater_bundles:
        active_skater_mode = next(iter(skater_bundles.keys()))
    skater_bundle = skater_bundles[active_skater_mode]

    pred_col = f"base_pred_{active_skater_mode}"
    if pred_col in df_pool.columns:
        df_pool["base_pred"] = df_pool[pred_col]
    elif "base_pred_regular" in df_pool.columns:
        df_pool["base_pred"] = df_pool["base_pred_regular"]

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
    )
    history_summary = recompute_tims_summary(tims_history)
    save_tims_history(tims_history, tims_history_path)
    print(
        f"Tims History Tracker -> Days: {history_summary['total_days_logged']} | "
        f"Settled: {history_summary['settled_days']} (Newly settled: {settled_now}, Snapshot saved: {logged_now}) | "
        f"AI Hit Rate: {history_summary['ai_hit_rate_pct']}% vs Random: {history_summary['random_expected_pct']}%"
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
        tims_history=history_summary,
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
