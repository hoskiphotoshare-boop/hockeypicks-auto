"""Headless production pipeline for NHL Hockeypicks AI + GitHub Pages dashboard."""

from __future__ import annotations

import argparse
import io
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
    normalize_skater_frame,
    save_skater_bundle,
    score_skater_candidates,
    train_skater_goal_model,
)
from src.team_game_model import (
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_power_rankings_and_defense,
    predict_upcoming_games,
    save_pairwise_bundle,
    train_pairwise_game_model,
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
                    # Include current active regular season as a target only once sample size is meaningful
                    df_curr = normalize_skater_frame(pd.read_csv(curr_path))
                    if len(df_curr[df_curr["games_played"] >= 15]) >= 250:
                        years.append(y)
    return years


def run_pipeline(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    docs_dir: str = "./docs",
    skip_network: bool = False,
    odds_api_key: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end MoneyPuck/NHL sync, XGBoost training, scoring, and Pages export."""
    archive_dir = os.path.join(data_dir, "moneypuck")
    logs_dir = os.path.join(data_dir, "nhl_game_logs")
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    odds_key = (odds_api_key if odds_api_key is not None else os.environ.get("ODDS_API_KEY", "")).strip()
    context = get_nhl_season_context()
    session = None if skip_network else create_http_session()

    print("=== HOCKEYPICKS AUTOMATED AI PIPELINE ===")
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
        print("\n[1/6] Syncing MoneyPuck Historical & Active CSVs...")
        sync_stats = sync_moneypuck_data(archive_dir, context, session=session)
        print(
            f"MoneyPuck Sync -> Downloaded: {sync_stats['downloaded']} | Cached: {sync_stats['cached']} | "
            f"Optional N/A: {sync_stats['unavailable_optional']} | Errors: {sync_stats['errors']} | "
            f"Active Year: {sync_stats['active_mp_year']}"
        )

    print("\n[2/6] Syncing NHL Game Logs & Computing Streaks...")
    max_age = 1e9 if skip_network else 6.0
    df_master_logs, latest_team_context, log_stats = sync_game_logs(
        logs_dir, context, session=session, max_active_age_hours=max_age, allow_network=not skip_network
    )
    print(
        f"Game Logs -> {log_stats['total_rows']} team-games compiled "
        f"(New: {log_stats['downloaded']} | Cached: {log_stats['cached']})"
    )

    print("\n[3/6] Fetching Live NHL Schedule & Tim Hortons Picks...")
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

    print(
        f"Mode: {schedule_info['active_mode']} (gameType={schedule_info['active_game_type']}) | "
        f"Scheduled Games: {len(schedule_info['games'])} | Tims API: {tims_status} ({len(df_tims)} players)"
    )

    print("\n[4/6] Training Pairwise Team Matchup Model & Calibrating Temperature T...")
    df_team_hist, df_stars, df_goalies = build_team_star_goalie_history(
        archive_dir, context["historical_mp_years"]
    )
    pairwise_bundle = train_pairwise_game_model(df_team_hist, df_stars, df_goalies, df_master_logs)
    save_pairwise_bundle(pairwise_bundle, os.path.join(models_dir, "pairwise_game_bundle.joblib"))
    print(
        f"Pairwise XGBoost -> {pairwise_bundle['metrics']['games_trained']} games | "
        f"Optimal T={pairwise_bundle['optimal_T']} | "
        f"OOF LogLoss: {pairwise_bundle['metrics']['oof_log_loss_raw']} -> "
        f"{pairwise_bundle['metrics']['oof_log_loss_calibrated']} (cal) | "
        f"Acc={pairwise_bundle['metrics']['oof_accuracy']:.1%}"
    )

    playoff_teams_df = None
    active_mp_year = int(sync_stats["active_mp_year"])
    if schedule_info["active_game_type"] == 3:
        pt_path = os.path.join(archive_dir, f"{active_mp_year}_playoffs_teams.csv")
        if os.path.exists(pt_path):
            playoff_teams_df = pd.read_csv(pt_path)

    profiles, _ = build_live_team_profiles(
        df_team_hist=df_team_hist,
        df_stars=df_stars,
        df_goalies=df_goalies,
        active_mp_year=active_mp_year,
        latest_team_context=latest_team_context,
        base_features=pairwise_bundle["base_features"],
        playoff_teams_df=playoff_teams_df,
    )
    power_rankings, defense_multipliers = compute_power_rankings_and_defense(profiles, pairwise_bundle)
    game_predictions, playoff_series = predict_upcoming_games(
        schedule_games=schedule_info["games"],
        profiles=profiles,
        bundle=pairwise_bundle,
        odds_api_key=odds_key,
        session=session,
    )
    print(f"Power Rankings computed for {len(power_rankings)} teams | Predicted {len(game_predictions)} games.")

    print("\n[5/6] Training Skater Goal Model (Empirical Bayes + Dynamic Per-Game Features)...")
    active_skater_mode = "playoffs" if schedule_info["active_game_type"] == 3 else "regular"
    target_years = _determine_skater_target_years(
        archive_dir, context["historical_mp_years"], context["mp_year"], active_skater_mode
    )
    if not target_years and active_skater_mode == "playoffs":
        active_skater_mode = "regular"
        target_years = _determine_skater_target_years(
            archive_dir, context["historical_mp_years"], context["mp_year"], "regular"
        )
    skater_bundle = train_skater_goal_model(archive_dir, target_years, mode=active_skater_mode)
    save_skater_bundle(skater_bundle, os.path.join(models_dir, f"skater_{active_skater_mode}_bundle.joblib"))
    print(
        f"Skater XGBoost ({active_skater_mode}) -> {skater_bundle['metrics']['samples']} rows × "
        f"{skater_bundle['metrics']['feature_count']} features | "
        f"Optimal N_half={skater_bundle['n_half']} | CV RMSE={skater_bundle['metrics']['cv_rmse']}"
    )

    df_baseline, baseline_label = build_inference_baseline(archive_dir, active_mp_year)
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
        df_baseline=df_baseline,
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

    print("\n[6/6] Exporting Structured JSON & Terminal Run Log to docs/...")
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
    )
    write_json_report(payload, os.path.join(docs_dir, "data.json"))
    print("Pipeline complete.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hockeypicks AI training & dashboard export pipeline.")
    parser.add_argument("--data-dir", default="./data", help="Directory for cached MoneyPuck & NHL game logs.")
    parser.add_argument("--models-dir", default="./models", help="Directory for saved joblib model bundles.")
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
        )

    write_terminal_log_html(
        buf.getvalue(),
        payload["meta"],
        os.path.join(args.docs_dir, "latest_run.html"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
