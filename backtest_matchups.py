"""Leak-free walk-forward backtest for Hockeypicks AI pairwise matchup & power rankings models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.data_sync import get_nhl_season_context, normalize_team_abbrev, sync_game_logs
from src.team_game_model import (
    CAUSAL_PAIRWISE_FEATURES,
    _build_causal_training_frame,
    apply_temperature,
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_causal_elo_history,
    compute_power_rankings_and_defense,
    normalize_team_frame_per_game,
    train_pairwise_game_model,
)


def evaluate_matchups_walk_forward(
    *,
    data_dir: str = "./data",
    test_seasons: list[int] | None = None,
) -> dict[str, Any]:
    """Run strictly causal out-of-sample walk-forward validation across NHL seasons."""
    archive_dir = os.path.join(data_dir, "moneypuck")
    logs_dir = os.path.join(data_dir, "nhl_game_logs")
    context = get_nhl_season_context()
    years = sorted(context["historical_mp_years"])

    df_logs, _, _ = sync_game_logs(logs_dir, context, allow_network=False)
    df_th, df_st, df_go = build_team_star_goalie_history(archive_dir, years)
    df_th_norm = normalize_team_frame_per_game(df_th)

    target_years = test_seasons or [2023, 2024, 2025]
    _, X_all, y_all, s_all = _build_causal_training_frame(df_th_norm, df_st, df_go, df_logs)

    all_probs: list[float] = []
    all_targets: list[int] = []
    all_b2b_home: list[int] = []
    all_b2b_away: list[int] = []
    season_reports: list[dict[str, Any]] = []
    ranking_reports: list[dict[str, Any]] = []

    for y_te in target_years:
        tr_years = [y for y in years if y < y_te]
        logs_tr = df_logs[df_logs["Season"].astype(str).str[:4].astype(int) < y_te].copy()
        if not tr_years or logs_tr.empty:
            continue

        bundle = train_pairwise_game_model(
            df_th[df_th["season"].isin(tr_years)],
            df_st[df_st["season"].isin(tr_years)],
            df_go[df_go["season"].isin(tr_years)],
            logs_tr,
        )
        te_mask = s_all == y_te
        if not np.any(te_mask):
            continue

        X_te = X_all.loc[te_mask, bundle["feature_names"]].fillna(0.0)
        y_te_arr = y_all[te_mask]
        p_x = bundle["model"].predict_proba(X_te)[:, 1]
        lr_m = bundle.get("lr_model")
        w_lr = float(bundle.get("lr_weight", 0.0) or 0.0)
        raw_p = ((1.0 - w_lr) * p_x + w_lr * lr_m.predict_proba(X_te)[:, 1]) if (lr_m is not None and w_lr > 0.0) else p_x
        p_te = apply_temperature(raw_p, T=float(bundle["optimal_T"]))

        all_probs.extend(p_te.tolist())
        all_targets.extend(y_te_arr.tolist())
        all_b2b_home.extend(X_te["Is_B2B_Home"].astype(int).tolist())
        all_b2b_away.extend(X_te["Is_B2B_Away"].astype(int).tolist())

        season_reports.append(
            {
                "season": y_te,
                "games": int(len(y_te_arr)),
                "accuracy_pct": round(float(accuracy_score(y_te_arr, p_te >= 0.5)) * 100.0, 2),
                "log_loss": round(float(log_loss(y_te_arr, p_te)), 4),
                "brier": round(float(brier_score_loss(y_te_arr, p_te)), 4),
                "roc_auc": round(float(roc_auc_score(y_te_arr, p_te)), 4),
            }
        )

        # Pre-season power rankings validation
        pr_yr = y_te - 1
        _, pre_elo = compute_causal_elo_history(logs_tr)
        reg_y_logs = df_logs[(df_logs["Season"].astype(str).str[:4].astype(int) == y_te) & (df_logs["GameType"] == 2)].copy()
        reg_y_logs["Team"] = reg_y_logs["Team"].apply(normalize_team_abbrev)
        act_win = reg_y_logs.groupby("Team")["Won"].mean().rename("actual_win_pct")
        pre_ctx = {
            t: {"Causal_Elo": 1500.0 + 0.65 * (pre_elo.get(t, 1500.0) - 1500.0)}
            for t in set(act_win.index) | set(pre_elo.keys())
        }
        profs, _ = build_live_team_profiles(
            df_th[df_th["season"].isin(tr_years)],
            df_st[df_st["season"].isin(tr_years)],
            df_go[df_go["season"].isin(tr_years)],
            active_mp_year=pr_yr,
            latest_team_context=pre_ctx,
            base_features=bundle["base_features"],
        )
        ranks, _ = compute_power_rankings_and_defense(profs, bundle)
        rdf = pd.DataFrame(ranks)[["team", "ai_power"]].set_index("team")
        joined = rdf.join(act_win, how="inner")
        rho_win, _ = spearmanr(joined["ai_power"], joined["actual_win_pct"]) if len(joined) >= 5 else (0.0, 1.0)
        ranking_reports.append({"season": y_te, "preseason_win_pct_spearman": round(float(rho_win), 3)})

    p_arr = np.asarray(all_probs, dtype=float)
    y_arr = np.asarray(all_targets, dtype=int)
    bh = np.asarray(all_b2b_home, dtype=int)
    ba = np.asarray(all_b2b_away, dtype=int)

    b2b_splits: dict[str, dict[str, Any]] = {}
    for lbl, mask in (
        ("home_rested_vs_away_b2b", (bh == 0) & (ba == 1)),
        ("home_b2b_vs_away_rested", (bh == 1) & (ba == 0)),
        ("both_rested", (bh == 0) & (ba == 0)),
        ("both_b2b", (bh == 1) & (ba == 1)),
    ):
        if int(np.sum(mask)) > 0:
            b2b_splits[lbl] = {
                "games": int(np.sum(mask)),
                "actual_home_win_pct": round(float(np.mean(y_arr[mask])) * 100.0, 2),
                "pred_home_win_pct": round(float(np.mean(p_arr[mask])) * 100.0, 2),
                "accuracy_pct": round(float(accuracy_score(y_arr[mask], p_arr[mask] >= 0.5)) * 100.0, 2),
                "log_loss": round(float(log_loss(y_arr[mask], p_arr[mask])), 4),
            }

    return {
        "overall": {
            "games": int(len(y_arr)),
            "accuracy_pct": round(float(accuracy_score(y_arr, p_arr >= 0.5)) * 100.0, 2) if len(y_arr) else 0.0,
            "log_loss": round(float(log_loss(y_arr, p_arr)), 4) if len(y_arr) else 0.0,
            "brier": round(float(brier_score_loss(y_arr, p_arr)), 4) if len(y_arr) else 0.0,
            "roc_auc": round(float(roc_auc_score(y_arr, p_arr)), 4) if len(y_arr) else 0.0,
            "feature_count": len(CAUSAL_PAIRWISE_FEATURES),
        },
        "seasons": season_reports,
        "b2b_splits": b2b_splits,
        "power_rankings_validation": ranking_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward NHL matchup & power rankings backtest.")
    parser.add_argument("--data-dir", default="./data", help="Data directory containing moneypuck/ and nhl_game_logs/.")
    args = parser.parse_args()

    report = evaluate_matchups_walk_forward(data_dir=args.data_dir)
    out_dir = os.path.join(args.data_dir, "backtest")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matchups_backtest_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
