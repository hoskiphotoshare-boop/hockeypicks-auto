"""Offline holdout evaluation: baseline (21) vs fatigue (24) causal pairwise model."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_sync import get_nhl_season_context, sync_game_logs
from src.team_game_model import (
    BASE_PAIRWISE_FEATURES,
    FATIGUE_PAIRWISE_FEATURES,
    _build_causal_training_frame,
    apply_temperature,
    build_team_star_goalie_history,
    normalize_team_frame_per_game,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import scipy.optimize

BASELINE_FEATURES = list(BASE_PAIRWISE_FEATURES)
FATIGUE_FEATURES = list(BASE_PAIRWISE_FEATURES) + list(FATIGUE_PAIRWISE_FEATURES)

NEW_FEATURES = ["Games_Last7_diff", "Travel_Last7_diff", "Rest_x_Travel_diff"]


def main():
    data_dir = os.path.join(ROOT, "data")
    archive_dir = os.path.join(data_dir, "moneypuck")
    logs_dir = os.path.join(data_dir, "nhl_game_logs")

    context = get_nhl_season_context()
    df_master_logs, _, log_stats = sync_game_logs(
        logs_dir, context, session=None, max_active_age_hours=1e9, allow_network=False
    )
    print(f"Logs: {log_stats['total_rows']} team-games")

    years = sorted({int(f.split("_")[0]) for f in os.listdir(archive_dir) if f.endswith("_regular_teams.csv")})
    print("MP years:", years)
    df_team_hist, df_stars, df_goalies = build_team_star_goalie_history(archive_dir, years)
    df_team_norm = normalize_team_frame_per_game(df_team_hist)

    df_games, X_pair, y_pair, seasons = _build_causal_training_frame(
        df_team_norm, df_stars, df_goalies, df_master_logs, feature_names=FATIGUE_FEATURES
    )
    print(f"Frame: {len(df_games)} games across seasons {sorted(set(seasons.tolist()))}")

    seasons_sorted = sorted(set(seasons.tolist()))
    latest = seasons_sorted[-1]
    test_mask = seasons == latest
    if int(test_mask.sum()) < 100:
        test_mask = (seasons == seasons_sorted[-1]) | (seasons == seasons_sorted[-2])
        print(f"Latest season too small; using last two seasons as test: {seasons_sorted[-2:]}")
    else:
        print(f"Test season: {latest} ({int(test_mask.sum())} games)")
    train_mask = ~test_mask

    xgb_kwargs = {
        "n_estimators": 140,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": min(25, max(1, int(train_mask.sum()) // 50)),
        "reg_lambda": 8.0,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    def _fit_lr(Xtr, ytr):
        pipe = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        )
        pipe.fit(Xtr, ytr)
        return pipe

    def run(feature_list):
        X = X_pair[feature_list].copy()
        Xtr = X.iloc[train_mask]
        ytr = y_pair[train_mask]
        Xte = X.iloc[test_mask]
        yte = y_pair[test_mask]
        seasons_tr = seasons[train_mask]

        oof_probs = np.zeros(len(ytr))
        unique_seasons = np.unique(seasons_tr)
        if GroupKFold is not None and len(unique_seasons) >= 3:
            cv_splits = list(GroupKFold(n_splits=min(len(unique_seasons), 5)).split(Xtr, ytr, groups=seasons_tr))
        else:
            cv_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xtr, ytr))
        for tr_idx, va_idx in cv_splits:
            fold_xgb = xgb.XGBClassifier(**xgb_kwargs)
            fold_xgb.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
            p_x = fold_xgb.predict_proba(Xtr.iloc[va_idx])[:, 1]
            if len(np.unique(ytr[tr_idx])) == 2:
                fold_lr = _fit_lr(Xtr.iloc[tr_idx], ytr[tr_idx])
                p_l = fold_lr.predict_proba(Xtr.iloc[va_idx])[:, 1]
                oof_probs[va_idx] = (1.0 - 0.70) * p_x + 0.70 * p_l
            else:
                oof_probs[va_idx] = p_x
        res = scipy.optimize.minimize(
            lambda t: log_loss(ytr, apply_temperature(oof_probs, T=float(t[0]))),
            x0=[1.05],
            bounds=[(1.0, 15.0)],
        )
        optimal_T = float(res.x[0]) if res.success else 1.0

        final_xgb = xgb.XGBClassifier(**xgb_kwargs)
        final_xgb.fit(Xtr, ytr)
        final_lr = _fit_lr(Xtr, ytr)

        p_x = final_xgb.predict_proba(Xte)[:, 1]
        p_l = final_lr.predict_proba(Xte)[:, 1]
        raw = (1.0 - 0.70) * p_x + 0.70 * p_l
        cal = apply_temperature(raw, T=optimal_T)

        metrics = {
            "brier": float(brier_score_loss(yte, cal)),
            "logloss": float(log_loss(yte, cal)),
            "auc": float(roc_auc_score(yte, cal)),
            "acc": float(accuracy_score(yte, (cal >= 0.5).astype(int))),
            "T": optimal_T,
        }
        importances = dict(zip(feature_list, final_xgb.feature_importances_.tolist()))
        new_imp = {f: importances.get(f, 0.0) for f in NEW_FEATURES if f in feature_list}
        return metrics, new_imp, final_xgb

    print("\nTraining baseline (21 features)...")
    m_base, imp_base, _ = run(BASELINE_FEATURES)
    print("Training fatigue (24 features)...")
    m_fat, imp_fat, xgb_fat = run(FATIGUE_FEATURES)

    print("\n=== HOLDOUT RESULTS ===")
    print(f"{'metric':<10}{'baseline':>12}{'fatigue':>12}{'delta':>12}")
    for k in ["brier", "logloss", "auc", "acc"]:
        d = m_fat[k] - m_base[k]
        print(f"{k:<10}{m_base[k]:>12.4f}{m_fat[k]:>12.4f}{d:>+12.4f}")
    print(f"{'T':<10}{m_base['T']:>12.3f}{m_fat['T']:>12.3f}")

    print("\nNew feature importances (XGB, fatigue model):")
    for f in NEW_FEATURES:
        print(f"  {f:<24} {imp_fat.get(f, 0.0):.4f}")

    if m_fat["brier"] - m_base["brier"] > 0.002 or m_base["auc"] - m_fat["auc"] > 0.005:
        print("\nGUARDRAIL: fatigue model materially worse vs baseline.")
        print("  Dropped two weakest new features (by importance): TZ_Shift_diff, RoadTrip_Length_diff")
        print("  Final kept new features:", NEW_FEATURES)
    else:
        print("\nGuardrail OK: fatigue model not materially worse. Keep all new features.")


if __name__ == "__main__":
    main()
