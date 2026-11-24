"""Walk-forward fatigue evaluation (2022-2026)."""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import scipy.optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_sync import get_nhl_season_context, sync_game_logs
from src.team_game_model import (
    BASE_PAIRWISE_FEATURES,
    _build_causal_training_frame,
    apply_temperature,
    build_team_star_goalie_history,
    normalize_team_frame_per_game,
    pairwise_feature_names,
)

EPS = 1e-15
B_BOOT = 10000


def fit_lr(Xtr, ytr):
    pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=42),
    )
    pipe.fit(Xtr, ytr)
    return pipe


def fit_predict(feature_list, X, y, seasons, train_mask, test_mask):
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
    Xf = X[feature_list].copy()
    Xtr = Xf.iloc[train_mask]
    ytr = y[train_mask]
    Xte = Xf.iloc[test_mask]
    yte = y[test_mask]
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
            fold_lr = fit_lr(Xtr.iloc[tr_idx], ytr[tr_idx])
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
    final_lr = fit_lr(Xtr, ytr)

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
        "n_test": int(len(yte)),
    }
    return metrics, cal, yte


def np_brier(y, p):
    return float(np.mean((p - y) ** 2))


def np_logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_pooled(y, p_base, p_var, B=B_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    db = np.empty(B)
    dl = np.empty(B)
    da = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p_base[idx]
        pv = p_var[idx]
        db[b] = np_brier(yb, pv) - np_brier(yb, pb)
        dl[b] = np_logloss(yb, pv) - np_logloss(yb, pb)
        try:
            da[b] = roc_auc_score(yb, pv) - roc_auc_score(yb, pb)
        except ValueError:
            da[b] = 0.0
    p_auc = float((np.sum(da <= 0) + 1) / (B + 1))
    return {
        "delta_brier_mean": float(np.mean(db)),
        "delta_brier_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
        "delta_logloss_mean": float(np.mean(dl)),
        "delta_logloss_ci": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
        "delta_auc_mean": float(np.mean(da)),
        "delta_auc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
        "p_auc_le0": p_auc,
    }


def pooled_metrics(y_list, p_list):
    y = np.concatenate(y_list)
    p = np.concatenate(p_list)
    return {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "n": int(len(y)),
    }


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

    base_features = pairwise_feature_names(include_fatigue=True)
    retained = ["Games_Last7_diff", "Travel_Last7_diff", "Rest_x_Travel_diff"]
    extra = ["TZ_Shift_diff", "RoadTrip_Length_diff"]
    all_fatigue = retained + extra
    all_features = base_features + extra

    df_games, X_pair, y_pair, seasons = _build_causal_training_frame(
        df_team_norm, df_stars, df_goalies, df_master_logs, feature_names=all_features
    )
    print(f"Frame: {len(df_games)} games across seasons {sorted(set(seasons.tolist()))}")

    test_seasons = [2022, 2023, 2024, 2025, 2026]

    variant_defs = {
        "Games_Last7_diff": base_features[:21] + ["Games_Last7_diff"],
        "Travel_Last7_diff": base_features[:21] + ["Travel_Last7_diff"],
        "Rest_x_Travel_diff": base_features[:21] + ["Rest_x_Travel_diff"],
        "all3": base_features[:21] + retained,
        "all5": base_features[:21] + all_fatigue,
    }
    # base_features[:21] == BASE_PAIRWISE_FEATURES
    baseline_features = list(BASE_PAIRWISE_FEATURES)
    for k in list(variant_defs.keys()):
        variant_defs[k] = [f for f in (variant_defs[k])]

    def run_for_seasons(season_list):
        # returns dict variant -> {per_season: {S: metrics}, pooled_metrics, pooled_y, pooled_base, pooled_var_probs, sign counts}
        per_season_metrics = {v: {} for v in variant_defs}
        per_season_base = {}
        pooled_base_probs = {s: [] for s in season_list}
        pooled_var_probs = {v: {s: [] for s in season_list} for v in variant_defs}
        pooled_ys = {s: [] for s in season_list}

        for S in season_list:
            train_mask = seasons < S
            test_mask = seasons == S
            n_test = int(test_mask.sum())
            flag = n_test < 100
            print(f"\n=== Season {S} (test games={n_test}){'  [FLAG <100]' if flag else ''} ===")

            # baseline once
            base_m, base_probs, yte = fit_predict(baseline_features, X_pair, y_pair, seasons, train_mask, test_mask)
            per_season_base[S] = base_m
            print(f"  baseline: Brier={base_m['brier']:.4f} AUC={base_m['auc']:.4f}")

            for vname, flist in variant_defs.items():
                m, probs, _ = fit_predict(flist, X_pair, y_pair, seasons, train_mask, test_mask)
                per_season_metrics[vname][S] = m
                # deltas
                db = m["brier"] - base_m["brier"]
                da = m["auc"] - base_m["auc"]
                print(f"  {vname:20s}: Brier={m['brier']:.4f} (dB={db:+.4f}) AUC={m['auc']:.4f} (dA={da:+.4f})")

            # store arrays
            pooled_ys[S] = list(yte)
            pooled_base_probs[S] = list(base_probs)
            for vname in variant_defs:
                # recompute probs? we didn't store; recompute quickly? Instead store from fit_predict return.
                pass

        # Need store probs; refactor: store above. We'll instead collect now by re-running? To avoid double, let's restructure:
        return None

    # Rerun with storage
    all_results = {}
    per_season_base = {}
    per_season_metrics = {v: {} for v in variant_defs}
    per_season_store = {}  # S -> {base_probs, var_probs: {v: probs}, yte}
    for S in test_seasons:
        train_mask = seasons < S
        test_mask = seasons == S
        n_test = int(test_mask.sum())
        flag = n_test < 100
        print(f"\n=== Season {S} (test games={n_test}){'  [FLAG <100]' if flag else ''} ===")

        base_m, base_probs, yte = fit_predict(baseline_features, X_pair, y_pair, seasons, train_mask, test_mask)
        per_season_base[S] = base_m
        var_probs = {}
        for vname, flist in variant_defs.items():
            m, probs, _ = fit_predict(flist, X_pair, y_pair, seasons, train_mask, test_mask)
            per_season_metrics[vname][S] = m
            var_probs[vname] = probs
            db = m["brier"] - base_m["brier"]
            da = m["auc"] - base_m["auc"]
            print(f"  {vname:20s}: Brier={m['brier']:.4f} (dB={db:+.4f}) AUC={m['auc']:.4f} (dA={da:+.4f})")
        per_season_store[S] = {"yte": yte, "base_probs": base_probs, "var_probs": var_probs, "n_test": n_test, "flag": flag}

    # Initialize containers
    per_season_metrics = {v: {} for v in variant_defs}
    # Re-extract metrics already printed but not stored; recompute quickly? We stored only probs. Need metrics. Let's recompute metrics from probs for pooled? Per-season metrics needed for output.
    # We'll recompute metrics now from stored? We didn't store. Let's instead modify: store metrics in per_season_store.
    # Oops. Let's adjust: we need metrics. We'll re-run metric computation from probs using yte.
    for S in test_seasons:
        yte = per_season_store[S]["yte"]
        base_probs = per_season_store[S]["base_probs"]
        base_m = {
            "brier": float(brier_score_loss(yte, base_probs)),
            "logloss": float(log_loss(yte, base_probs)),
            "auc": float(roc_auc_score(yte, base_probs)),
            "acc": float(accuracy_score(yte, (np.array(base_probs) >= 0.5).astype(int))),
            "n_test": int(len(yte)),
        }
        per_season_base[S] = base_m
        for vname in variant_defs:
            probs = per_season_store[S]["var_probs"][vname]
            m = {
                "brier": float(brier_score_loss(yte, probs)),
                "logloss": float(log_loss(yte, probs)),
                "auc": float(roc_auc_score(yte, probs)),
                "acc": float(accuracy_score(yte, (np.array(probs) >= 0.5).astype(int))),
                "n_test": int(len(yte)),
            }
            per_season_metrics[vname][S] = m

    def aggregate(season_list, variant_name):
        ys = np.concatenate([np.array(per_season_store[S]["yte"]) for S in season_list])
        bs = np.concatenate([np.array(per_season_store[S]["base_probs"]) for S in season_list])
        vs = np.concatenate([np.array(per_season_store[S]["var_probs"][variant_name]) for S in season_list])
        base_pm = pooled_metrics(list(per_season_store[S]["yte"] for S in season_list),
                                 list(per_season_store[S]["base_probs"] for S in season_list))
        var_pm = pooled_metrics(list(per_season_store[S]["yte"] for S in season_list),
                                 list(per_season_store[S]["var_probs"][variant_name] for S in season_list))
        boot = bootstrap_pooled(ys, bs, vs)
        return base_pm, var_pm, boot, ys, bs, vs

    # Per-variant pooled + bootstrap for full set
    print("\n========== POOLED (2022-2026) ==========")
    variant_summary = {}
    for vname in variant_defs:
        base_pm, var_pm, boot, ys, bs, vs = aggregate(test_seasons, vname)
        # sign counts (seasons where delta AUC >=0 and delta Brier <=0)
        auc_signs = sum(1 for S in test_seasons if per_season_metrics[vname][S]["auc"] - per_season_base[S]["auc"] >= 0)
        brier_signs = sum(1 for S in test_seasons if per_season_metrics[vname][S]["brier"] - per_season_base[S]["brier"] <= 0)
        db_pooled = var_pm["brier"] - base_pm["brier"]
        da_pooled = var_pm["auc"] - base_pm["auc"]
        rule = (db_pooled <= 0) and (da_pooled >= 0) and (auc_signs >= 3)
        variant_summary[vname] = {
            "pooled_base": base_pm,
            "pooled_var": var_pm,
            "pooled_delta_brier": float(db_pooled),
            "pooled_delta_auc": float(da_pooled),
            "pooled_delta_logloss": float(var_pm["logloss"] - base_pm["logloss"]),
            "auc_sign_count": int(auc_signs),
            "brier_sign_count": int(brier_signs),
            "bootstrap": boot,
            "rule_pass": bool(rule),
            "per_season": {
                str(S): {
                    "var": per_season_metrics[vname][S],
                    "base": per_season_base[S],
                    "delta_brier": float(per_season_metrics[vname][S]["brier"] - per_season_base[S]["brier"]),
                    "delta_auc": float(per_season_metrics[vname][S]["auc"] - per_season_base[S]["auc"]),
                    "n_test": int(per_season_metrics[vname][S]["n_test"]),
                    "flag_lt100": bool(per_season_store[S]["flag"]),
                }
                for S in test_seasons
            },
        }
        print(f"\n--- {vname} ---")
        print(f"  pooled: dBrier={db_pooled:+.5f} dAUC={da_pooled:+.5f} dLogLoss={var_pm['logloss']-base_pm['logloss']:+.5f}")
        print(f"  AUC sign count (of {len(test_seasons)}): {auc_signs} | Brier sign count: {brier_signs}")
        print(f"  bootstrap dAUC mean={boot['delta_auc_mean']:+.5f} CI=[{boot['delta_auc_ci'][0]:+.5f},{boot['delta_auc_ci'][1]:+.5f}] p(<=0)={boot['p_auc_le0']:.4f}")
        print(f"  rule pass: {rule}")

    # Sensitivity 2023-2026
    sens_seasons = [2023, 2024, 2025, 2026]
    print("\n========== SENSITIVITY (2023-2026) ==========")
    sens_summary = {}
    for vname in variant_defs:
        ys = np.concatenate([np.array(per_season_store[S]["yte"]) for S in sens_seasons])
        bs = np.concatenate([np.array(per_season_store[S]["base_probs"]) for S in sens_seasons])
        vs = np.concatenate([np.array(per_season_store[S]["var_probs"][vname]) for S in sens_seasons])
        base_pm = pooled_metrics(list(per_season_store[S]["yte"] for S in sens_seasons),
                                 list(per_season_store[S]["base_probs"] for S in sens_seasons))
        var_pm = pooled_metrics(list(per_season_store[S]["yte"] for S in sens_seasons),
                                 list(per_season_store[S]["var_probs"][vname] for S in sens_seasons))
        auc_signs = sum(1 for S in sens_seasons if per_season_metrics[vname][S]["auc"] - per_season_base[S]["auc"] >= 0)
        db_pooled = var_pm["brier"] - base_pm["brier"]
        da_pooled = var_pm["auc"] - base_pm["auc"]
        rule = (db_pooled <= 0) and (da_pooled >= 0) and (auc_signs >= 3)  # 3 of 4
        # adjust threshold: >=3 of 4
        rule = (db_pooled <= 0) and (da_pooled >= 0) and (auc_signs >= 3)
        sens_summary[vname] = {
            "pooled_delta_brier": float(db_pooled),
            "pooled_delta_auc": float(da_pooled),
            "auc_sign_count": int(auc_signs),
            "rule_pass": bool(rule),
        }
        print(f"  {vname:20s}: dBrier={db_pooled:+.5f} dAUC={da_pooled:+.5f} auc_signs={auc_signs}/4 rule={rule}")

    # Compact table
    print("\n========== COMPACT TABLE (pooled 2022-2026) ==========")
    print(f"{'variant':<22}{'dBrier':>10}{'dAUC':>10}{'dLogLoss':>11}{'AUCsigns':>10}{'rule':>7}")
    for vname in variant_defs:
        vs = variant_summary[vname]
        print(f"{vname:<22}{vs['pooled_delta_brier']:>+10.5f}{vs['pooled_delta_auc']:>+10.5f}{vs['pooled_delta_logloss']:>+11.5f}{vs['auc_sign_count']:>10}{str(vs['rule_pass']):>7}")

    # Recommendation
    passing = [v for v in variant_defs if variant_summary[v]["rule_pass"]]
    if passing:
        print("\nPre-registered rule PASSES for:", ", ".join(passing))
        print("Recommendation: enable", passing[0])
    else:
        print("\nPre-registered rule does NOT pass for any variant.")
        print("Recommendation: keep fatigue disabled.")

    # Write JSON
    out_dir = os.path.join(data_dir, "backtest")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fatigue_walkforward.json")
    payload = {
        "test_seasons": test_seasons,
        "variants": variant_summary,
        "sensitivity_2023_2026": sens_summary,
        "passing_variants": passing,
        "recommendation": (passing[0] if passing else None),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
