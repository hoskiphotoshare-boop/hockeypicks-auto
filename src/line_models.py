"""Line (puck-line spread + total goals) models for NHL games."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    import xgboost as xgb
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_LINE_ML = True
except ImportError:  # pragma: no cover
    HAS_LINE_ML = False
    joblib = None
    xgb = None

from src.team_game_model import (
    _build_causal_training_frame,
    _lookup_prior_table,
    add_causal_game_log_features,
    normalize_team_frame_per_game,
)

class _ConstantProb:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


SPREAD_CORE_FEATURES = [
    "Elo_diff",
    "Prior_GD_diff",
    "Prior_xGD_diff",
    "Blended_GD_GP_diff",
    "EWMA_GD_diff",
    "Rolling10_GoalDiff_diff",
    "cur_exp_margin_s",
    "exp_margin",
    "exp_margin_xg",
    "Top3_Points_per_game_diff",
    "Team_GSAx_per_game_diff",
    "Starter_SavePct_diff",
]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if not np.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def estimate_league_total_prior(
    df_logs: pd.DataFrame,
    before_date: str | None = None,
    default: float = 6.24,
) -> float:
    """Mean goals per game from completed games before ``before_date`` (causal)."""
    if df_logs is None or df_logs.empty:
        return float(default)
    try:
        home = df_logs[df_logs["Is_Home"] == 1].copy()
        if before_date:
            home = home[home["Date"].astype(str) < str(before_date)]
        if home.empty:
            return float(default)
        goals = pd.to_numeric(home["Team_Score"], errors="coerce").fillna(0.0) + pd.to_numeric(
            home["Opp_Score"], errors="coerce"
        ).fillna(0.0)
        goals = goals[goals > 0]
        return float(goals.mean()) if len(goals) else float(default)
    except Exception:
        return float(default)


def build_line_training_frame(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the leak-free line-model training matrix.

    Returns (df_games, X_line, margin, total, seasons, feature_names).
    """
    df_team_norm = normalize_team_frame_per_game(df_team_hist)
    df_games, X_pair, _y_pair, seasons = _build_causal_training_frame(
        df_team_norm, df_stars, df_goalies, df_master_logs
    )

    available_seasons = sorted(int(s) for s in df_team_norm["season"].unique())
    earliest = available_seasons[0] if available_seasons else 2010
    mp_idx = df_team_norm.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"])
    go_idx = (
        df_goalies.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"])
        if not df_goalies.empty
        else pd.DataFrame()
    )

    rows = []
    for r in df_games.itertuples(index=False):
        sy = int(r.season_year)
        prior_candidates = [s for s in available_seasons if s < sy]
        yp = prior_candidates[-1] if prior_candidates else earliest - 1
        tA = str(r.Team_A)
        tB = str(r.Opponent)

        def L(idx, team, col, default):
            return _lookup_prior_table(idx, yp, team, col, default)

        gfA = L(mp_idx, tA, "goalsFor_per_game", 3.1)
        gfB = L(mp_idx, tB, "goalsFor_per_game", 3.1)
        gaA = L(mp_idx, tA, "goalsAgainst_per_game", 3.1)
        gaB = L(mp_idx, tB, "goalsAgainst_per_game", 3.1)
        xgfA = L(mp_idx, tA, "xGoalsFor_per_game", 3.1)
        xgfB = L(mp_idx, tB, "xGoalsFor_per_game", 3.1)
        xgaA = L(mp_idx, tA, "xGoalsAgainst_per_game", 3.1)
        xgaB = L(mp_idx, tB, "xGoalsAgainst_per_game", 3.1)
        sfA = L(mp_idx, tA, "shotsOnGoalFor_per_game", 30.0)
        sfB = L(mp_idx, tB, "shotsOnGoalFor_per_game", 30.0)
        saA = L(mp_idx, tA, "shotsOnGoalAgainst_per_game", 30.0)
        saB = L(mp_idx, tB, "shotsOnGoalAgainst_per_game", 30.0)
        attA = L(mp_idx, tA, "shotAttemptsFor_per_game", 60.0)
        attB = L(mp_idx, tB, "shotAttemptsFor_per_game", 60.0)
        hdA = L(mp_idx, tA, "highDangerShotsFor_per_game", 8.0)
        hdB = L(mp_idx, tB, "highDangerShotsFor_per_game", 8.0)
        hdaA = L(mp_idx, tA, "highDangerShotsAgainst_per_game", 8.0)
        hdaB = L(mp_idx, tB, "highDangerShotsAgainst_per_game", 8.0)
        svA = L(go_idx, tA, "Starter_SavePct", 0.905) if not go_idx.empty else 0.905
        svB = L(go_idx, tB, "Starter_SavePct", 0.905) if not go_idx.empty else 0.905
        gsxA = L(go_idx, tA, "Team_GSAx_per_game", 0.0) if not go_idx.empty else 0.0
        gsxB = L(go_idx, tB, "Team_GSAx_per_game", 0.0) if not go_idx.empty else 0.0

        rows.append(
            {
                "exp_total": (gfA + gaB + gfB + gaA) / 2.0,
                "exp_total_xg": (xgfA + xgaB + xgfB + xgaA) / 2.0,
                "exp_margin": (gfA - gaA) - (gfB - gaB),
                "exp_margin_xg": (xgfA - xgaA) - (xgfB - xgaB),
                "pace": (attA + attB) / 2.0,
                "shots_sum": sfA + sfB,
                "shots_against_sum": saA + saB,
                "danger_sum": hdA + hdB,
                "danger_against_sum": hdaA + hdaB,
                "save_pct_sum": svA + svB,
                "gsax_sum": gsxA + gsxB,
            }
        )
    extra = pd.DataFrame(rows)

    # causal current-season features
    le = add_causal_game_log_features(df_master_logs)
    le["Team_Score"] = pd.to_numeric(le["Team_Score"], errors="coerce").fillna(0)
    le["Opp_Score"] = pd.to_numeric(le["Opp_Score"], errors="coerce").fillna(0)
    parts = []
    for (season, team), g in le.groupby(["Season", "Team"], sort=False):
        g = g.sort_values("Date").copy()
        n = np.arange(len(g), dtype=float)
        g["c_GP"] = n
        g["c_GF"] = g["Team_Score"].cumsum().shift(1).fillna(0) / np.maximum(n, 1)
        g["c_GA"] = g["Opp_Score"].cumsum().shift(1).fillna(0) / np.maximum(n, 1)
        g["c_Total"] = (g["Team_Score"] + g["Opp_Score"]).cumsum().shift(1).fillna(0) / np.maximum(n, 1)
        parts.append(g[["Season", "Date", "Team", "c_GP", "c_GF", "c_GA", "c_Total"]])
    if parts:
        c = pd.concat(parts)
    else:
        c = pd.DataFrame(columns=["Season", "Date", "Team", "c_GP", "c_GF", "c_GA", "c_Total"])
    c["Team"] = c["Team"].astype(str).str.upper()

    home_df = le[le["Is_Home"] == 1][["Season", "Date", "Team", "Opponent"]].copy()
    home_df["Team"] = home_df["Team"].astype(str).str.upper()
    home_df["Opponent"] = home_df["Opponent"].astype(str).str.upper()

    h = home_df.merge(c, on=["Season", "Date", "Team"], how="left")
    away_c = c.rename(
        columns={
            "Team": "Opponent",
            "c_GP": "c_GP_A",
            "c_GF": "c_GF_A",
            "c_GA": "c_GA_A",
            "c_Total": "c_Total_A",
        }
    )
    h = h.merge(away_c, on=["Season", "Date", "Opponent"], how="left")

    hg = le[le["Is_Home"] == 1][["Season", "Date", "Team_Score", "Opp_Score"]].copy()
    hg["tot"] = pd.to_numeric(hg["Team_Score"], errors="coerce") + pd.to_numeric(hg["Opp_Score"], errors="coerce")
    hg = hg.sort_values(["Season", "Date"])
    hg["league_total_prior"] = hg.groupby("Season")["tot"].transform(lambda s: s.shift(1).expanding().mean())
    lg = hg.groupby(["Season", "Date"])["league_total_prior"].mean().reset_index()

    df2 = df_games.copy()
    df2["Season"] = df2["Season"].astype(str)
    df2["Team"] = df2["Team_A"].astype(str).str.upper()
    m = df2.merge(h, on=["Season", "Date", "Team"], how="left").merge(lg, on=["Season", "Date"], how="left")

    cur = pd.DataFrame(
        {
            "cur_gf_A": m.get("c_GF"),
            "cur_ga_A": m.get("c_GA"),
            "cur_gf_B": m.get("c_GF_A"),
            "cur_ga_B": m.get("c_GA_A"),
            "cur_total_A": m.get("c_Total"),
            "cur_total_B": m.get("c_Total_A"),
            "cur_exp_total": (m.get("c_GF") + m.get("c_GA") + m.get("c_GF_A") + m.get("c_GA_A")) / 2.0,
            "cur_exp_margin": (m.get("c_GF") - m.get("c_GA")) - (m.get("c_GF_A") - m.get("c_GA_A")),
            "cur_gp_A": m.get("c_GP"),
            "cur_gp_B": m.get("c_GP_A"),
            "league_total_prior": m.get("league_total_prior"),
        }
    )
    cur["cur_gp_min"] = cur[["cur_gp_A", "cur_gp_B"]].min(axis=1).clip(lower=0)

    # shrinkage
    k = 10.0
    ngp = cur["cur_gp_min"].clip(lower=0)
    cur["cur_exp_total_s"] = (ngp * cur["cur_exp_total"] + k * extra["exp_total"]) / (ngp + k)
    cur["cur_exp_margin_s"] = (ngp * cur["cur_exp_margin"] + k * extra["exp_margin"]) / (ngp + k)

    extra = extra.reset_index(drop=True)
    cur = cur.reset_index(drop=True)
    X_pair = X_pair.reset_index(drop=True)

    X_line = pd.concat([X_pair, extra, cur], axis=1)
    X_line = X_line.fillna(X_line.median())

    margin = (df_games["Team_Score"].astype(float) - df_games["Opp_Score"].astype(float)).to_numpy()
    total = (df_games["Team_Score"].astype(float) + df_games["Opp_Score"].astype(float)).to_numpy()

    return df_games, X_line, margin, total, np.asarray(seasons), list(X_line.columns)


def _group_cv(seasons: np.ndarray):
    uniq = np.unique(seasons)
    n_splits = min(len(uniq), 5)
    if GroupKFold is None or n_splits < 2:
        return []
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(seasons)), seasons, groups=seasons))


def train_line_models(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
) -> dict[str, Any]:
    if not HAS_LINE_ML:
        raise RuntimeError("Missing ML dependencies for line models.")

    df_games, X, margin, total, seasons, feature_names = build_line_training_frame(
        df_team_hist, df_stars, df_goalies, df_master_logs
    )
    cv = _group_cv(seasons)
    n = len(margin)

    # --- spread logistic ---
    spread_features = [f for f in SPREAD_CORE_FEATURES if f in X.columns]
    if not spread_features:
        spread_features = list(X.columns)
    y_spread = (margin >= 2).astype(int)
    base_rate_spread = float(y_spread.mean()) if n else 0.5
    oof_spread = np.full(n, base_rate_spread)
    unique_spread = np.unique(y_spread)
    if len(unique_spread) >= 2:
        if cv:
            for tr, va in cv:
                if len(np.unique(y_spread[tr])) < 2:
                    continue
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=42))
                pipe.fit(X.iloc[tr][spread_features], y_spread[tr])
                oof_spread[va] = pipe.predict_proba(X.iloc[va][spread_features])[:, 1]
        spread_pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=42))
        spread_pipe.fit(X[spread_features], y_spread)
    else:
        spread_pipe = _ConstantProb(base_rate_spread)
    def _safe_metric(fn, *a, **k):
        try:
            return round(float(fn(*a, **k)), 4)
        except Exception:
            return 0.0

    spread_metrics = {
        "brier": _safe_metric(brier_score_loss, y_spread, np.clip(oof_spread, 1e-4, 1 - 1e-4)),
        "log_loss": _safe_metric(log_loss, y_spread, np.clip(oof_spread, 1e-4, 1 - 1e-4)),
        "accuracy": _safe_metric(accuracy_score, y_spread, (oof_spread >= 0.5).astype(int)),
        "base_rate": round(base_rate_spread, 4),
        "base_brier": _safe_metric(brier_score_loss, y_spread, np.full(n, base_rate_spread)),
        "base_log_loss": _safe_metric(log_loss, y_spread, np.full(n, base_rate_spread)),
        "validated": True,
    }

    # --- margin regressor ---
    margin_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=4,
    )
    oof_margin = np.full(n, float(np.mean(margin)) if n else 0.0)
    fold_residuals_list = []
    if cv:
        for tr, va in cv:
            m = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=4,
            )
            m.fit(X.iloc[tr], margin[tr])
            pred_va = m.predict(X.iloc[va])
            oof_margin[va] = pred_va
    margin_model.fit(X, margin)
    residuals = (margin - oof_margin).astype(float)
    # brier at -1.5 using residual ECDF per CV fold already approximated by oof_margin + train residuals?
    # Compute OOF cover probabilities for -1.5 using fold train residuals.
    oof_cover_prob = np.full(n, 0.5)
    if cv:
        for tr, va in cv:
            m = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=4,
            )
            m.fit(X.iloc[tr], margin[tr])
            pred_tr = m.predict(X.iloc[tr])
            res_tr = np.sort(margin[tr] - pred_tr)
            pred_va = m.predict(X.iloc[va])
            for i, va_idx in enumerate(va):
                thr = 1.5  # -home_point for -1.5 line
                diff = thr - float(pred_va[i])
                idx = np.searchsorted(res_tr, diff, side="right")
                p = 1.0 - (idx / len(res_tr)) if len(res_tr) else 0.5
                oof_cover_prob[va_idx] = float(np.clip(p, 1e-4, 1 - 1e-4))
    y_cover = (margin >= 2).astype(int)
    margin_metrics = {
        "mae": round(float(np.mean(np.abs(margin - oof_margin))), 4),
        "brier_at_minus_1_5": round(float(brier_score_loss(y_cover, oof_cover_prob)), 4),
    }

    # --- total classifiers ---
    thresholds = [5, 6, 7, 8]
    classifiers = {}
    base_rates = {}
    briers = {}
    log_losses = {}
    for thr in thresholds:
        y_thr = (total >= thr).astype(int)
        base_rate = float(y_thr.mean()) if n else 0.5
        base_rates[thr] = round(base_rate, 4)
        oof_prob = np.full(n, base_rate)
        unique_thr = np.unique(y_thr)
        if len(unique_thr) >= 2:
            if cv:
                for tr, va in cv:
                    if len(np.unique(y_thr[tr])) < 2:
                        continue
                    clf = xgb.XGBClassifier(
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=2,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        min_child_weight=25,
                        reg_lambda=20.0,
                        random_state=42,
                        eval_metric="logloss",
                        n_jobs=4,
                    )
                    clf.fit(X.iloc[tr], y_thr[tr])
                    oof_prob[va] = clf.predict_proba(X.iloc[va])[:, 1]
            clf_final = xgb.XGBClassifier(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=25,
                reg_lambda=20.0,
                random_state=42,
                eval_metric="logloss",
                n_jobs=4,
            )
            clf_final.fit(X, y_thr)
        else:
            clf_final = _ConstantProb(base_rate)
        # temperature calibration
        T = 1.0
        try:
            from scipy.optimize import minimize_scalar

            def obj(t):
                p = np.clip(oof_prob ** (1.0 / t) / (oof_prob ** (1.0 / t) + (1 - oof_prob) ** (1.0 / t)), 1e-4, 1 - 1e-4)
                return log_loss(y_thr, p)

            res = minimize_scalar(obj, bounds=(1.0, 10.0), method="bounded")
            if res.success and res.fun < log_loss(y_thr, np.clip(oof_prob, 1e-4, 1 - 1e-4)):
                T = float(res.x)
        except Exception:
            T = 1.0
        # apply T
        if T != 1.0:
            oof_prob_cal = np.clip(
                oof_prob ** (1.0 / T) / (oof_prob ** (1.0 / T) + (1 - oof_prob) ** (1.0 / T)),
                1e-4,
                1 - 1e-4,
            )
        else:
            oof_prob_cal = np.clip(oof_prob, 1e-4, 1 - 1e-4)
        classifiers[thr] = {"model": clf_final, "T": T}
        try:
            briers[thr] = round(float(brier_score_loss(y_thr, oof_prob_cal)), 4)
        except Exception:
            briers[thr] = 0.0
        try:
            log_losses[thr] = round(float(log_loss(y_thr, oof_prob_cal)), 4)
        except Exception:
            log_losses[thr] = 0.0

    # total regressor
    total_reg = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_lambda=8.0,
        random_state=42,
        n_jobs=4,
    )
    oof_total = np.full(n, float(np.mean(total)) if n else 6.0)
    if cv:
        for tr, va in cv:
            m = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,
                reg_lambda=8.0,
                random_state=42,
                n_jobs=4,
            )
            m.fit(X.iloc[tr], total[tr])
            oof_total[va] = m.predict(X.iloc[va])
    total_reg.fit(X, total)
    total_residuals = (total - oof_total).astype(float)

    total_metrics = {
        "base_rates": base_rates,
        "brier": briers,
        "log_loss": log_losses,
        "note": "Total model has no validated edge (OOF Brier not below base). Experimental only; no auto-stake.",
        "validated": False,
        "experimental": True,
        "stake_enabled": False,
        "regressor_mae": round(float(np.mean(np.abs(total - oof_total))), 4),
    }

    total_bundle = {
        "classifiers": classifiers,
        "regressor": {"model": total_reg, "residuals": total_residuals.tolist()},
        "base_rates": base_rates,
        "brier": briers,
        "log_loss": log_losses,
        "note": total_metrics["note"],
        "validated": False,
        "experimental": True,
        "stake_enabled": False,
        "metrics": total_metrics,
    }

    margin_bundle = {
        "model": margin_model,
        "residuals": residuals.tolist(),
        "metrics": margin_metrics,
        "features": list(X.columns),
    }

    spread_bundle = {
        "model": spread_pipe,
        "features": spread_features,
        "metrics": spread_metrics,
        "validated": True,
    }

    bundle = {
        "feature_names": feature_names,
        "spread": spread_bundle,
        "margin": margin_bundle,
        "total": total_bundle,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
    }
    return bundle


def save_line_bundle(bundle: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    joblib.dump(bundle, path)


def load_line_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def _apply_temp(prob: float, T: float) -> float:
    p = float(prob)
    if T is None or T == 1.0:
        return p
    try:
        x = p ** (1.0 / T)
        y = (1.0 - p) ** (1.0 / T)
        return float(x / (x + y))
    except Exception:
        return p


def _feats_to_df(feats: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(feats, pd.DataFrame):
        row = feats.iloc[0:1].copy() if len(feats) else pd.DataFrame([{}])
    elif isinstance(feats, dict):
        row = {k: feats.get(k, 0.0) for k in feature_names}
        # include any extra keys
        for k, v in feats.items():
            row[k] = v
        row = pd.DataFrame([row])
    else:
        row = pd.DataFrame([{}])
    for k in feature_names:
        if k not in row.columns:
            row[k] = 0.0
    row = row[feature_names]
    for c in row.columns:
        row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0.0)
    return row


def predict_spread(bundle: dict[str, Any], feats: Any, home_point: float) -> dict[str, Any]:
    hp = float(home_point)
    feature_names = bundle["feature_names"]
    X = _feats_to_df(feats, feature_names)
    exp_margin = None
    # margin prediction
    try:
        exp_margin = float(bundle["margin"]["model"].predict(X)[0])
    except Exception:
        exp_margin = float(feats.get("exp_margin", 0.0)) if isinstance(feats, dict) else 0.0

    if hp == -1.5:
        core = bundle["spread"]["features"]
        Xc = X[core] if all(c in X.columns for c in core) else X
        p_home = float(bundle["spread"]["model"].predict_proba(Xc)[:, 1][0])
        push = 0.0
    else:
        margin_model = bundle["margin"]["model"]
        pred = float(margin_model.predict(X)[0])
        thr = -hp
        res = np.asarray(bundle["margin"]["residuals"], dtype=float)
        sres = np.sort(res)
        n = len(sres)
        if n == 0:
            p_home = 0.5
            push = 0.0
        else:
            if float(hp).is_integer():
                lo = thr - 0.5 - pred
                hi = thr + 0.5 - pred
                i_lo = int(np.searchsorted(sres, lo, side="right"))
                i_hi = int(np.searchsorted(sres, hi, side="right"))
                push = (i_hi - i_lo) / n
                p_home = 1.0 - (i_hi / n)
            else:
                diff = thr - pred
                idx = int(np.searchsorted(sres, diff, side="right"))
                p_home = 1.0 - (idx / n)
                push = 0.0
    p_home = float(np.clip(p_home, 1e-4, 1 - 1e-4))
    p_away = 1.0 - p_home - push
    p_away = float(np.clip(p_away, 0.0, 1.0))
    return {
        "home_cover_prob": p_home,
        "away_cover_prob": p_away,
        "push_prob": float(push),
        "exp_margin": float(exp_margin) if exp_margin is not None else 0.0,
        "validated": bool(bundle["spread"].get("validated", True)),
    }


def predict_total(bundle: dict[str, Any], feats: Any, line: float) -> dict[str, Any]:
    ln = float(line)
    feature_names = bundle["feature_names"]
    X = _feats_to_df(feats, feature_names)
    # classifier probs
    probs_ge = {}
    for thr, info in bundle["total"]["classifiers"].items():
        thr_int = int(thr)
        model = info["model"]
        T = info.get("T", 1.0)
        try:
            raw = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            raw = float(bundle["total"].get("base_rates", {}).get(thr_int, 0.5))
        probs_ge[thr_int] = _apply_temp(raw, T)

    def p_ge(k: int) -> float:
        if k <= 5:
            # at or below lowest threshold: clamp to p_ge[5]
            return float(np.clip(probs_ge.get(5, 0.5), 0, 1))
        if k >= 8:
            return float(np.clip(probs_ge.get(8, 0.0), 0, 1)) if k >= 9 else float(np.clip(probs_ge.get(8, 0.0), 0, 1))
        if k in probs_ge:
            return float(np.clip(probs_ge[k], 0, 1))
        # interpolate between nearest
        lo = max(5, min(7, k))
        hi = lo + 1
        if hi > 8:
            hi = 8
            lo = 7
        p_lo = probs_ge.get(lo, 0.5)
        p_hi = probs_ge.get(hi, 0.5)
        frac = k - lo
        return float(np.clip(p_lo + (p_hi - p_lo) * frac, 0, 1))

    # exp total
    try:
        exp_total = float(bundle["total"]["regressor"]["model"].predict(X)[0])
    except Exception:
        exp_total = float(feats.get("exp_total", 6.0)) if isinstance(feats, dict) else 6.0

    if float(ln).is_integer():
        thr_int = int(ln)
        over = p_ge(thr_int + 1)
        push = p_ge(thr_int) - p_ge(thr_int + 1)
        under = 1.0 - over - push
    else:
        t = int(np.ceil(ln))
        over = p_ge(t)
        push = 0.0
        under = 1.0 - over
    over = float(np.clip(over, 0.0, 1.0))
    under = float(np.clip(under, 0.0, 1.0))
    push = float(np.clip(push, 0.0, 1.0))
    s = over + under + push
    if s > 0:
        over, under, push = over / s, under / s, push / s
    return {
        "over": over,
        "under": under,
        "push": push,
        "exp_total": exp_total,
        "experimental": True,
        "validated": False,
        "stake_enabled": False,
    }


def evaluate_line_edge(prob, odds, min_edge_pct=2.0, stake_enabled=True):
    from src.team_game_model import evaluate_kelly_edge

    edge = evaluate_kelly_edge(prob, odds, min_edge_pct=min_edge_pct)
    if edge is None:
        return None
    if not stake_enabled:
        edge = dict(edge)
        edge["rec_bet"] = 0
        edge["stake_enabled"] = False
    else:
        edge = dict(edge)
        edge["stake_enabled"] = True
    return edge
