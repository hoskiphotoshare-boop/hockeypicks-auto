#!/usr/bin/env python3
"""Build player-game features, train per-stat XGBoost models, and walk-forward test.

Outputs:
  data/backtest/player_stat_models.json
  models/player_stat_bundles.joblib  (only when the gate passes)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GAMES = "data/player_game_logs/history/player_games.csv.gz"
LOGS_DIR = "data/nhl_game_logs"
MP_DIR = "data/moneypuck"
OUT = "data/backtest/player_stat_models.json"
BUNDLE = "models/player_stat_bundles.joblib"
TARGETS = ["goals", "assists", "points", "sog"]
ROLL_STATS = ["goals", "assists", "points", "sog", "toi_min", "pp_points"]
WINDOWS = (3, 5, 10)


def norm_season(value: Any) -> str:
    s = str(value or "").strip()
    if len(s) == 4 and s.isdigit():
        return f"{s}{int(s) + 1}"
    return s[:8]


def load_team_logs() -> pd.DataFrame:
    frames = []
    for name in os.listdir(LOGS_DIR):
        if not name.endswith("_gamelog.csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(LOGS_DIR, name))
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue
        df["Season"] = df["Season"].apply(norm_season) if "Season" in df.columns else name.split("_")[1]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "GameType" in df.columns:
        df = df[pd.to_numeric(df["GameType"], errors="coerce").fillna(2) == 2]
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["team_gf"] = pd.to_numeric(df["Team_Score"], errors="coerce")
    df["team_ga"] = pd.to_numeric(df["Opp_Score"], errors="coerce")
    df["won"] = pd.to_numeric(df.get("Won", 0), errors="coerce").fillna(0)
    df = df.sort_values(["Season", "Team", "Date"])
    g = df.groupby(["Season", "Team"], group_keys=False)
    df["team_gf_pg"] = g["team_gf"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["team_ga_pg"] = g["team_ga"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["team_win_pct"] = g["won"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["team_rest"] = g["Date"].apply(lambda s: pd.to_datetime(s).diff().dt.days.clip(0, 10))
    return df


def team_prior_rates() -> dict[tuple[int, str], dict[str, float]]:
    out: dict[tuple[int, str], dict[str, float]] = {}
    for year in range(2019, 2027):
        path = os.path.join(MP_DIR, f"{year}_regular_teams.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "situation" in df.columns:
            df = df[df["situation"].astype(str).str.lower() == "all"]
        if "position" in df.columns:
            df = df[df["position"].astype(str).str.lower().isin(["all", "team", "team level"])]
        if "team" not in df.columns:
            continue
        for r in df.drop_duplicates(subset=["team"]).itertuples(index=False):
            gp = float(getattr(r, "games_played", 0) or 0)
            if gp <= 0:
                continue
            ga = float(getattr(r, "goalsAgainst", 0) or 0)
            sa = float(getattr(r, "shotsOnGoalAgainst", 0) or 0)
            out[(year, str(r.team).upper())] = {
                "gf_pg": float(getattr(r, "goalsFor", 0) or 0) / gp,
                "ga_pg": ga / gp,
                "xgf_pg": float(getattr(r, "xGoalsFor", 0) or 0) / gp,
                "xga_pg": float(getattr(r, "xGoalsAgainst", 0) or 0) / gp,
                "sf_pg": float(getattr(r, "shotsOnGoalFor", 0) or 0) / gp,
                "sa_pg": sa / gp,
                "save_pct": (1.0 - ga / sa) if sa > 0 else 0.9,
            }
    return out


def season_rates() -> dict[tuple[int, str], dict[str, float]]:
    out: dict[tuple[int, str], dict[str, float]] = {}
    for year in range(2019, 2027):
        path = os.path.join(MP_DIR, f"{year}_regular_skaters.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "situation" in df.columns:
            df = df[df["situation"].astype(str).str.lower() == "all"]
        if "playerId" not in df.columns:
            continue
        df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce")
        df = df.dropna(subset=["playerId"])
        season = f"{year}{year + 1}"
        for r in df.itertuples(index=False):
            gp = float(getattr(r, "games_played", 0) or 0)
            if gp <= 0:
                continue
            assists = float(getattr(r, "I_F_primaryAssists", 0) or 0) + float(getattr(r, "I_F_secondaryAssists", 0) or 0)
            out[(int(r.playerId), season)] = {
                "g_pg": float(getattr(r, "I_F_goals", 0) or 0) / gp,
                "a_pg": assists / gp,
                "p_pg": float(getattr(r, "I_F_points", 0) or 0) / gp,
                "sog_pg": float(getattr(r, "I_F_shotsOnGoal", 0) or 0) / gp,
            }
    return out


def build_features(df: pd.DataFrame, team: pd.DataFrame, prior: dict[tuple[int, str], dict[str, float]], team_prior: dict[tuple[int, str], dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    df["Season"] = df["season"].apply(norm_season)
    df["Date"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df = df.dropna(subset=["Date", "playerId"])
    df["playerId"] = df["playerId"].astype(int)
    df["is_home"] = pd.to_numeric(df["home"], errors="coerce").fillna(0).astype(int)
    df["is_defense"] = (df["position"].astype(str).str.upper().str[:1] == "D").astype(int)
    df = df.sort_values(["playerId", "Date"])
    g = df.groupby("playerId", group_keys=False)
    for col in ROLL_STATS:
        for w in WINDOWS:
            df[f"roll{w}_{col}"] = g[col].apply(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
    for col in ["goals", "assists", "points", "sog", "toi_min"]:
        df[f"season_{col}_pg"] = (
            g.apply(lambda x, c=col: x.groupby("Season")[c].transform(lambda s: s.shift(1).expanding().mean()))
            .reset_index(level=0, drop=True)
            .sort_index()
        )
    df["gp_season"] = g.apply(lambda x: x.groupby("Season").cumcount()).reset_index(level=0, drop=True).sort_index()
    df["rest_days"] = g["Date"].apply(lambda s: s.diff().dt.days.clip(0, 10)).fillna(7)
    df["is_b2b"] = (df["rest_days"] <= 1).astype(int)

    # prior-season rates
    prior_rows = []
    for r in df.itertuples(index=False):
        year = int(str(r.Season)[:4]) - 1
        pr = prior.get((int(r.playerId), f"{year}{year + 1}"), {})
        prior_rows.append(pr)
    pr_df = pd.DataFrame(prior_rows, index=df.index).fillna(0.0)
    for c in ("g_pg", "a_pg", "p_pg", "sog_pg"):
        if c in pr_df.columns:
            df[f"prior_{c}"] = pr_df[c]
        else:
            df[f"prior_{c}"] = 0.0

    # career-to-date rates (prior games only)
    for col in ("goals", "assists", "points", "sog"):
        df[f"career_{col}_pg"] = (
            g[col].apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
            .sort_index()
        )
    df["career_gp"] = g.cumcount().reset_index(level=0, drop=True).sort_index()
    df["pos_group"] = np.where(df["is_defense"] == 1, "D", "F")
    df["roll10_sh_pct"] = (df["roll10_goals"] / df["roll10_sog"].clip(lower=0.05)).clip(0.02, 0.25)
    df["prior_sh_pct"] = (df["prior_g_pg"] / df["prior_sog_pg"].clip(lower=0.05)).clip(0.02, 0.25)

    # opponent defense vs position group: points/goals/sog allowed, prior games only
    allowed = (
        df.groupby(["Season", "Date", "opponent", "pos_group"], observed=True)
        .agg(allowed_g=("goals", "sum"), allowed_a=("assists", "sum"),
             allowed_p=("points", "sum"), allowed_sog=("sog", "sum"))
        .reset_index()
        .sort_values(["opponent", "pos_group", "Date"])
    )
    for col in ("allowed_g", "allowed_a", "allowed_p", "allowed_sog"):
        allowed[f"opp_pos_{col}_pg"] = (
            allowed.groupby(["opponent", "pos_group"])[col]
            .apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
        )
    df = df.merge(
        allowed[["Season", "Date", "opponent", "pos_group", "opp_pos_allowed_g_pg",
                 "opp_pos_allowed_a_pg", "opp_pos_allowed_p_pg", "opp_pos_allowed_sog_pg"]],
        on=["Season", "Date", "opponent", "pos_group"], how="left",
    )

    # prior-season team context (stable, known before the season)
    for side, team_col in (("team", "team"), ("opp", "opponent")):
        rows = []
        for r in df.itertuples(index=False):
            year = int(str(r.Season)[:4]) - 1
            rows.append(team_prior.get((year, str(getattr(r, team_col)).upper()), {}))
        tp = pd.DataFrame(rows, index=df.index)
        for c in ("gf_pg", "ga_pg", "xgf_pg", "xga_pg", "sf_pg", "sa_pg", "save_pct"):
            df[f"prior_{side}_{c}"] = tp[c].fillna({"save_pct": 0.9}.get(c, 0.0))

    # team / opponent context
    t = team[["Season", "Team", "Date", "team_gf_pg", "team_ga_pg", "team_win_pct", "team_rest"]].copy()
    df = df.merge(t.rename(columns={"Team": "team"}), on=["Season", "team", "Date"], how="left", suffixes=("", "_team"))
    opp = t.rename(columns={"Team": "opponent", "team_gf_pg": "opp_gf_pg", "team_ga_pg": "opp_ga_pg", "team_win_pct": "opp_win_pct"})
    df = df.merge(opp, on=["Season", "opponent", "Date"], how="left")
    for c in ("team_gf_pg", "team_ga_pg", "team_win_pct", "opp_gf_pg", "opp_ga_pg", "opp_win_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(3.0 if c.endswith("_pg") else 0.5)

    df["goal_rate"] = df["roll10_goals"].fillna(df["prior_g_pg"])
    return df


def features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in ("goals", "assists", "points", "sog", "toi_min"):
        for w in (5, 10):
            cols.append(f"roll{w}_{col}")
    cols += [f"season_{c}_pg" for c in ("goals", "assists", "points", "sog", "toi_min")]
    cols += ["prior_g_pg", "prior_a_pg", "prior_p_pg", "prior_sog_pg", "prior_sh_pct"]
    cols += ["team_gf_pg", "team_ga_pg", "team_win_pct", "opp_gf_pg", "opp_ga_pg", "opp_win_pct"]
    cols += ["prior_team_gf_pg", "prior_team_ga_pg", "prior_team_save_pct", "prior_opp_gf_pg", "prior_opp_ga_pg", "prior_opp_save_pct"]
    cols += ["gp_season", "rest_days", "is_home", "is_defense", "is_b2b"]
    return cols


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * np.mean(terms))


def train_eval(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for target in TARGETS:
        model_pred = np.zeros(len(test), dtype=float)
        q10_pred = np.zeros(len(test), dtype=float)
        q90_pred = np.zeros(len(test), dtype=float)
        models_by_group: dict[str, Any] = {}
        for group, gmask_tr in (("F", train["is_defense"] == 0), ("D", train["is_defense"] == 1)):
            tr = train[gmask_tr]
            te_idx = test.index[test["is_defense"] == (1 if group == "D" else 0)]
            if len(tr) < 200 or len(te_idx) == 0:
                continue
            Xtr = tr[cols].fillna(0).to_numpy(dtype=float)
            Xte = test.loc[te_idx, cols].fillna(0).to_numpy(dtype=float)
            ytr = tr[target].to_numpy(dtype=float)
            prior_col = "prior_" + ("g" if target == "goals" else "a" if target == "assists" else "p" if target == "points" else "sog") + "_pg"
            bm_tr = np.log(np.clip(tr[prior_col].fillna(0.05).to_numpy(dtype=float), 0.02, 5.0))
            bm_te = np.log(np.clip(test.loc[te_idx, prior_col].fillna(0.05).to_numpy(dtype=float), 0.02, 5.0))
            model = XGBRegressor(
                objective="count:poisson", n_estimators=400, max_depth=4,
                learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, tree_method="hist", n_jobs=4, random_state=42,
            )
            model.fit(Xtr, ytr, base_margin=bm_tr)
            model_pred[test.index.get_indexer(te_idx)] = np.clip(model.predict(Xte, base_margin=bm_te), 0, None)
            q_models = {}
            for q in (0.1, 0.9):
                qm = XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=q,
                    n_estimators=250, max_depth=3, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    n_jobs=4, random_state=42,
                )
                qm.fit(Xtr, ytr, base_margin=bm_tr)
                q_models[q] = qm
                predq = np.clip(qm.predict(Xte, base_margin=bm_te), 0, None)
                if q == 0.1:
                    q10_pred[test.index.get_indexer(te_idx)] = predq
                else:
                    q90_pred[test.index.get_indexer(te_idx)] = predq
            models_by_group[group] = {"mean": model, **{f"q{int(q*100)}": m for q, m in q_models.items()}}

        yte = test[target].to_numpy(dtype=float)
        pred = np.clip(model_pred, 0, None)
        lo = np.minimum(np.clip(q10_pred, 0, None), pred)
        hi = np.maximum(np.clip(q90_pred, 0, None), pred)
        coverage = float(np.mean((yte >= lo) & (yte <= hi)))
        base_cols = {
            "last3": f"roll3_{target}", "last5": f"roll5_{target}",
            "last10": f"roll10_{target}", "season": f"season_{target}_pg",
            "prior": f"prior_{'g' if target=='goals' else 'a' if target=='assists' else 'p' if target=='points' else 'sog'}_pg",
        }
        baselines = {}
        for name, col in base_cols.items():
            if col not in test.columns:
                continue
            bp = test[col].fillna(0).to_numpy(dtype=float)
            baselines[name] = {
                "mae": float(np.mean(np.abs(yte - bp))),
                "rmse": float(np.sqrt(np.mean((yte - bp) ** 2))),
                "poisson_deviance": poisson_deviance(yte, bp),
                "spearman": float(pd.Series(yte).corr(pd.Series(bp), method="spearman") or 0.0),
            }
        prior_pred = test[base_cols["prior"]].fillna(0).to_numpy(dtype=float)
        blends = {}
        for w in (0.25, 0.5, 0.75):
            bp = w * pred + (1 - w) * prior_pred
            blends[str(w)] = {"mae": float(np.mean(np.abs(yte - bp)))}
        results[target] = {
            "rows": int(len(yte)),
            "mae": float(np.mean(np.abs(yte - pred))),
            "rmse": float(np.sqrt(np.mean((yte - pred) ** 2))),
            "poisson_deviance": poisson_deviance(yte, pred),
            "mean_pred": float(np.mean(pred)),
            "mean_actual": float(np.mean(yte)),
            "coverage_80": coverage,
            "mean_width": float(np.mean(hi - lo)),
            "spearman": float(pd.Series(yte).corr(pd.Series(pred), method="spearman") or 0.0),
            "baselines": baselines,
            "blends_with_prior": blends,
            "models": models_by_group,
            "_pred": pred,
            "_lo": lo,
            "_hi": hi,
            "_y": yte,
        }
    return results


def main() -> None:
    df = pd.read_csv(GAMES)
    print(f"loaded {len(df)} player-games", flush=True)
    team = load_team_logs()
    prior = season_rates()
    team_prior = team_prior_rates()
    df = build_features(df, team, prior, team_prior)
    cols = features(df)
    df = df[df["gp_season"] >= 3].copy()
    print(f"model rows {len(df)} | seasons {sorted(df['Season'].unique())}", flush=True)

    splits = [("20242025", "20252026"), ("20252026", "20262027")]
    report = {"targets": TARGETS, "features": cols, "splits": {}, "gate": {}}
    calibration: dict[str, float] = {}
    interval_scale: dict[str, float] = {}
    production_models: dict[str, Any] = {}
    production_metrics: dict[str, Any] = {}

    for idx, (train_last, test_season) in enumerate(splits):
        tr = df[df["Season"] <= train_last]
        te = df[df["Season"] == test_season]
        print(f"split train<= {train_last} ({len(tr)}) test {test_season} ({len(te)})", flush=True)
        res = train_eval(tr, te, cols)
        public = {}
        for target, r in res.items():
            public[target] = {k: v for k, v in r.items() if not k.startswith("_") and k != "models"}
            best_base = min(v["mae"] for v in r["baselines"].values())
            best_dev = min(v["poisson_deviance"] for v in r["baselines"].values())
            print(
                f"  {target}: MAE {r['mae']:.4f} (best base {best_base:.4f}) "
                f"dev {r['poisson_deviance']:.4f} (best base {best_dev:.4f}) "
                f"rho {r['spearman']:.3f} cov {r['coverage_80']:.3f} "
                f"mean {r['mean_pred']:.3f}/{r['mean_actual']:.3f}",
                flush=True,
            )
            if idx == 0:
                calibration[target] = float(r["mean_actual"] / r["mean_pred"]) if r["mean_pred"] > 0 else 1.0
                best_k, best_cov = 1.0, 0.0
                for k in [round(0.5 + 0.05 * i, 2) for i in range(23)]:
                    lo = np.maximum(0, r["_pred"] * calibration[target] - k * (r["_pred"] - r["_lo"]))
                    hi = np.maximum(r["_pred"] * calibration[target], r["_pred"] * calibration[target] + k * (r["_hi"] - r["_pred"]))
                    cov = float(np.mean((r["_y"] >= lo) & (r["_y"] <= hi)))
                    if abs(cov - 0.80) < abs(best_cov - 0.80):
                        best_k, best_cov = k, cov
                interval_scale[target] = best_k
                print(f"    calibrate {target}: mean x{calibration[target]:.3f}, interval k={best_k:.2f} (2025 cov {best_cov:.3f})", flush=True)
            else:
                cal = calibration.get(target, 1.0)
                k = interval_scale.get(target, 1.0)
                pred = r["_pred"] * cal
                lo = np.maximum(0, pred - k * (r["_pred"] - r["_lo"]))
                hi = np.maximum(pred, pred + k * (r["_hi"] - r["_pred"]))
                y = r["_y"]
                cov = float(np.mean((y >= lo) & (y <= hi)))
                from scipy.stats import poisson as _poisson
                p_lo = _poisson.ppf(0.10, np.maximum(pred, 1e-9))
                p_hi = _poisson.ppf(0.90, np.maximum(pred, 1e-9))
                cov_p = float(np.mean((y >= p_lo) & (y <= p_hi)))
                cal_metrics = {
                    "mae": float(np.mean(np.abs(y - pred))),
                    "poisson_deviance": poisson_deviance(y, pred),
                    "coverage_80": cov,
                    "poisson_coverage_80": cov_p,
                    "mean_pred": float(np.mean(pred)),
                    "mean_actual": float(np.mean(y)),
                    "calibration": cal,
                    "interval_scale": k,
                }
                public[target]["calibrated"] = cal_metrics
                production_models[target] = r["models"]
                production_metrics[target] = cal_metrics
                print(
                    f"    calibrated {target}: MAE {cal_metrics['mae']:.4f} dev {cal_metrics['poisson_deviance']:.4f} "
                    f"cov {cov:.3f} poisson {cov_p:.3f} mean {cal_metrics['mean_pred']:.3f}/{cal_metrics['mean_actual']:.3f}",
                    flush=True,
                )
        report["splits"][f"train<={train_last}_test={test_season}"] = public

    latest = report["splits"][list(report["splits"].keys())[-1]]
    for target in TARGETS:
        r = latest[target]
        cal = r.get("calibrated", {})
        best_dev = min(v["poisson_deviance"] for v in r["baselines"].values())
        best_mae = min(v["mae"] for v in r["baselines"].values())
        ok = (cal.get("poisson_deviance", 999) < best_dev) and (0.74 <= cal.get("coverage_80", 0) <= 0.88)
        report["gate"][target] = {
            "pass": bool(ok),
            "mae_improvement_vs_best_baseline": float((best_mae - cal.get("mae", best_mae)) / best_mae),
            "deviance_improvement_vs_best_baseline": float((best_dev - cal.get("poisson_deviance", best_dev)) / best_dev),
            "coverage_80": cal.get("coverage_80"),
            "poisson_coverage_80": cal.get("poisson_coverage_80"),
            "calibration": calibration.get(target),
            "interval_scale": interval_scale.get(target),
            "rows": r["rows"],
        }
    report["calibration"] = calibration
    report["interval_scale"] = interval_scale
    print("gate:", json.dumps(report["gate"], indent=2), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", OUT, flush=True)

    if all(g["pass"] for g in report["gate"].values()):
        import joblib
        from datetime import datetime, timezone
        joblib.dump(
            {
                "features": cols,
                "models": production_models,
                "calibration": calibration,
                "interval_scale": interval_scale,
                "metrics": production_metrics,
                "gate": report["gate"],
                "trained_utc": datetime.now(timezone.utc).isoformat(),
                "train_through": splits[-1][0],
            },
            BUNDLE,
        )
        print("wrote", BUNDLE, flush=True)


if __name__ == "__main__":
    main()
