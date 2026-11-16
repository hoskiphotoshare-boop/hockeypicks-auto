"""Skater goal-scoring XGBoost models, Empirical Bayes updating, and daily picks scoring."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold

    HAS_SKATER_ML = True
except ImportError:  # pragma: no cover
    joblib = None
    xgb = None
    SimpleImputer = None
    mean_absolute_error = None
    mean_squared_error = None
    KFold = None
    HAS_SKATER_ML = False

from src.data_sync import clean_name, normalize_team_abbrev


SKATER_EXCLUDE_COLS = {
    "playerId",
    "season",
    "name",
    "team",
    "position",
    "situation",
    "games_played",
    "I_F_goals",
    "I_F_points",
    "I_F_primaryAssists",
    "I_F_secondaryAssists",
    "playoff_points",
    "playoff_games_played",
    "playoff_P_GP",
    "target_gp",
    "target_goals",
    "target_goals_per_game",
    "playoff_goals",
    "playoff_gp",
    "playoff_goals_per_game",
    "reg_next_goals",
    "reg_next_gp",
    "reg_next_goals_per_game",
    "season_year",
}

RATE_MARKERS = ("Percentage", "percentage", "Rank", "Rate")


def _is_rate_column(col_name: str) -> bool:
    return any(marker in col_name for marker in RATE_MARKERS)


def normalize_skater_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter MoneyPuck skaters DataFrame to situation=='all' and normalize team codes."""
    if df.empty:
        return df.copy()
    out = df[df["situation"] == "all"].copy() if "situation" in df.columns else df.copy()
    if "team" in out.columns:
        out["team"] = out["team"].apply(normalize_team_abbrev)
    if "games_played" in out.columns:
        out["games_played"] = pd.to_numeric(out["games_played"], errors="coerce").fillna(0.0)
    return out


def combine_skater_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine skater stat segments, summing counts and GP-weighting rate/percentage columns.

    Fixes the V3.1 bug where percentage, rank, and season columns were summed across splits.
    """
    valid = [normalize_skater_frame(f) for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1 and valid[0]["playerId"].is_unique:
        return valid[0].copy()

    # Keep metadata from the newest/last valid frame (falling back to earlier frames if player absent)
    meta_cols = [c for c in ("playerId", "name", "team", "position", "situation") if c in valid[-1].columns]
    meta_frames = [f[meta_cols] for f in valid if all(c in f.columns for c in meta_cols)]
    metadata = (
        pd.concat(meta_frames, ignore_index=True)
        .drop_duplicates(subset=["playerId"], keep="last")
        if meta_frames
        else pd.DataFrame(columns=["playerId"])
    )

    combined = pd.concat(valid, ignore_index=True)
    numeric_cols = list(
        dict.fromkeys(
            c
            for c in combined.select_dtypes(include=[np.number]).columns
            if c not in ("playerId", "season", "season_year") and c not in meta_cols
        )
    )
    rate_cols = [c for c in numeric_cols if _is_rate_column(c)]
    count_cols = [c for c in numeric_cols if c not in rate_cols]

    # Sum additive counting statistics per player
    grouped_counts = combined.groupby("playerId", as_index=False)[count_cols].sum()

    # Weighted-average percentage/rank/rate columns by games_played
    if rate_cols and "games_played" in combined.columns:
        weights = pd.to_numeric(combined["games_played"], errors="coerce").fillna(0.0).clip(lower=0.0)
        weighted_df = pd.DataFrame({"playerId": combined["playerId"], "_w": weights})
        for col in rate_cols:
            vals = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
            weighted_df[col] = vals * weights

        agg_rates = weighted_df.groupby("playerId", as_index=False).sum()
        denom = agg_rates["_w"].replace(0.0, np.nan)
        for col in rate_cols:
            agg_rates[col] = (agg_rates[col] / denom).fillna(0.0)
        agg_rates = agg_rates.drop(columns=["_w"])
        result = pd.merge(metadata, grouped_counts, on="playerId", how="inner")
        result = pd.merge(result, agg_rates, on="playerId", how="inner")
    else:
        result = pd.merge(metadata, grouped_counts, on="playerId", how="inner")

    if "season" in valid[-1].columns and not valid[-1].empty and not result.empty:
        season_vals = pd.to_numeric(valid[-1]["season"], errors="coerce").dropna()
        if not season_vals.empty:
            result["season"] = int(season_vals.iloc[0])

    return result


def prep_dynamic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert cumulative counting stats to per-game rates cleanly without fragmentation."""
    out = df.loc[:, ~df.columns.duplicated()].copy()
    gp = pd.to_numeric(out["games_played"], errors="coerce").fillna(0.0).replace(0.0, 1.0)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    raw_feature_cols = [
        c
        for c in numeric_cols
        if c not in SKATER_EXCLUDE_COLS and not c.startswith("target_")
    ]

    new_cols: dict[str, pd.Series] = {}
    for col in raw_feature_cols:
        per_game_col = f"{col}_per_game"
        if (
            not _is_rate_column(col)
            and not col.endswith("_per_game")
            and per_game_col not in out.columns
            and per_game_col not in SKATER_EXCLUDE_COLS
            and not per_game_col.startswith("target_")
        ):
            new_cols[per_game_col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) / gp

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    if "onIce_xGoalsPercentage" in out.columns:
        out["onIce_xGoalsPercentage"] = pd.to_numeric(
            out["onIce_xGoalsPercentage"], errors="coerce"
        ).fillna(0.0)

    feature_cols = sorted(
        {
            c
            for c in out.select_dtypes(include=[np.number]).columns
            if c not in SKATER_EXCLUDE_COLS
            and not c.startswith("target_")
            and ("_per_game" in c or _is_rate_column(c))
        }
    )
    return out, feature_cols


def compute_empirical_bayes_n_half(prior_rates: np.ndarray | list[float], fallback: float = 9.23) -> float:
    """Estimate Gamma-Poisson prior half-life N_half = mean(rate) / var(rate)."""
    arr = np.asarray(prior_rates, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return fallback
    mean_rate = float(np.mean(arr))
    var_rate = float(np.var(arr))
    if var_rate <= 1e-8 or mean_rate <= 0.0:
        return fallback
    n_half = mean_rate / var_rate
    return round(float(np.clip(n_half, 2.0, 35.0)), 2)


def _read_optional_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def train_skater_goal_model(
    archive_dir: str,
    target_years: list[int],
    mode: str = "regular",
) -> dict[str, Any]:
    """Train an XGBoost skater goals-per-game bundle for 'regular' or 'playoffs' mode."""
    if not HAS_SKATER_ML:
        raise RuntimeError("Missing ML dependencies (xgboost, scikit-learn, joblib).")
    features_frames: list[pd.DataFrame] = []
    prior_rates_for_eb: list[float] = []

    for year in target_years:
        if mode == "playoffs":
            df_reg_curr = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_skaters.csv"))
            df_play_curr = _read_optional_csv(os.path.join(archive_dir, f"{year}_playoffs_skaters.csv"))
            df_play_prev = _read_optional_csv(os.path.join(archive_dir, f"{year - 1}_playoffs_skaters.csv"))
            if df_reg_curr.empty or df_play_curr.empty:
                continue

            df_feat = combine_skater_frames([df_play_prev, df_reg_curr])
            df_targ = normalize_skater_frame(df_play_curr)
            df_targ = df_targ[df_targ["games_played"] >= 4][["playerId", "games_played", "I_F_goals"]].rename(
                columns={"games_played": "target_gp", "I_F_goals": "target_goals"}
            )
            df_feat = df_feat[df_feat["games_played"] >= 20]
        else:
            # Predict year regular season G/GP from (year-1 regular + year-1 playoffs)
            df_reg_prev = _read_optional_csv(os.path.join(archive_dir, f"{year - 1}_regular_skaters.csv"))
            df_play_prev = _read_optional_csv(os.path.join(archive_dir, f"{year - 1}_playoffs_skaters.csv"))
            df_reg_curr = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_skaters.csv"))
            if df_reg_prev.empty or df_reg_curr.empty:
                continue

            df_feat = combine_skater_frames([df_reg_prev, df_play_prev])
            df_targ = normalize_skater_frame(df_reg_curr)
            df_targ = df_targ[df_targ["games_played"] >= 15][["playerId", "games_played", "I_F_goals"]].rename(
                columns={"games_played": "target_gp", "I_F_goals": "target_goals"}
            )
            df_feat = df_feat[df_feat["games_played"] >= 20]

        merged = pd.merge(df_feat, df_targ, on="playerId", how="inner")
        if merged.empty:
            continue
        merged["target_goals_per_game"] = merged["target_goals"] / merged["target_gp"]
        merged["season_year"] = year
        prior_rates_for_eb.extend((merged["I_F_goals"] / merged["games_played"]).tolist())
        features_frames.append(merged)

    if not features_frames:
        raise RuntimeError(f"No valid training rows found for skater {mode} model in {archive_dir}.")

    df_train = pd.concat(features_frames, ignore_index=True)
    df_prepped, candidate_features = prep_dynamic_features(df_train)

    # Filter out columns absent/empty across most rows
    valid_features = [c for c in candidate_features if c in df_prepped.columns]
    y_raw = df_prepped["target_goals_per_game"]
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    y_all = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float).reshape(-1)
    valid_mask = np.isfinite(y_all)
    if not np.any(valid_mask):
        raise RuntimeError(f"No finite training targets found for skater {mode} model in {archive_dir}.")

    X_df = df_prepped.loc[valid_mask, valid_features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X = imputer.fit_transform(X_df)
    y = y_all[valid_mask]

    # 5-fold cross-validation diagnostics
    cv_rmse = 0.0
    cv_mae = 0.0
    if len(y) >= 25:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in kf.split(X):
            fold_model = xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            fold_model.fit(X[tr_idx], y[tr_idx])
            fold_preds = np.asarray(fold_model.predict(X[va_idx]), dtype=float).reshape(-1)
            oof[va_idx] = np.clip(fold_preds, 0.0, 2.0)
        cv_rmse = float(np.sqrt(mean_squared_error(y, oof)))
        cv_mae = float(mean_absolute_error(y, oof))

    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # Feature importance summary
    importances = model.feature_importances_
    top_feats = (
        pd.DataFrame({"feature": valid_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    feat_imp_list = [
        {"feature": str(row["feature"]), "importance": round(float(row["importance"]), 4)}
        for _, row in top_feats.iterrows()
    ]

    n_half = compute_empirical_bayes_n_half(prior_rates_for_eb, fallback=9.23 if mode == "playoffs" else 14.0)

    return {
        "mode": mode,
        "model": model,
        "imputer": imputer,
        "features": valid_features,
        "n_half": n_half,
        "metrics": {
            "samples": int(len(y)),
            "feature_count": int(len(valid_features)),
            "cv_rmse": round(cv_rmse, 4),
            "cv_mae": round(cv_mae, 4),
            "years": [int(y_val) for y_val in sorted(df_train["season_year"].unique())],
        },
        "feature_importance": feat_imp_list,
    }


def save_skater_bundle(bundle: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def build_inference_baseline(archive_dir: str, active_mp_year: int) -> tuple[pd.DataFrame, str]:
    """Assemble current skater baseline combining prior season/playoffs safely when GP is small."""
    df_curr_reg = _read_optional_csv(os.path.join(archive_dir, f"{active_mp_year}_regular_skaters.csv"))
    df_prev_play = _read_optional_csv(os.path.join(archive_dir, f"{active_mp_year - 1}_playoffs_skaters.csv"))
    df_prev_reg = _read_optional_csv(os.path.join(archive_dir, f"{active_mp_year - 1}_regular_skaters.csv"))

    curr_clean = normalize_skater_frame(df_curr_reg)
    median_gp = float(curr_clean["games_played"].median()) if not curr_clean.empty else 0.0

    if not curr_clean.empty and median_gp >= 12.0:
        frames = [f for f in (df_prev_play, df_curr_reg) if not f.empty]
        source_label = f"{active_mp_year} Regular + {active_mp_year - 1} Playoffs"
    else:
        frames = [f for f in (df_prev_reg, df_prev_play, df_curr_reg) if not f.empty]
        source_label = f"{active_mp_year - 1} Reg/Playoffs + {active_mp_year} Regular"

    df_combined = combine_skater_frames(frames)
    if df_combined.empty:
        return pd.DataFrame(), source_label
    df_prepped, _ = prep_dynamic_features(df_combined)
    df_prepped["name_clean"] = df_prepped["name"].apply(clean_name)
    return df_prepped, source_label


def score_skater_candidates(
    bundle: dict[str, Any],
    df_baseline: pd.DataFrame,
    df_tims: pd.DataFrame,
    daily_matchups: dict[str, str],
    live_skaters: dict[str, dict[str, Any]],
    defense_multipliers: dict[str, float],
    active_mode: str,
) -> dict[str, Any]:
    """Score Tim Hortons Challenge groups AND all skaters playing on today/tomorrow's slate."""
    features: list[str] = bundle["features"]
    model = bundle["model"]
    imputer = bundle["imputer"]
    n_half = float(bundle["n_half"])

    # Pre-score the entire baseline skater pool cleanly in vectorized form
    scored_pool = df_baseline.copy()
    if scored_pool.empty:
        return {"n_half_used": n_half, "tims_groups": {}, "slate_top_scorers": []}
    X_all = scored_pool.reindex(columns=features, fill_value=0.0).replace([np.inf, -np.inf], np.nan)
    X_imp = imputer.transform(X_all)
    preds = np.asarray(model.predict(X_imp), dtype=float).reshape(-1)
    scored_pool["base_pred"] = np.clip(preds, 0.0, 2.0)

    lookup_by_name = {
        row["name_clean"]: row
        for _, row in scored_pool.sort_values("games_played", ascending=False)
        .drop_duplicates("name_clean", keep="first")
        .iterrows()
        if row["name_clean"]
    }

    def _evaluate_player(
        name: str,
        name_clean_key: str,
        team: str,
        position: str,
        opponent: str,
        group: int | None = None,
    ) -> dict[str, Any]:
        baseline_row = lookup_by_name.get(name_clean_key)
        if baseline_row is not None:
            base_pred = float(baseline_row["base_pred"])
            position = position or str(baseline_row.get("position", ""))
            mp_team = str(baseline_row.get("team", team))
            team = team if team and team != "UNK" else mp_team
        else:
            base_pred = 0.14

        live = live_skaters.get(name_clean_key, {})
        live_gp = int(live.get("live_gp", 0))
        live_goals = int(live.get("live_goals", 0))
        live_g_gp = (live_goals / live_gp) if live_gp > 0 else 0.0

        weight_live = live_gp / (live_gp + n_half) if (live_gp + n_half) > 0 else 0.0
        blended = (base_pred * (1.0 - weight_live)) + (live_g_gp * weight_live)

        opp_clean = normalize_team_abbrev(opponent)
        def_mult = float(defense_multipliers.get(opp_clean, 1.0))
        final_score = max(0.0, blended * def_mult)

        item = {
            "player": name,
            "team": team,
            "position": position or "F",
            "opponent": opp_clean if opp_clean != "UNK" else daily_matchups.get(team, "UNK"),
            "base_pred": round(base_pred, 3),
            "live_gp": live_gp,
            "live_goals": live_goals,
            "live_g_gp": round(live_g_gp, 3),
            "eb_live_weight_pct": round(weight_live * 100.0, 1),
            "def_multiplier": round(def_mult, 3),
            "final_score": round(final_score, 3),
            "mode": active_mode,
        }
        if group is not None:
            item["group"] = int(group)
        return item

    # 1. Tim Hortons Hockey Challenge Groups 1–3
    tims_groups: dict[str, list[dict[str, Any]]] = {}
    if df_tims is not None and not df_tims.empty:
        scored_tims = []
        for _, pick in df_tims.iterrows():
            p_team = normalize_team_abbrev(pick.get("team"))
            opp = daily_matchups.get(p_team, normalize_team_abbrev(pick.get("opponentTeam", "UNK")))
            scored_tims.append(
                _evaluate_player(
                    name=str(pick.get("player_name", "")),
                    name_clean_key=str(pick.get("name_clean", "")),
                    team=p_team,
                    position=str(pick.get("position", "")),
                    opponent=opp,
                    group=int(pick.get("group", 1)),
                )
            )
        df_t = pd.DataFrame(scored_tims).sort_values(["group", "final_score"], ascending=[True, False])
        for g_num, g_df in df_t.groupby("group"):
            tims_groups[str(int(g_num))] = g_df.to_dict(orient="records")

    # Update current team from live NHL roster stats when available (handles mid-season trades)
    if live_skaters:
        scored_pool["team"] = scored_pool.apply(
            lambda r: live_skaters.get(r["name_clean"], {}).get("team", r["team"]),
            axis=1,
        )

    # 2. Full Slate Top Projected Scorers across active games
    slate_scorers: list[dict[str, Any]] = []
    if daily_matchups:
        active_teams_set = set(daily_matchups.keys())
        df_slate = scored_pool[scored_pool["team"].isin(active_teams_set)].copy()
        # Exclude low-sample depth callups from top leaderboard when main roster exists
        df_slate = df_slate[df_slate["games_played"] >= 5] if len(df_slate[df_slate["games_played"] >= 5]) >= 10 else df_slate
        for _, row in df_slate.iterrows():
            p_team = str(row["team"])
            slate_scorers.append(
                _evaluate_player(
                    name=str(row.get("name", "")),
                    name_clean_key=str(row.get("name_clean", "")),
                    team=p_team,
                    position=str(row.get("position", "")),
                    opponent=daily_matchups.get(p_team, "UNK"),
                )
            )
        slate_scorers.sort(key=lambda x: x["final_score"], reverse=True)
        slate_scorers = slate_scorers[:40]

    return {
        "n_half_used": n_half,
        "tims_groups": tims_groups,
        "slate_top_scorers": slate_scorers,
    }
