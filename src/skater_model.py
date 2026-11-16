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

from src.availability import build_injury_matcher
from src.data_sync import PlayerMatcher, clean_name, normalize_team_abbrev


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


def compute_empirical_bayes_n_half(
    prior_rates: np.ndarray | list[float],
    fallback: float = 9.23,
    residual_mse: float | None = None,
    min_n_half: float = 9.0,
    max_n_half: float = 25.0,
) -> float:
    """Estimate Gamma-Poisson prior half-life N_half = mean(rate) / var_eff.

    Uses conditional out-of-fold residual MSE when available so shrinkage
    reflects uncertainty around the XGBoost player prior rather than global
    cross-position variance.
    """
    arr = np.asarray(prior_rates, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return fallback
    mean_rate = float(np.mean(arr))
    if residual_mse is not None and residual_mse > 1e-8 and mean_rate > 0.0:
        return round(float(np.clip(mean_rate / residual_mse, min_n_half, max_n_half)), 2)
    var_rate = float(np.var(arr))
    if var_rate <= 1e-8 or mean_rate <= 0.0:
        return fallback
    n_half = mean_rate / var_rate
    return round(float(np.clip(n_half, 2.0, max(35.0, max_n_half))), 2)


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
    y = np.clip(y_all[valid_mask], 0.0, 2.0)
    sample_weights = (
        pd.to_numeric(df_prepped.loc[valid_mask, "target_gp"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
        .to_numpy(dtype=float)
    )
    mcw = min(40, max(1, len(y) // 15))
    xgb_kwargs = {
        "objective": "count:poisson",
        "n_estimators": 200,
        "learning_rate": 0.04,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": mcw,
        "reg_lambda": 10.0,
        "random_state": 42,
    }

    # 5-fold GP-weighted cross-validation diagnostics
    cv_rmse = 0.0
    cv_mae = 0.0
    if len(y) >= 25:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=float)
        for tr_idx, va_idx in kf.split(X):
            fold_model = xgb.XGBRegressor(**xgb_kwargs)
            fold_model.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
            fold_preds = np.asarray(fold_model.predict(X[va_idx]), dtype=float).reshape(-1)
            oof[va_idx] = np.clip(fold_preds, 0.0, 2.0)
        w_sum = float(np.sum(sample_weights))
        if w_sum > 0.0:
            cv_mse_w = float(np.average((y - oof) ** 2, weights=sample_weights))
            cv_rmse = float(np.sqrt(cv_mse_w))
            cv_mae = float(np.average(np.abs(y - oof), weights=sample_weights))
        else:
            cv_rmse = float(np.sqrt(mean_squared_error(y, oof)))
            cv_mae = float(mean_absolute_error(y, oof))

    model = xgb.XGBRegressor(**xgb_kwargs)
    model.fit(X, y, sample_weight=sample_weights)

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

    residual_mse = (cv_rmse ** 2) if cv_rmse > 0.0 else None
    n_half = compute_empirical_bayes_n_half(
        prior_rates_for_eb,
        fallback=16.0 if mode == "playoffs" else 32.0,
        residual_mse=residual_mse,
        min_n_half=16.0,
        max_n_half=40.0,
    )

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


def load_skater_bundle(path: str) -> dict[str, Any]:
    if not HAS_SKATER_ML:
        raise RuntimeError("Missing ML dependencies (xgboost, scikit-learn, joblib).")
    return joblib.load(path)


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


def compute_skater_base_predictions(df_baseline: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """Predict clipped XGBoost goals-per-game base predictions for a prepared skater pool."""
    if df_baseline.empty:
        return np.array([], dtype=float)
    X_all = (
        df_baseline.reindex(columns=bundle["features"], fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
    )
    X_imp = bundle["imputer"].transform(X_all)
    preds = np.asarray(bundle["model"].predict(X_imp), dtype=float).reshape(-1)
    return np.clip(preds, 0.0, 2.0)


def blend_skater_goal_rate(
    base_pred: float,
    live_gp: float | int,
    live_goals: float | int,
    n_half: float,
    *,
    prior_sog_gp: float = 0.0,
    live_shots: float | int | None = None,
) -> tuple[float, float]:
    """Empirical Bayes blend of XGBoost goal rate with live goals and shrunk live shot pace.

    Returns (blended_goals_per_game, live_goal_weight).
    """
    gp_val = float(live_gp or 0.0)
    goals_val = float(live_goals or 0.0)
    live_g_gp = (goals_val / gp_val) if gp_val > 0.0 else 0.0
    weight_live = gp_val / (gp_val + n_half) if (gp_val + n_half) > 0.0 else 0.0
    goals_eb = (base_pred * (1.0 - weight_live)) + (live_g_gp * weight_live)

    if gp_val > 0.0 and live_shots is not None and prior_sog_gp > 0.0:
        shots_val = float(live_shots or 0.0)
        live_sog_gp = shots_val / gp_val
        w_shots = gp_val / (gp_val + 14.0)
        sog_eb = (prior_sog_gp * (1.0 - w_shots)) + (live_sog_gp * w_shots)
        prior_sh_pct = float(np.clip(base_pred / max(0.35, prior_sog_gp), 0.03, 0.22))
        sh_pct_eb = (goals_val + (150.0 * prior_sh_pct)) / (shots_val + 150.0)
        shot_rate_pred = sog_eb * sh_pct_eb
        blended = (0.70 * goals_eb) + (0.30 * shot_rate_pred)
    else:
        blended = goals_eb
    return float(blended), float(weight_live)


def score_skater_candidates(
    bundle: dict[str, Any],
    df_baseline: pd.DataFrame,
    df_tims: pd.DataFrame,
    daily_matchups: dict[str, str],
    live_skaters: dict[str, dict[str, Any]],
    defense_multipliers: dict[str, float],
    active_mode: str,
    injury_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score Tim Hortons Challenge groups AND all skaters playing on today/tomorrow's slate."""
    n_half = float(bundle["n_half"])
    injury_matcher = build_injury_matcher(injury_records or [])

    # Pre-score the entire baseline skater pool cleanly in vectorized form (or reuse precomputed snapshot)
    scored_pool = df_baseline.copy()
    if scored_pool.empty:
        return {"n_half_used": n_half, "tims_groups": {}, "slate_top_scorers": []}
    if "base_pred" in scored_pool.columns:
        scored_pool["base_pred"] = (
            pd.to_numeric(scored_pool["base_pred"], errors="coerce").fillna(0.14).clip(0.0, 2.0)
        )
    else:
        scored_pool["base_pred"] = compute_skater_base_predictions(scored_pool, bundle)

    dedup_subset = ["playerId", "name_clean"] if "playerId" in scored_pool.columns else ["name_clean"]
    pool_entries = []
    for _, row in (
        scored_pool.sort_values("games_played", ascending=False)
        .drop_duplicates(dedup_subset, keep="first")
        .iterrows()
    ):
        nc = str(row.get("name_clean", "") or "")
        nm = str(row.get("name", "") or nc)
        if not nc and not nm:
            continue
        pid_raw = row.get("playerId", 0)
        pool_entries.append(
            {
                "player_id": int(pid_raw) if pd.notna(pid_raw) and pid_raw else 0,
                "name": nm,
                "name_clean": nc,
                "team": row.get("team", "UNK"),
                "position": row.get("position", ""),
                "payload": row,
            }
        )
    pool_matcher = PlayerMatcher(pool_entries)

    live_entries = []
    for lkey, lval in sorted(
        (live_skaters or {}).items(),
        key=lambda kv: int(kv[1].get("live_gp", 0) or 0),
        reverse=True,
    ):
        live_entries.append(
            {
                "player_id": int(lval.get("player_id", 0) or 0),
                "name": str(lval.get("name") or lkey),
                "name_clean": str(lkey),
                "team": lval.get("team", "UNK"),
                "position": lval.get("position", ""),
                "payload": lval,
            }
        )
    live_matcher = PlayerMatcher(live_entries)

    def _evaluate_player(
        name: str,
        name_clean_key: str,
        team: str,
        position: str,
        opponent: str,
        group: int | None = None,
        player_id: int = 0,
    ) -> dict[str, Any]:
        live = (
            live_matcher.match(
                name=name,
                name_clean_key=name_clean_key,
                team=team,
                position=position,
                player_id=player_id,
                require_compatible_pos=True,
            )
            or {}
        )
        live_gp = int(live.get("live_gp", 0))
        live_goals = int(live.get("live_goals", 0))
        live_shots = int(live.get("live_shots", 0) or 0)
        live_g_gp = (live_goals / live_gp) if live_gp > 0 else 0.0
        resolved_pid = player_id or int(live.get("player_id", 0) or 0)

        baseline_row = pool_matcher.match(
            name=name,
            name_clean_key=name_clean_key,
            team=team,
            position=position,
            player_id=resolved_pid,
            require_compatible_pos=True,
        )
        prior_sog_gp = 0.0
        if baseline_row is not None:
            base_pred = float(baseline_row["base_pred"])
            position = position or str(baseline_row.get("position", ""))
            mp_team = str(baseline_row.get("team", team))
            team = team if team and team != "UNK" else mp_team
            sog_val = baseline_row.get("I_F_shotsOnGoal_per_game")
            if sog_val is not None and pd.notna(sog_val):
                prior_sog_gp = float(sog_val)
            else:
                raw_sog = baseline_row.get("I_F_shotsOnGoal")
                raw_gp = baseline_row.get("games_played")
                if raw_sog is not None and pd.notna(raw_sog) and raw_gp and float(raw_gp) > 0.0:
                    prior_sog_gp = float(raw_sog) / float(raw_gp)
        else:
            base_pred = 0.14

        blended, weight_live = blend_skater_goal_rate(
            base_pred=base_pred,
            live_gp=live_gp,
            live_goals=live_goals,
            n_half=n_half,
            prior_sog_gp=prior_sog_gp,
            live_shots=live_shots if "live_shots" in live else None,
        )

        opp_clean = normalize_team_abbrev(opponent)
        def_mult = float(defense_multipliers.get(opp_clean, 1.0))
        final_score = max(0.0, blended * def_mult)

        inj = injury_matcher.match(
            name=name,
            name_clean_key=name_clean_key,
            team=team,
            position=position,
            allow_initial_fallback=False,
            require_same_team=True,
            require_compatible_pos=True,
        )

        item = {
            "player": name,
            "team": team,
            "position": position or "F",
            "opponent": opp_clean if opp_clean != "UNK" else daily_matchups.get(team, "UNK"),
            "base_pred": round(base_pred, 3),
            "live_gp": live_gp,
            "live_goals": live_goals,
            "live_shots": live_shots,
            "live_g_gp": round(live_g_gp, 3),
            "eb_live_weight_pct": round(weight_live * 100.0, 1),
            "def_multiplier": round(def_mult, 3),
            "final_score": round(final_score, 3),
            "mode": active_mode,
            "injury_status": str(inj["status"]) if inj else None,
            "injury_out": bool(inj.get("is_out")) if inj else False,
            "injury_dtd": bool(inj.get("is_dtd")) if inj else False,
            "injury_note": str(inj.get("note") or "") if inj else "",
        }
        if group is not None:
            item["group"] = int(group)
        return item

    # 1. Tim Hortons Hockey Challenge Groups 1–3 (prioritize healthy/DTD skaters ahead of confirmed Out/IR)
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
        df_t = pd.DataFrame(scored_tims).sort_values(
            ["group", "injury_out", "final_score"], ascending=[True, True, False]
        )
        for g_num, g_df in df_t.groupby("group"):
            tims_groups[str(int(g_num))] = g_df.to_dict(orient="records")

    # Update current team from live NHL roster stats when available (handles mid-season trades)
    if live_skaters:
        def _resolve_live_team(r: pd.Series) -> str:
            pid_val = r.get("playerId", 0)
            pid_int = int(pid_val) if pd.notna(pid_val) and pid_val else 0
            matched_live = live_matcher.match(
                name=str(r.get("name", "") or ""),
                name_clean_key=str(r.get("name_clean", "") or ""),
                team=str(r.get("team", "UNK") or "UNK"),
                position=str(r.get("position", "") or ""),
                player_id=pid_int,
                require_compatible_pos=True,
            )
            return str(matched_live.get("team", r["team"])) if matched_live else str(r["team"])

        scored_pool["team"] = scored_pool.apply(_resolve_live_team, axis=1)

    # 2. Full Slate Top Projected Scorers across active games (exclude confirmed Out/IR skaters)
    slate_scorers: list[dict[str, Any]] = []
    if daily_matchups:
        active_teams_set = set(daily_matchups.keys())
        df_slate = scored_pool[scored_pool["team"].isin(active_teams_set)].copy()
        # Exclude low-sample depth callups from top leaderboard when main roster exists
        df_slate = df_slate[df_slate["games_played"] >= 5] if len(df_slate[df_slate["games_played"] >= 5]) >= 10 else df_slate
        for _, row in df_slate.iterrows():
            p_team = str(row["team"])
            pid_val = row.get("playerId", 0)
            evaluated = _evaluate_player(
                name=str(row.get("name", "")),
                name_clean_key=str(row.get("name_clean", "")),
                team=p_team,
                position=str(row.get("position", "")),
                opponent=daily_matchups.get(p_team, "UNK"),
                player_id=int(pid_val) if pd.notna(pid_val) and pid_val else 0,
            )
            if not evaluated["injury_out"]:
                slate_scorers.append(evaluated)
        slate_scorers.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "n_half_used": n_half,
        "tims_groups": tims_groups,
        "slate_top_scorers": slate_scorers[:40],
        "all_slate_scorers": slate_scorers,
    }
