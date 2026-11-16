"""Pairwise team matchup model, temperature calibration, power rankings, and Kelly edges."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import expit, logit

try:
    import joblib
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold

    HAS_GAME_ML = True
except ImportError:  # pragma: no cover
    joblib = None
    xgb = None
    accuracy_score = None
    log_loss = None
    StratifiedKFold = None
    HAS_GAME_ML = False

from src.data_sync import TEAM_FULL_TO_ABBREV, normalize_team_abbrev


TEAM_EXCLUDE_COLS = {
    "team",
    "season",
    "name",
    "team.1",
    "position",
    "situation",
    "playoff_games_played",
}


def apply_temperature(prob: float | np.ndarray, T: float = 5.0) -> float | np.ndarray:
    """Apply logit-space temperature scaling cleanly to prevent extreme overconfidence."""
    eps = 1e-7
    clipped = np.clip(prob, eps, 1.0 - eps)
    scaled = expit(logit(clipped) / max(1.0, float(T)))
    return float(scaled) if np.isscalar(prob) else scaled


def single_game_to_best_of_7_prob(p: float, wins_a: int = 0, wins_b: int = 0) -> float:
    """Compute exact Markov probability that Team A wins a best-of-7 series from (wins_a, wins_b)."""
    if wins_a >= 4:
        return 1.0
    if wins_b >= 4:
        return 0.0
    memo: dict[tuple[int, int], float] = {}
    q = 1.0 - p

    def dp(wa: int, wb: int) -> float:
        if wa >= 4:
            return 1.0
        if wb >= 4:
            return 0.0
        key = (wa, wb)
        if key not in memo:
            memo[key] = p * dp(wa + 1, wb) + q * dp(wa, wb + 1)
        return memo[key]

    return float(dp(wins_a, wins_b))


def expected_remaining_series_games(p: float, wins_a: int = 0, wins_b: int = 0) -> float:
    """Compute expected remaining games in a best-of-7 series from (wins_a, wins_b)."""
    if wins_a >= 4 or wins_b >= 4:
        return 0.0
    memo: dict[tuple[int, int], float] = {}
    q = 1.0 - p

    def dp(wa: int, wb: int) -> float:
        if wa >= 4 or wb >= 4:
            return 0.0
        key = (wa, wb)
        if key not in memo:
            memo[key] = 1.0 + p * dp(wa + 1, wb) + q * dp(wa, wb + 1)
        return memo[key]

    return float(dp(wins_a, wins_b))


def _read_optional_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def build_team_star_goalie_history(
    archive_dir: str,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compile multi-season team metrics, Top-3 skater Star Power, and Goalie GSAx tables."""
    team_history: list[pd.DataFrame] = []
    stars_rows: list[dict[str, Any]] = []
    goalies_rows: list[dict[str, Any]] = []

    for year in years:
        reg_teams = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_teams.csv"))
        play_teams = _read_optional_csv(os.path.join(archive_dir, f"{year}_playoffs_teams.csv"))
        reg_skaters = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_skaters.csv"))
        reg_goalies = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_goalies.csv"))

        if not reg_teams.empty:
            df_t = reg_teams[reg_teams["situation"] == "all"].copy() if "situation" in reg_teams.columns else reg_teams.copy()
            df_t["team"] = df_t["team"].apply(normalize_team_abbrev)
            df_t["season"] = year
            if not play_teams.empty:
                df_pt = play_teams[play_teams["situation"] == "all"].copy() if "situation" in play_teams.columns else play_teams.copy()
                df_pt["team"] = df_pt["team"].apply(normalize_team_abbrev)
                target_df = df_pt[["team", "games_played"]].rename(columns={"games_played": "playoff_games_played"})
                df_t = pd.merge(df_t, target_df, on="team", how="left")
                df_t["playoff_games_played"] = df_t["playoff_games_played"].fillna(0.0)
            else:
                df_t["playoff_games_played"] = 0.0
            team_history.append(df_t)

        if not reg_skaters.empty:
            df_s = reg_skaters[reg_skaters["situation"] == "all"].copy() if "situation" in reg_skaters.columns else reg_skaters.copy()
            df_s["team"] = df_s["team"].apply(normalize_team_abbrev)
            for team, grp in df_s.groupby("team"):
                top3 = grp.sort_values("I_F_points", ascending=False).head(3)
                stars_rows.append(
                    {
                        "season": year,
                        "team": str(team),
                        "Top3_Points": float(top3["I_F_points"].sum()),
                        "Top3_xGoals": float(top3["I_F_xGoals"].sum()) if "I_F_xGoals" in top3.columns else 0.0,
                    }
                )

        if not reg_goalies.empty:
            df_g = reg_goalies[reg_goalies["situation"] == "all"].copy() if "situation" in reg_goalies.columns else reg_goalies.copy()
            df_g["team"] = df_g["team"].apply(normalize_team_abbrev)
            for team, grp in df_g.groupby("team"):
                gsax = float(grp["goalsSavedAboveExpected"].sum()) if "goalsSavedAboveExpected" in grp.columns else 0.0
                if "games_played" in grp.columns and "savePercentage" in grp.columns and not grp.empty:
                    starter_sv = float(grp.sort_values("games_played", ascending=False).iloc[0]["savePercentage"])
                else:
                    starter_sv = 0.900
                goalies_rows.append(
                    {
                        "season": year,
                        "team": str(team),
                        "Team_GSAx": gsax,
                        "Starter_SavePct": starter_sv,
                    }
                )

    df_team_hist = pd.concat(team_history, ignore_index=True) if team_history else pd.DataFrame()
    df_stars = pd.DataFrame(stars_rows)
    df_goalies = pd.DataFrame(goalies_rows)
    return df_team_hist, df_stars, df_goalies


def _select_top_team_features(df_team_hist: pd.DataFrame, max_features: int = 20) -> list[str]:
    """Identify the strongest historical team features for predicting playoff success."""
    numeric_cols = [
        c
        for c in df_team_hist.select_dtypes(include=[np.number]).columns
        if c not in TEAM_EXCLUDE_COLS and not df_team_hist[c].isna().all()
    ]
    X = df_team_hist[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(df_team_hist["playoff_games_played"], errors="coerce").fillna(0.0)

    baseline_reg = xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
    baseline_reg.fit(X, y)
    importances = baseline_reg.feature_importances_
    ranked = (
        pd.DataFrame({"feature": numeric_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(max_features)
    )
    return ranked["feature"].tolist()


def _col_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely return a numeric Series matching df.index, defaulting to 0.0 when absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _merge_side_metrics(
    df_games: pd.DataFrame,
    metric_df: pd.DataFrame,
    team_col: str,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    """Merge a season/team table onto df_games cleanly and drop join keys without collisions."""
    if metric_df.empty:
        return df_games
    clean = metric_df.drop_duplicates(subset=["season", "team"])
    merged = df_games.merge(
        clean,
        left_on=["mp_lookup_season", team_col],
        right_on=["season", "team"],
        how="left",
        suffixes=("", "_dup"),
    )
    merged = merged.rename(columns=rename_map)
    return merged.drop(columns=[c for c in ("season", "team", "season_dup", "team_dup") if c in merged.columns])


def train_pairwise_game_model(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
) -> dict[str, Any]:
    """Train Pairwise XGBClassifier on Home-Away diffs (Star Power + GSAx + Momentum) and calibrate T."""
    if not HAS_GAME_ML:
        raise RuntimeError("Missing ML dependencies (xgboost, scikit-learn, joblib).")
    if df_team_hist.empty or df_master_logs.empty:
        raise RuntimeError("Cannot train pairwise game model: missing team history or NHL game logs.")

    base_features = _select_top_team_features(df_team_hist, max_features=20)
    df_games = df_master_logs[df_master_logs["Is_Home"] == 1].copy()
    df_games["season_year"] = df_games["Season"].astype(str).str[:4].astype(int)
    df_games["Team"] = df_games["Team"].apply(normalize_team_abbrev)
    df_games["Opponent"] = df_games["Opponent"].apply(normalize_team_abbrev)

    # Fallback map when MoneyPuck current season hasn't populated for every historical season
    available_seasons = sorted(df_team_hist["season"].unique())
    latest_mp_season = int(available_seasons[-1])
    df_games["mp_lookup_season"] = df_games["season_year"].apply(
        lambda y: y if y in available_seasons else latest_mp_season
    )

    stats_cols = ["team", "season"] + base_features
    df_stats_clean = df_team_hist[stats_cols].drop_duplicates(subset=["team", "season"]).copy()

    # Home team stats -> *_A
    df_games = (
        df_games.merge(
            df_stats_clean,
            left_on=["Team", "mp_lookup_season"],
            right_on=["team", "season"],
            how="inner",
        )
        .rename(columns={f: f"{f}_A" for f in base_features})
        .drop(columns=["team", "season"], errors="ignore")
    )

    # Away team stats -> *_B
    df_games = (
        df_games.merge(
            df_stats_clean,
            left_on=["Opponent", "mp_lookup_season"],
            right_on=["team", "season"],
            how="inner",
        )
        .rename(columns={f: f"{f}_B" for f in base_features})
        .drop(columns=["team", "season"], errors="ignore")
    )

    # Join Away pre-game momentum cleanly
    df_away = (
        df_master_logs[df_master_logs["Is_Home"] == 0][["Date", "Team", "Last_Game_Won", "Current_Streak"]]
        .drop_duplicates(subset=["Date", "Team"])
        .copy()
    )
    df_away["Team"] = df_away["Team"].apply(normalize_team_abbrev)
    df_games = df_games.merge(
        df_away,
        left_on=["Date", "Opponent"],
        right_on=["Date", "Team"],
        how="inner",
        suffixes=("_A", "_B"),
    )
    if "Team_A" not in df_games.columns and "Team" in df_games.columns:
        df_games = df_games.rename(columns={"Team": "Team_A"})

    df_games = _merge_side_metrics(
        df_games, df_stars, "Team_A", {"Top3_Points": "Top3_Points_A", "Top3_xGoals": "Top3_xGoals_A"}
    )
    df_games = _merge_side_metrics(
        df_games, df_stars, "Opponent", {"Top3_Points": "Top3_Points_B", "Top3_xGoals": "Top3_xGoals_B"}
    )
    df_games = _merge_side_metrics(
        df_games, df_goalies, "Team_A", {"Team_GSAx": "Team_GSAx_A", "Starter_SavePct": "Starter_SavePct_A"}
    )
    df_games = _merge_side_metrics(
        df_games, df_goalies, "Opponent", {"Team_GSAx": "Team_GSAx_B", "Starter_SavePct": "Starter_SavePct_B"}
    )

    # Vectorized diff computation
    extra_feats = ["Top3_Points", "Top3_xGoals", "Team_GSAx", "Starter_SavePct", "Last_Game_Won", "Streak"]
    diff_data: dict[str, pd.Series] = {}
    for feat in base_features + ["Top3_Points", "Top3_xGoals", "Team_GSAx", "Starter_SavePct"]:
        diff_data[f"{feat}_diff"] = _col_series(df_games, f"{feat}_A") - _col_series(df_games, f"{feat}_B")

    diff_data["Last_Game_Won_diff"] = _col_series(df_games, "Last_Game_Won_A") - _col_series(df_games, "Last_Game_Won_B")
    diff_data["Streak_diff"] = _col_series(df_games, "Current_Streak_A") - _col_series(df_games, "Current_Streak_B")

    feature_names = [f"{f}_diff" for f in base_features + extra_feats]
    X_pair = pd.DataFrame(diff_data)[feature_names].fillna(0.0)
    y_pair = df_games["Won"].astype(int).to_numpy()
    weights = np.where(df_games["GameType"] == 3, 5.0, 1.0)

    # 5-fold StratifiedKFold OOF calibration for Temperature T
    optimal_T = 1.5
    oof_ll_raw = 0.693
    oof_ll_cal = 0.693
    oof_acc = 0.5

    if len(y_pair) >= 50 and len(np.unique(y_pair)) == 2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y_pair))
        for tr_idx, va_idx in cv.split(X_pair, y_pair):
            fold_clf = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
                eval_metric="logloss",
            )
            fold_clf.fit(X_pair.iloc[tr_idx], y_pair[tr_idx], sample_weight=weights[tr_idx])
            oof_probs[va_idx] = fold_clf.predict_proba(X_pair.iloc[va_idx])[:, 1]

        oof_ll_raw = float(log_loss(y_pair, oof_probs))
        res = scipy.optimize.minimize(
            lambda t: log_loss(y_pair, apply_temperature(oof_probs, T=float(t[0]))),
            x0=[1.5],
            bounds=[(1.0, 15.0)],
        )
        optimal_T = float(res.x[0]) if res.success else 1.5
        calibrated_oof = apply_temperature(oof_probs, T=optimal_T)
        oof_ll_cal = float(log_loss(y_pair, calibrated_oof))
        oof_acc = float(accuracy_score(y_pair, (calibrated_oof >= 0.5).astype(int)))

    clf = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_pair, y_pair, sample_weight=weights)

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    top_feat_list = [
        {"feature": str(r["feature"]), "importance": round(float(r["importance"]), 4)}
        for _, r in imp_df.iterrows()
    ]

    return {
        "model": clf,
        "base_features": base_features,
        "feature_names": feature_names,
        "optimal_T": round(optimal_T, 3),
        "metrics": {
            "games_trained": int(len(y_pair)),
            "oof_log_loss_raw": round(oof_ll_raw, 4),
            "oof_log_loss_calibrated": round(oof_ll_cal, 4),
            "oof_accuracy": round(oof_acc, 4),
            "optimal_T": round(optimal_T, 3),
        },
        "feature_importance": top_feat_list,
    }


def save_pairwise_bundle(bundle: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def build_live_team_profiles(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    active_mp_year: int,
    latest_team_context: dict[str, dict[str, int]],
    base_features: list[str],
    playoff_teams_df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Assemble current 32-team feature profiles for inference and defense multipliers."""
    available_year = (
        active_mp_year
        if active_mp_year in df_team_hist["season"].values
        else int(df_team_hist["season"].max())
    )
    df_curr = df_team_hist[df_team_hist["season"] == available_year].drop_duplicates(subset=["team"]).copy()

    # Optional playoff stats blending if active playoff CSV exists
    play_clean = pd.DataFrame()
    if playoff_teams_df is not None and not playoff_teams_df.empty:
        play_clean = playoff_teams_df[playoff_teams_df["situation"] == "all"].copy() if "situation" in playoff_teams_df.columns else playoff_teams_df.copy()
        play_clean["team"] = play_clean["team"].apply(normalize_team_abbrev)

    profiles: dict[str, dict[str, Any]] = {}
    stars_curr = df_stars[df_stars["season"] == available_year].set_index("team") if not df_stars.empty else pd.DataFrame()
    goalies_curr = df_goalies[df_goalies["season"] == available_year].set_index("team") if not df_goalies.empty else pd.DataFrame()

    for _, row in df_curr.iterrows():
        team = str(row["team"])
        stat: dict[str, Any] = {f: float(row.get(f, 0.0) or 0.0) for f in base_features}
        gp_reg = float(row.get("games_played", 1.0) or 1.0)

        if not play_clean.empty and team in play_clean["team"].values:
            p_row = play_clean[play_clean["team"] == team].iloc[0]
            gp_play = float(p_row.get("games_played", 0.0) or 0.0)
            tot_gp = max(1.0, gp_reg + gp_play)
            for feat in base_features:
                if "Percentage" in feat:
                    stat[feat] = (stat[feat] * gp_reg + float(p_row.get(feat, 0.0) or 0.0) * gp_play) / tot_gp
                else:
                    stat[feat] = stat[feat] + float(p_row.get(feat, 0.0) or 0.0)

        if not stars_curr.empty and team in stars_curr.index:
            stat["Top3_Points"] = float(stars_curr.loc[team, "Top3_Points"])
            stat["Top3_xGoals"] = float(stars_curr.loc[team, "Top3_xGoals"])
        else:
            stat["Top3_Points"] = 0.0
            stat["Top3_xGoals"] = 0.0

        if not goalies_curr.empty and team in goalies_curr.index:
            stat["Team_GSAx"] = float(goalies_curr.loc[team, "Team_GSAx"])
            stat["Starter_SavePct"] = float(goalies_curr.loc[team, "Starter_SavePct"])
        else:
            stat["Team_GSAx"] = 0.0
            stat["Starter_SavePct"] = 0.900

        ctx = latest_team_context.get(team, {})
        stat["Last_Game_Won"] = int(ctx.get("Last_Game_Won", 0))
        stat["Streak"] = int(ctx.get("Current_Streak", 0))
        stat["games_played"] = int(row.get("games_played", 0) or 0)
        stat["goalsAgainst"] = float(row.get("goalsAgainst", 0.0) or 0.0)
        stat["xGoalsAgainst"] = float(row.get("xGoalsAgainst", 0.0) or 0.0)
        stat["highDangerxGoalsAgainst"] = float(row.get("highDangerxGoalsAgainst", 0.0) or 0.0)
        profiles[team] = stat

    return profiles, df_curr


def _pairwise_win_prob(
    team_a: str,
    team_b: str,
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
) -> float:
    if team_a not in profiles or team_b not in profiles:
        return 0.5
    stat_a, stat_b = profiles[team_a], profiles[team_b]
    feature_names: list[str] = bundle["feature_names"]
    row_dict = {}
    for feat in feature_names:
        base_key = feat[:-5] if feat.endswith("_diff") else feat
        row_dict[feat] = [float(stat_a.get(base_key, 0.0)) - float(stat_b.get(base_key, 0.0))]
    X_m = pd.DataFrame(row_dict)[feature_names].fillna(0.0)
    raw_prob = float(bundle["model"].predict_proba(X_m)[0][1])
    return float(apply_temperature(raw_prob, T=float(bundle["optimal_T"])))


def compute_power_rankings_and_defense(
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute AI Power Ratings across all teams and centered/damped 70/30 Defense Multiplier."""
    teams = sorted(profiles.keys())
    if not teams:
        return [], {}

    avg_win_probs: dict[str, float] = {}
    ai_power_dict: dict[str, float] = {}
    denom = max(1, len(teams) - 1)

    for t1 in teams:
        total_p = sum(
            _pairwise_win_prob(t1, t2, profiles, bundle)
            for t2 in teams
            if t2 != t1
        )
        avg_p = total_p / denom
        avg_win_probs[t1] = avg_p
        ai_power_dict[t1] = float(np.clip((avg_p * 12.0) + 3.0, 3.0, 15.0))

    # League averages for GA/GP, xGA/GP, HDxGA/GP
    ga_list, xga_list, hdxga_list = [], [], []
    per_team_rates: dict[str, tuple[float, float, float]] = {}
    for team, st in profiles.items():
        gp = max(1.0, float(st.get("games_played", 1.0)))
        ga_gp = float(st.get("goalsAgainst", 0.0)) / gp
        xga_gp = float(st.get("xGoalsAgainst", 0.0)) / gp
        hdxga_gp = float(st.get("highDangerxGoalsAgainst", 0.0)) / gp
        per_team_rates[team] = (ga_gp, xga_gp, hdxga_gp)
        ga_list.append(ga_gp)
        xga_list.append(xga_gp)
        hdxga_list.append(hdxga_gp)

    league_ga = max(1e-6, float(np.mean(ga_list)))
    league_xga = max(1e-6, float(np.mean(xga_list)))
    league_hdxga = max(1e-6, float(np.mean(hdxga_list)))

    raw_base_mults: dict[str, float] = {}
    raw_ai_factors: dict[str, float] = {}
    for team in teams:
        ga_gp, xga_gp, hdxga_gp = per_team_rates[team]
        raw_base_mults[team] = (
            (0.2 * (ga_gp / league_ga))
            + (0.4 * (xga_gp / league_xga))
            + (0.4 * (hdxga_gp / league_hdxga))
        )
        raw_ai_factors[team] = 1.0 / max(3.0, min(15.0, ai_power_dict[team]))

    mean_base = max(1e-6, float(np.mean(list(raw_base_mults.values()))))
    mean_ai = max(1e-6, float(np.mean(list(raw_ai_factors.values()))))
    raw_combined = {
        t: (0.7 * (raw_base_mults[t] / mean_base)) + (0.3 * (raw_ai_factors[t] / mean_ai))
        for t in teams
    }
    mean_combined = max(1e-6, float(np.mean(list(raw_combined.values()))))

    defense_multipliers: dict[str, float] = {}
    rankings: list[dict[str, Any]] = []

    for team in teams:
        ga_gp, xga_gp, hdxga_gp = per_team_rates[team]
        centered_def = raw_combined[team] / mean_combined
        def_mult = float(np.clip(1.0 + 0.35 * (centered_def - 1.0), 0.85, 1.15))
        defense_multipliers[team] = def_mult

        st = profiles[team]
        rankings.append(
            {
                "team": team,
                "ai_power": round(ai_power_dict[team], 2),
                "avg_win_prob": round(avg_win_probs[team] * 100.0, 1),
                "gp": int(st.get("games_played", 0)),
                "streak": int(st.get("Streak", 0)),
                "last_won": int(st.get("Last_Game_Won", 0)),
                "gsax": round(float(st.get("Team_GSAx", 0.0)), 1),
                "starter_sv_pct": round(float(st.get("Starter_SavePct", 0.9)), 3),
                "top3_pts": int(round(float(st.get("Top3_Points", 0.0)))),
                "top3_xgoals": round(float(st.get("Top3_xGoals", 0.0)), 1),
                "ga_gp": round(ga_gp, 2),
                "xga_gp": round(xga_gp, 2),
                "def_multiplier": round(def_mult, 3),
            }
        )

    rankings.sort(key=lambda r: r["avg_win_prob"], reverse=True)
    for idx, row in enumerate(rankings, 1):
        row["rank"] = idx

    return rankings, defense_multipliers


def evaluate_kelly_edge(
    model_prob: float,
    decimal_odds: float | None,
    bankroll: float = 1000.0,
    min_edge_pct: float = 1.0,
) -> dict[str, Any] | None:
    """Compute Betting Edge and Quarter-Kelly stake given model win prob and decimal odds."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    edge = (model_prob * decimal_odds) - 1.0
    if edge <= (min_edge_pct / 100.0):
        return None
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    f_star = max(0.0, (model_prob * b - q) / b)
    quarter_kelly = min(0.05, f_star / 4.0)
    return {
        "odds": round(decimal_odds, 2),
        "edge_pct": round(edge * 100.0, 2),
        "quarter_kelly_pct": round(quarter_kelly * 100.0, 2),
        "rec_bet": round(quarter_kelly * bankroll, 2),
    }


def _fetch_nhl_odds_lookup(odds_api_key: str, session: Any) -> dict[tuple[str, str], dict[str, float]]:
    """Fetch live Pinnacle / best NHL h2h moneyline odds when ODDS_API_KEY env var is present."""
    if not odds_api_key:
        return {}
    url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/?apiKey={odds_api_key}&regions=us,eu,uk&markets=h2h"
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            return {}
        data = res.json()
    except Exception:
        return {}

    odds_map: dict[tuple[str, str], dict[str, float]] = {}
    for game in data:
        h_full = game.get("home_team", "")
        a_full = game.get("away_team", "")
        home_abbr = TEAM_FULL_TO_ABBREV.get(h_full)
        away_abbr = TEAM_FULL_TO_ABBREV.get(a_full)
        if not home_abbr or not away_abbr:
            continue

        bookies = game.get("bookmakers", [])
        sharp = next((b for b in bookies if b.get("key") == "pinnacle"), None) or (bookies[0] if bookies else None)
        if not sharp:
            continue
        for market in sharp.get("markets", []):
            if market.get("key") != "h2h":
                continue
            h_odds, a_odds = 0.0, 0.0
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == h_full:
                    h_odds = float(outcome.get("price", 0.0) or 0.0)
                elif outcome.get("name") == a_full:
                    a_odds = float(outcome.get("price", 0.0) or 0.0)
            if h_odds > 1.0 and a_odds > 1.0:
                odds_map[(away_abbr, home_abbr)] = {
                    home_abbr: h_odds,
                    away_abbr: a_odds,
                    "bookie": sharp.get("title", "Sharp"),
                }
    return odds_map


def predict_upcoming_games(
    schedule_games: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
    odds_api_key: str = "",
    session: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate single-game win probabilities and optional active playoff series projections."""
    odds_map = _fetch_nhl_odds_lookup(odds_api_key, session) if (odds_api_key and session) else {}

    predictions: list[dict[str, Any]] = []
    seen_series: set[tuple[str, str, int]] = set()
    playoff_series: list[dict[str, Any]] = []

    for g in schedule_games:
        home = g["home"]
        away = g["away"]
        home_prob = _pairwise_win_prob(home, away, profiles, bundle)
        away_prob = 1.0 - home_prob
        st_h = profiles.get(home, {})
        st_a = profiles.get(away, {})

        game_odds = odds_map.get((away, home), {})
        home_edge = evaluate_kelly_edge(home_prob, game_odds.get(home))
        away_edge = evaluate_kelly_edge(away_prob, game_odds.get(away))

        predictions.append(
            {
                "id": g["id"],
                "date": g["date"],
                "is_today": g["is_today"],
                "start_time_et": g["start_time_et"],
                "game_state": g["game_state"],
                "game_type": g["game_type"],
                "away": away,
                "home": home,
                "away_score": g.get("away_score"),
                "home_score": g.get("home_score"),
                "away_prob": round(away_prob * 100.0, 1),
                "home_prob": round(home_prob * 100.0, 1),
                "favored": home if home_prob >= away_prob else away,
                "fav_prob": round(max(home_prob, away_prob) * 100.0, 1),
                "away_streak": int(st_a.get("Streak", 0)),
                "home_streak": int(st_h.get("Streak", 0)),
                "away_gsax": round(float(st_a.get("Team_GSAx", 0.0)), 1),
                "home_gsax": round(float(st_h.get("Team_GSAx", 0.0)), 1),
                "away_top3_pts": int(round(float(st_a.get("Top3_Points", 0.0)))),
                "home_top3_pts": int(round(float(st_h.get("Top3_Points", 0.0)))),
                "bookie": game_odds.get("bookie"),
                "away_edge": away_edge,
                "home_edge": home_edge,
            }
        )

        # Check active NHL Playoff Series state if game carries seriesStatus
        ss = g.get("series_status")
        if ss and g["game_type"] == 3:
            t1 = normalize_team_abbrev(ss.get("topSeedTeamAbbrev", home))
            t2 = normalize_team_abbrev(ss.get("bottomSeedTeamAbbrev", away))
            rnd = int(ss.get("round", 1) or 1)
            s_key = (min(t1, t2), max(t1, t2), rnd)
            if s_key not in seen_series:
                seen_series.add(s_key)
                w1 = int(ss.get("topSeedWins", 0) or 0)
                w2 = int(ss.get("bottomSeedWins", 0) or 0)
                p_single = _pairwise_win_prob(t1, t2, profiles, bundle)
                p_series_t1 = single_game_to_best_of_7_prob(p_single, w1, w2)
                exp_rem = expected_remaining_series_games(p_single, w1, w2)
                playoff_series.append(
                    {
                        "round": rnd,
                        "top_seed": t1,
                        "top_wins": w1,
                        "bottom_seed": t2,
                        "bottom_wins": w2,
                        "top_series_prob": round(p_series_t1 * 100.0, 1),
                        "bottom_series_prob": round((1.0 - p_series_t1) * 100.0, 1),
                        "expected_remaining_games": round(exp_rem, 1),
                    }
                )

    return predictions, playoff_series
