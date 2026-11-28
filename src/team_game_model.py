"""Pairwise team matchup model, temperature calibration, power rankings, and Kelly edges."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import expit, logit

try:
    import joblib
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_GAME_ML = True
except ImportError:  # pragma: no cover
    joblib = None
    xgb = None
    SimpleImputer = None
    LogisticRegression = None
    accuracy_score = None
    brier_score_loss = None
    log_loss = None
    roc_auc_score = None
    GroupKFold = None
    StratifiedKFold = None
    make_pipeline = None
    StandardScaler = None
    HAS_GAME_ML = False

from src.data_sync import TEAM_FULL_TO_ABBREV, normalize_team_abbrev
from src.team_geo import haversine_miles, team_coords, travel_miles, tz_shift_hours

FRANCHISE_PRIOR_MAP: dict[str, str] = {"UTA": "ARI"}
ODDS_API_BOOKMAKERS = "pinnacle,hardrockbet,betonlineag,draftkings,fanduel,betmgm,betrivers,espnbet,bovada,playnow_ca"


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


HOME_ICE_PATTERN_22111: tuple[bool, ...] = (True, True, False, False, True, False, True)


def series_prob_22111(
    p_home: float,
    p_away: float,
    wins_high: int = 0,
    wins_low: int = 0,
) -> float:
    """Exact NHL 2-2-1-1-1 Best-of-7 win probability for the higher seed.

    Games 1, 2, 5, 7 use ``p_home`` (high seed at home); Games 3, 4, 6 use
    ``p_away`` (high seed on road).
    """
    if wins_high >= 4:
        return 1.0
    if wins_low >= 4:
        return 0.0
    p_h = float(np.clip(p_home, 0.0, 1.0))
    p_a = float(np.clip(p_away, 0.0, 1.0))
    memo: dict[tuple[int, int], float] = {}

    def dp(wh: int, wl: int) -> float:
        if wh >= 4:
            return 1.0
        if wl >= 4:
            return 0.0
        key = (wh, wl)
        if key not in memo:
            game_idx = wh + wl
            p_g = p_h if (game_idx < len(HOME_ICE_PATTERN_22111) and HOME_ICE_PATTERN_22111[game_idx]) else p_a
            memo[key] = p_g * dp(wh + 1, wl) + (1.0 - p_g) * dp(wh, wl + 1)
        return memo[key]

    return float(dp(wins_high, wins_low))


def expected_remaining_games_22111(
    p_home: float,
    p_away: float,
    wins_high: int = 0,
    wins_low: int = 0,
) -> float:
    """Compute expected remaining games under 2-2-1-1-1 home-ice alternation."""
    if wins_high >= 4 or wins_low >= 4:
        return 0.0
    p_h = float(np.clip(p_home, 0.0, 1.0))
    p_a = float(np.clip(p_away, 0.0, 1.0))
    memo: dict[tuple[int, int], float] = {}

    def dp(wh: int, wl: int) -> float:
        if wh >= 4 or wl >= 4:
            return 0.0
        key = (wh, wl)
        if key not in memo:
            game_idx = wh + wl
            p_g = p_h if (game_idx < len(HOME_ICE_PATTERN_22111) and HOME_ICE_PATTERN_22111[game_idx]) else p_a
            memo[key] = 1.0 + p_g * dp(wh + 1, wl) + (1.0 - p_g) * dp(wh, wl + 1)
        return memo[key]

    return float(dp(wins_high, wins_low))


def _read_optional_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _is_team_rate_column(col: str) -> bool:
    low = col.lower()
    return col.endswith("_per_game") or any(tok in low for tok in ("percentage", "pct", "save", "shooting"))


def normalize_team_frame_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-game rate columns alongside raw MoneyPuck team totals cleanly without fragmentation."""
    if df.empty or "games_played" not in df.columns:
        return df.copy()
    out = df.loc[:, ~df.columns.duplicated()].copy()
    gp = pd.to_numeric(out["games_played"], errors="coerce").fillna(1.0).clip(lower=1.0)
    skip = set(TEAM_EXCLUDE_COLS) | {"games_played", "playoff_games_played"}
    new_cols: dict[str, pd.Series] = {}
    for col in out.select_dtypes(include=[np.number]).columns:
        if col in skip or _is_team_rate_column(col):
            continue
        pg_col = f"{col}_per_game"
        if pg_col not in out.columns:
            new_cols[pg_col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) / gp
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


def _row_venue(row) -> str:
    try:
        if int(row.get("Is_Home", 0) or 0) == 1:
            return str(row.get("Team", "") or "").strip().upper()
    except Exception:
        pass
    opp = row.get("Opp", None)
    if opp is None:
        opp = row.get("Opponent", None)
    return str(opp or "").strip().upper()


def add_causal_game_log_features(df_logs: pd.DataFrame) -> pd.DataFrame:
    """Add strictly pre-game rest days, B2B/3-in-4 flags, cumulative & EWMA form per (Season, Team)."""
    if df_logs.empty:
        return df_logs.copy()
    parts: list[pd.DataFrame] = []
    for _, grp in df_logs.groupby(["Season", "Team"], sort=False):
        g = grp.sort_values("Date").copy().reset_index(drop=True)
        dates = pd.to_datetime(g["Date"], errors="coerce")
        diff_days = dates.diff().dt.days
        rest_days = (diff_days - 1.0).clip(lower=0.0, upper=7.0).fillna(3.0)
        g["Is_B2B"] = (rest_days == 0.0).astype(float)
        g["Rest_Days"] = rest_days

        # 3-games-in-4-nights flag: >= 2 games played in prior 3 calendar days (d - 3 .. d - 1)
        d_vals = dates.to_numpy(dtype="datetime64[D]")
        is_3in4: list[float] = []
        for i, d_cur in enumerate(d_vals):
            if np.isnat(d_cur):
                is_3in4.append(0.0)
                continue
            cnt = 0
            for j in range(i - 1, -1, -1):
                if np.isnat(d_vals[j]):
                    continue
                if 0 < int((d_cur - d_vals[j]).astype(int)) <= 3:
                    cnt += 1
                else:
                    break
            is_3in4.append(1.0 if cnt >= 2 else 0.0)
        g["Is_3in4"] = is_3in4

        won = pd.to_numeric(g.get("Won", 0.0), errors="coerce").fillna(0.0)
        t_score = pd.to_numeric(g.get("Team_Score", 0.0), errors="coerce").fillna(0.0)
        o_score = pd.to_numeric(g.get("Opp_Score", 0.0), errors="coerce").fillna(0.0)
        gd = t_score - o_score
        gp_prev = pd.Series(np.arange(len(g), dtype=float), index=g.index)

        g["Causal_GP"] = gp_prev
        g["Causal_WinPct"] = (won.shift(1).cumsum().fillna(0.0) + 2.0) / (gp_prev + 4.0)
        g["Causal_GD_GP"] = gd.shift(1).cumsum().fillna(0.0) / (gp_prev + 8.0)

        w10_sum = won.shift(1).rolling(10, min_periods=1).sum().fillna(0.0)
        w10_cnt = won.shift(1).rolling(10, min_periods=1).count().fillna(0.0)
        g["Rolling10_WinPct"] = (w10_sum + 2.5) / (w10_cnt + 5.0)
        gd_sum = gd.shift(1).rolling(10, min_periods=1).sum().fillna(0.0)
        g["Rolling10_GoalDiff"] = gd_sum / (w10_cnt + 5.0)
        g["EWMA_WinPct"] = won.shift(1).ewm(halflife=15, min_periods=1).mean().fillna(0.5)
        g["EWMA_GD"] = gd.shift(1).ewm(halflife=15, min_periods=1).mean().fillna(0.0)

        # Fatigue: venues
        venues = [_row_venue(row) for _, row in g.iterrows()]
        if "Is_Home" in g.columns:
            is_home_list = [int(x or 0) for x in pd.to_numeric(g["Is_Home"], errors="coerce").fillna(0.0).tolist()]
        else:
            is_home_list = [0 for _ in range(len(g))]

        games_last7: list[float] = []
        games_last5: list[float] = []
        travel_last7: list[float] = []
        tz_shift: list[float] = []
        road_trip: list[float] = []
        rest_x_travel: list[float] = []

        for i in range(len(g)):
            cur_date = pd.to_datetime(g["Date"].iloc[i], errors="coerce")
            tonight_venue = venues[i]
            prior_indices = list(range(i))
            # prior dates
            prior_dates = [pd.to_datetime(g["Date"].iloc[j], errors="coerce") for j in prior_indices]
            prior_dates_valid = [ (j, d) for j, d in zip(prior_indices, prior_dates) if pd.notna(d) and pd.notna(cur_date) ]
            # Games_Last7: prior games with 1 <= (cur - d).days <= 7
            gl7 = 0
            for j, d in prior_dates_valid:
                delta = int((cur_date - d).days)
                if 1 <= delta <= 7:
                    gl7 += 1
            games_last7.append(float(gl7))
            games_last5.append(
                float(sum(1 for _, d in prior_dates_valid if 1 <= int((cur_date - d).days) <= 5))
            )

            # recent prior within 7 days (for travel chain)
            recent_prior = [(j, d) for j, d in prior_dates_valid if 1 <= int((cur_date - d).days) <= 7]
            # travel sum across consecutive venue pairs among recent_prior
            travel = 0.0
            for k in range(1, len(recent_prior)):
                v_prev = venues[recent_prior[k-1][0]]
                v_curr = venues[recent_prior[k][0]]
                travel += travel_miles(v_prev, v_curr)
            # last prior venue (most recent prior game overall)
            last_prior_venue = None
            if prior_dates_valid:
                # most recent by date (already sorted), last in list
                last_prior_venue = venues[prior_dates_valid[-1][0]]
            # plus last prior venue -> tonight
            if last_prior_venue:
                travel += travel_miles(last_prior_venue, tonight_venue)
            travel_last7.append(float(travel))

            # TZ_Shift
            if last_prior_venue and tonight_venue:
                tzv = abs(tz_shift_hours(last_prior_venue, tonight_venue, cur_date))
            else:
                tzv = 0.0
            tz_shift.append(float(tzv))

            # RoadTrip_Length: trailing consecutive away games ending at last prior game
            road = 0
            if prior_dates_valid:
                # iterate prior games reverse
                for j in range(i - 1, -1, -1):
                    # away if Is_Home != 1
                    if int(is_home_list[j] or 0) == 1:
                        break
                    road += 1
                    if road >= 7:
                        break
            road_trip.append(float(min(road, 7)))

            # Rest_Days for this row already computed (rest_days series). Use it.
            rd = float(rest_days.iloc[i]) if i < len(rest_days) else 0.0
            rest_x_travel.append(float(rd * (travel / 1000.0)))

        g["Games_Last7"] = games_last7
        g["Games_Last5"] = games_last5
        g["Travel_Last7"] = travel_last7
        g["TZ_Shift"] = tz_shift
        g["RoadTrip_Length"] = road_trip
        g["Rest_x_Travel"] = rest_x_travel
        # Binary/threshold variants + rest interaction for the walk-forward experiment.
        gl7 = pd.to_numeric(g["Games_Last7"], errors="coerce").fillna(0.0)
        gl5 = pd.to_numeric(g["Games_Last5"], errors="coerce").fillna(0.0)
        g["Games_Last7_ge4"] = (gl7 >= 4.0).astype(float)
        g["Games_Last5_ge3"] = (gl5 >= 3.0).astype(float)
        g["Rest_x_Games7_ge4"] = pd.to_numeric(g["Rest_Days"], errors="coerce").fillna(0.0) * g["Games_Last7_ge4"]

        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df_logs.copy()


def compute_causal_elo_history(
    df_logs: pd.DataFrame,
    *,
    k_factor: float = 6.5,
    home_adv: float = 30.0,
    carryover: float = 0.65,
) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, float]]:
    """
    Chronologically compute pre-game (Elo_Home, Elo_Away) per unique (Date, HomeTeam) game
    plus final post-game team Elo ratings, with seasonal mean regression & ARI->UTA continuity.
    """
    if df_logs.empty or "Is_Home" not in df_logs.columns:
        return {}, {}

    work = df_logs[df_logs["Is_Home"] == 1].copy()
    if work.empty:
        return {}, {}
    work["Team"] = work["Team"].apply(normalize_team_abbrev)
    work["Opponent"] = work["Opponent"].apply(normalize_team_abbrev)
    work["season_year"] = work["Season"].astype(str).str[:4].astype(int)
    work = work.sort_values(["Date", "Team"]).drop_duplicates(subset=["Date", "Team"])

    elo: dict[str, float] = {}
    last_season: int | None = None
    pregame_map: dict[tuple[str, str], tuple[float, float]] = {}

    for row in work.itertuples(index=False):
        sy = int(row.season_year)
        if last_season is not None and sy != last_season:
            regressed = {t: 1500.0 + carryover * (val - 1500.0) for t, val in elo.items()}
            for new_tm, old_tm in FRANCHISE_PRIOR_MAP.items():
                if new_tm not in regressed and old_tm in regressed:
                    regressed[new_tm] = regressed[old_tm]
            elo = regressed
        last_season = sy

        home_tm = str(row.Team)
        away_tm = str(row.Opponent)
        d_str = str(row.Date)
        e_h = elo.get(home_tm, 1500.0)
        e_a = elo.get(away_tm, 1500.0)
        pregame_map[(d_str, home_tm)] = (e_h, e_a)

        exp_h = 1.0 / (1.0 + 10.0 ** ((e_a - (e_h + home_adv)) / 400.0))
        won = float(getattr(row, "Won", 0.0) or 0.0)
        mov = abs(float(getattr(row, "Team_Score", 0.0) or 0.0) - float(getattr(row, "Opp_Score", 0.0) or 0.0))
        mov_mult = float(np.log(max(1.0, mov) + 1.0) * (2.2 / ((abs(e_h - e_a) * 0.001) + 2.2)))
        delta = k_factor * mov_mult * (won - exp_h)
        elo[home_tm] = e_h + delta
        elo[away_tm] = e_a - delta

    return pregame_map, elo


def context_spans_multiple_seasons(df_logs: pd.DataFrame | None) -> bool:
    """True when the logs frame covers >1 season.

    `Causal_Elo` is trained on the full multi-season history: it regresses each
    team toward 1500 between seasons (carryover 0.65). Recomputing it from a
    current-season-only frame (what an infer-only run downloads) restarts every
    team at 1500 and skews the model's highest-importance feature.
    """
    if df_logs is None or getattr(df_logs, "empty", True):
        return False
    if "Season" not in df_logs.columns:
        return False
    return len({str(s)[:4] for s in df_logs["Season"].astype(str)}) > 1


def extract_latest_team_form(df_logs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute each team's live entering-tonight rolling form, Elo, & recent dates."""
    form_map: dict[str, dict[str, Any]] = {}
    if df_logs.empty:
        return form_map
    _, latest_elo = compute_causal_elo_history(df_logs)
    df_work = df_logs.copy()
    df_work["Team"] = df_work["Team"].apply(normalize_team_abbrev)
    df_work = df_work.sort_values("Date")
    for team, grp in df_work.groupby("Team"):
        last_season = str(grp.iloc[-1].get("Season", ""))
        season_full = grp[grp["Season"].astype(str) == last_season]
        season_grp = season_full.tail(10)
        n_full = float(len(season_full))
        w_full = float(pd.to_numeric(season_full["Won"], errors="coerce").fillna(0.0).sum())
        gd_series_full = (
            pd.to_numeric(season_full["Team_Score"], errors="coerce").fillna(0.0)
            - pd.to_numeric(season_full["Opp_Score"], errors="coerce").fillna(0.0)
        )
        n_games = float(len(season_grp))
        wins = float(pd.to_numeric(season_grp["Won"], errors="coerce").fillna(0.0).sum())
        gd = float(
            (
                pd.to_numeric(season_grp["Team_Score"], errors="coerce").fillna(0.0)
                - pd.to_numeric(season_grp["Opp_Score"], errors="coerce").fillna(0.0)
            ).sum()
        )
        won_full_s = pd.to_numeric(season_full["Won"], errors="coerce").fillna(0.0)
        ewma_win = float(won_full_s.ewm(halflife=15, min_periods=1).mean().iloc[-1]) if n_full > 0 else 0.5
        ewma_gd = float(gd_series_full.ewm(halflife=15, min_periods=1).mean().iloc[-1]) if n_full > 0 else 0.0

        last_row = grp.iloc[-1]
        last_won = int(last_row.get("Won", 0) or 0)
        prev_streak = int(last_row.get("Current_Streak", 0) or 0)
        next_streak = (prev_streak + 1 if prev_streak > 0 else 1) if last_won == 1 else (
            prev_streak - 1 if prev_streak < 0 else -1
        )
        recent = grp.tail(10)
        recent_dates = [str(d) for d in recent["Date"].tolist() if d is not None and str(d).strip() != ""]
        recent_is_home = [int(x or 0) for x in pd.to_numeric(recent.get("Is_Home", 0.0), errors="coerce").fillna(0.0).tolist()] if "Is_Home" in recent.columns else [0 for _ in recent_dates]
        recent_venues = []
        for _, rrow in recent.iterrows():
            if int(rrow.get("Is_Home", 0) or 0) == 1:
                recent_venues.append(str(rrow.get("Team", "") or "").strip().upper())
            else:
                opp = rrow.get("Opp", None)
                if opp is None:
                    opp = rrow.get("Opponent", None)
                recent_venues.append(str(opp or "").strip().upper())
        t_str = str(team)
        form_map[t_str] = {
            "Last_Game_Won": last_won,
            "Current_Streak": next_streak,
            "Last_Game_Date": str(last_row.get("Date", "") or ""),
            "Recent_Game_Dates": recent_dates,
            "Recent_Game_Venues": recent_venues,
            "Recent_Is_Home": recent_is_home,
            "Rolling10_WinPct": round((wins + 2.5) / (n_games + 5.0), 4),
            "Rolling10_GoalDiff": round(gd / (n_games + 5.0), 4),
            "Causal_GP": int(n_full),
            "Causal_WinPct": round((w_full + 2.0) / (n_full + 4.0), 4),
            "Causal_GD_GP": round(float(gd_series_full.sum()) / (n_full + 8.0), 4),
            "EWMA_WinPct": round(ewma_win, 4),
            "EWMA_GD": round(ewma_gd, 4),
            "Causal_Elo": round(float(latest_elo.get(t_str, 1500.0)), 2),
        }
    return form_map


def build_team_star_goalie_history(
    archive_dir: str,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compile multi-season team rate metrics, Top-3 skater Star Power, and Goalie GSAx tables."""
    team_history: list[pd.DataFrame] = []
    stars_rows: list[dict[str, Any]] = []
    goalies_rows: list[dict[str, Any]] = []

    for year in years:
        reg_teams = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_teams.csv"))
        play_teams = _read_optional_csv(os.path.join(archive_dir, f"{year}_playoffs_teams.csv"))
        reg_skaters = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_skaters.csv"))
        reg_goalies = _read_optional_csv(os.path.join(archive_dir, f"{year}_regular_goalies.csv"))

        team_gp_map: dict[str, float] = {}
        if not reg_teams.empty:
            df_t = reg_teams[reg_teams["situation"] == "all"].copy() if "situation" in reg_teams.columns else reg_teams.copy()
            df_t["team"] = df_t["team"].apply(normalize_team_abbrev)
            df_t["season"] = year
            df_t = normalize_team_frame_per_game(df_t)
            for _, trow in df_t.iterrows():
                team_gp_map[str(trow["team"])] = max(1.0, float(trow.get("games_played", 1.0) or 1.0))
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
                t_str = str(team)
                gp = pd.to_numeric(grp.get("games_played", 1.0), errors="coerce").fillna(0.0)
                pts = pd.to_numeric(grp.get("I_F_points", 0.0), errors="coerce").fillna(0.0)
                xg = pd.to_numeric(grp.get("I_F_xGoals", 0.0), errors="coerce").fillna(0.0)
                safe_gp = gp.where(gp > 0.0)
                work = grp.assign(
                    _ppg=(pts / safe_gp).fillna(0.0),
                    _xgpg=(xg / safe_gp).fillna(0.0),
                    _gp=gp,
                )
                # Per-game definition (aligned with the live roster-aware table). Avoid
                # tiny samples when the team has enough established skaters.
                eligible = work[work["_gp"] >= 10.0] if int((work["_gp"] >= 10.0).sum()) >= 3 else work
                top3 = eligible.sort_values("_ppg", ascending=False).head(3)
                depth_pool = eligible.drop(top3.index)
                stars_rows.append(
                    {
                        "season": year,
                        "team": t_str,
                        "Top3_Points": float(top3["I_F_points"].sum()),
                        "Top3_xGoals": float(top3["I_F_xGoals"].sum()) if "I_F_xGoals" in top3.columns else 0.0,
                        "Top3_Points_per_game": float(top3["_ppg"].sum()),
                        "Top3_xGoals_per_game": float(top3["_xgpg"].sum()),
                        "Depth_Points_per_game": float(depth_pool["_ppg"].sum()),
                        "Depth_xGoals_per_game": float(depth_pool["_xgpg"].sum()),
                    }
                )

        if not reg_goalies.empty:
            df_g = reg_goalies[reg_goalies["situation"] == "all"].copy() if "situation" in reg_goalies.columns else reg_goalies.copy()
            df_g["team"] = df_g["team"].apply(normalize_team_abbrev)
            for team, grp in df_g.groupby("team"):
                t_str = str(team)
                t_gp = team_gp_map.get(t_str, max(1.0, float(grp["games_played"].sum() if "games_played" in grp.columns else 1.0)))
                if "goalsSavedAboveExpected" in grp.columns:
                    gsax = float(pd.to_numeric(grp["goalsSavedAboveExpected"], errors="coerce").fillna(0.0).sum())
                elif "xGoals" in grp.columns and "goals" in grp.columns:
                    xg = pd.to_numeric(grp["xGoals"], errors="coerce").fillna(0.0)
                    gl = pd.to_numeric(grp["goals"], errors="coerce").fillna(0.0)
                    gsax = float((xg - gl).sum())
                else:
                    gsax = 0.0

                starter_sv = 0.900
                if not grp.empty:
                    sort_col = "games_played" if "games_played" in grp.columns else ("icetime" if "icetime" in grp.columns else None)
                    top_g = grp.sort_values(sort_col, ascending=False).iloc[0] if sort_col else grp.iloc[0]
                    if "savePercentage" in grp.columns and pd.notna(top_g.get("savePercentage")):
                        starter_sv = float(top_g["savePercentage"])
                    elif "ongoal" in grp.columns and "goals" in grp.columns:
                        og = float(pd.to_numeric(top_g.get("ongoal"), errors="coerce") or 0.0)
                        gl = float(pd.to_numeric(top_g.get("goals"), errors="coerce") or 0.0)
                        if og > 0.0:
                            starter_sv = (og - gl) / og

                goalies_rows.append(
                    {
                        "season": year,
                        "team": t_str,
                        "Team_GSAx": gsax,
                        "Team_GSAx_per_game": gsax / t_gp,
                        "Starter_SavePct": starter_sv,
                    }
                )

    df_team_hist = pd.concat(team_history, ignore_index=True) if team_history else pd.DataFrame()
    df_stars = pd.DataFrame(stars_rows)
    df_goalies = pd.DataFrame(goalies_rows)
    return df_team_hist, df_stars, df_goalies


def load_active_goalie_lookup(
    archive_dir: str,
    active_mp_year: int,
) -> dict[str, list[dict[str, Any]]]:
    """Load per-team individual goalie GSAx & SV% stats from active-season MoneyPuck CSV."""
    for yr in (active_mp_year, active_mp_year - 1):
        df_g = _read_optional_csv(os.path.join(archive_dir, f"{yr}_regular_goalies.csv"))
        if df_g.empty:
            continue
        if "situation" in df_g.columns:
            df_g = df_g[df_g["situation"] == "all"].copy()
        if df_g.empty:
            continue
        df_g["team"] = df_g["team"].apply(normalize_team_abbrev)
        by_team: dict[str, list[dict[str, Any]]] = {}
        for _, r in df_g.iterrows():
            team = str(r.get("team", "UNK"))
            name = str(r.get("name", "") or "").strip()
            if team == "UNK" or not name:
                continue
            gp = float(pd.to_numeric(r.get("games_played"), errors="coerce") or 0.0)
            og = float(pd.to_numeric(r.get("ongoal"), errors="coerce") or 0.0)
            gl = float(pd.to_numeric(r.get("goals"), errors="coerce") or 0.0)
            xg = float(pd.to_numeric(r.get("xGoals"), errors="coerce") or 0.0)
            gsax = float(r["goalsSavedAboveExpected"]) if "goalsSavedAboveExpected" in r and pd.notna(r.get("goalsSavedAboveExpected")) else (xg - gl)
            sv = float(r["savePercentage"]) if "savePercentage" in r and pd.notna(r.get("savePercentage")) else ((og - gl) / og if og > 0.0 else 0.900)
            by_team.setdefault(team, []).append(
                {
                    "player_id": int(pd.to_numeric(r.get("playerId"), errors="coerce") or 0),
                    "name": name,
                    "games_played": gp,
                    "ongoal": og,
                    "gsax": gsax,
                    "save_pct": sv,
                }
            )
        if by_team:
            return by_team
    return {}


def _select_top_team_features(df_team_hist: pd.DataFrame, max_features: int = 20) -> list[str]:
    """Identify the strongest scale-invariant team rate features for predicting playoff success."""
    df_norm = normalize_team_frame_per_game(df_team_hist)
    numeric_cols = [
        c
        for c in df_norm.select_dtypes(include=[np.number]).columns
        if c not in TEAM_EXCLUDE_COLS and _is_team_rate_column(c) and not df_norm[c].isna().all()
    ]
    if not numeric_cols:
        numeric_cols = [
            c
            for c in df_norm.select_dtypes(include=[np.number]).columns
            if c not in TEAM_EXCLUDE_COLS and not df_norm[c].isna().all()
        ]
    X = df_norm[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(df_norm["playoff_games_played"], errors="coerce").fillna(0.0)

    baseline_reg = xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
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


BASE_PAIRWISE_FEATURES: list[str] = [
    "Elo_diff",
    "Prior_xGoalsPct_diff",
    "Prior_FenwickPct_diff",
    "Prior_xGD_diff",
    "Prior_GD_diff",
    "Top3_Points_per_game_diff",
    "Team_GSAx_per_game_diff",
    "Starter_SavePct_diff",
    "Blended_WinPct_diff",
    "Blended_GD_GP_diff",
    "EWMA_WinPct_diff",
    "EWMA_GD_diff",
    "Rolling10_WinPct_diff",
    "Rolling10_GoalDiff_diff",
    "Streak_diff",
    "Last_Game_Won_diff",
    "Is_B2B_Home",
    "Is_B2B_Away",
    "Is_B2B_diff",
    "Rest_Days_diff",
    "Is_3in4_diff",
]

# Fatigue experiment. The holdout on the 2026 season regressed versus the 21-feature
# baseline (Brier 0.2288 -> 0.2299, AUC 0.6778 -> 0.6710 with 3 features), so these
# stay OFF by default. Re-run `scripts/eval_fatigue_features.py` on a larger sample
# before flipping ENABLE_FATIGUE_FEATURES; the columns are still computed and stored.
FATIGUE_PAIRWISE_FEATURES: list[str] = [
    "Games_Last7_diff",
    "Travel_Last7_diff",
    "Rest_x_Travel_diff",
]

# Threshold/interaction variants tested in the 2026 walk-forward extension.
FATIGUE_BINARY_FEATURES: list[str] = [
    "Games_Last7_ge4_diff",
    "Games_Last5_ge3_diff",
    "Rest_x_Games7_ge4_diff",
]

ENABLE_FATIGUE_FEATURES: bool = False

# Roster depth candidates (gated; evaluated before joining the schema).
DEPTH_PAIRWISE_FEATURES: list[str] = [
    "Depth_Points_per_game_diff",
    "Depth_xGoals_per_game_diff",
]

CAUSAL_PAIRWISE_FEATURES: list[str] = BASE_PAIRWISE_FEATURES + (
    FATIGUE_PAIRWISE_FEATURES if ENABLE_FATIGUE_FEATURES else []
)


def pairwise_feature_names(*, include_fatigue: bool | None = None) -> list[str]:
    """Return the active pairwise feature schema (fatigue opt-in)."""
    if include_fatigue is None:
        include_fatigue = ENABLE_FATIGUE_FEATURES
    return BASE_PAIRWISE_FEATURES + (FATIGUE_PAIRWISE_FEATURES if include_fatigue else [])


def _lookup_prior_table(
    tbl: pd.DataFrame,
    season: int,
    team: str,
    col: str,
    default: float = 0.0,
) -> float:
    """Lookup (season, team) value with ARI->UTA franchise continuity fallback."""
    if tbl.empty or col not in tbl.columns:
        return default
    if (season, team) in tbl.index:
        v = tbl.loc[(season, team), col]
        return float(v) if pd.notna(v) else default
    prev_tm = FRANCHISE_PRIOR_MAP.get(team)
    if prev_tm and (season, prev_tm) in tbl.index:
        v = tbl.loc[(season, prev_tm), col]
        return float(v) if pd.notna(v) else default
    return default


def _build_causal_training_frame(
    df_team_norm: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble strictly leak-free pre-game pairwise features using prior-season MoneyPuck + causal logs & Elo."""
    logs_enriched = add_causal_game_log_features(df_master_logs)
    pregame_elo, _ = compute_causal_elo_history(logs_enriched)

    df_home = logs_enriched[logs_enriched["Is_Home"] == 1].copy()
    df_home["season_year"] = df_home["Season"].astype(str).str[:4].astype(int)
    df_home["Team"] = df_home["Team"].apply(normalize_team_abbrev)
    df_home["Opponent"] = df_home["Opponent"].apply(normalize_team_abbrev)

    away_cols = [
        "Season",
        "Date",
        "Team",
        "Last_Game_Won",
        "Current_Streak",
        "Is_B2B",
        "Rest_Days",
        "Is_3in4",
        "Games_Last7",
        "Games_Last5",
        "Travel_Last7",
        "Rest_x_Travel",
        "TZ_Shift",
        "RoadTrip_Length",
        "Games_Last7_ge4",
        "Games_Last5_ge3",
        "Rest_x_Games7_ge4",
        "Causal_GP",
        "Causal_WinPct",
        "Causal_GD_GP",
        "Rolling10_WinPct",
        "Rolling10_GoalDiff",
        "EWMA_WinPct",
        "EWMA_GD",
    ]
    df_away = (
        logs_enriched[logs_enriched["Is_Home"] == 0][away_cols]
        .drop_duplicates(subset=["Season", "Date", "Team"])
        .copy()
    )
    df_away["Team"] = df_away["Team"].apply(normalize_team_abbrev)
    df_games = df_home.merge(
        df_away,
        left_on=["Season", "Date", "Opponent"],
        right_on=["Season", "Date", "Team"],
        how="inner",
        suffixes=("_A", "_B"),
    )
    if "Team_A" not in df_games.columns and "Team" in df_games.columns:
        df_games = df_games.rename(columns={"Team": "Team_A"})
    df_games = df_games.sort_values(["Date", "Team_A"]).reset_index(drop=True)

    available_seasons = sorted(int(s) for s in df_team_norm["season"].unique())
    earliest_season = available_seasons[0]
    mp_idx = df_team_norm.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"])
    st_idx = df_stars.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"]) if not df_stars.empty else pd.DataFrame()
    go_idx = df_goalies.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"]) if not df_goalies.empty else pd.DataFrame()

    rows: list[dict[str, float]] = []
    for r in df_games.itertuples(index=False):
        sy = int(r.season_year)
        prior_candidates = [s for s in available_seasons if s < sy]
        # For the earliest season there is no prior MoneyPuck table; use earliest_season - 1
        # so _lookup_prior_table misses and returns its defaults instead of leaking the
        # same season's rates.
        yp = prior_candidates[-1] if prior_candidates else earliest_season - 1
        tA = str(r.Team_A)
        tB = str(r.Opponent)
        eA, eB = pregame_elo.get((str(r.Date), tA), (1500.0, 1500.0))

        p_gf_A = _lookup_prior_table(mp_idx, yp, tA, "goalsFor_per_game", 3.1)
        p_gf_B = _lookup_prior_table(mp_idx, yp, tB, "goalsFor_per_game", 3.1)
        p_ga_A = _lookup_prior_table(mp_idx, yp, tA, "goalsAgainst_per_game", 3.1)
        p_ga_B = _lookup_prior_table(mp_idx, yp, tB, "goalsAgainst_per_game", 3.1)
        p_xgf_A = _lookup_prior_table(mp_idx, yp, tA, "xGoalsFor_per_game", 3.1)
        p_xgf_B = _lookup_prior_table(mp_idx, yp, tB, "xGoalsFor_per_game", 3.1)
        p_xga_A = _lookup_prior_table(mp_idx, yp, tA, "xGoalsAgainst_per_game", 3.1)
        p_xga_B = _lookup_prior_table(mp_idx, yp, tB, "xGoalsAgainst_per_game", 3.1)
        p_xgp_A = _lookup_prior_table(mp_idx, yp, tA, "xGoalsPercentage", 0.5)
        p_xgp_B = _lookup_prior_table(mp_idx, yp, tB, "xGoalsPercentage", 0.5)
        p_fen_A = _lookup_prior_table(mp_idx, yp, tA, "fenwickPercentage", 0.5)
        p_fen_B = _lookup_prior_table(mp_idx, yp, tB, "fenwickPercentage", 0.5)
        p_top3_A = _lookup_prior_table(st_idx, yp, tA, "Top3_Points_per_game", 2.4)
        p_top3_B = _lookup_prior_table(st_idx, yp, tB, "Top3_Points_per_game", 2.4)
        p_gsax_A = _lookup_prior_table(go_idx, yp, tA, "Team_GSAx_per_game", 0.0)
        p_gsax_B = _lookup_prior_table(go_idx, yp, tB, "Team_GSAx_per_game", 0.0)
        p_sv_A = _lookup_prior_table(go_idx, yp, tA, "Starter_SavePct", 0.905)
        p_sv_B = _lookup_prior_table(go_idx, yp, tB, "Starter_SavePct", 0.905)

        gpA = float(getattr(r, "Causal_GP_A", 0.0) or 0.0)
        gpB = float(getattr(r, "Causal_GP_B", 0.0) or 0.0)
        wA = gpA / (gpA + 20.0)
        wB = gpB / (gpB + 20.0)
        prior_win_A = 0.5 + 0.65 * ((p_xgp_A - 0.5) * 1.4)
        prior_win_B = 0.5 + 0.65 * ((p_xgp_B - 0.5) * 1.4)
        c_win_A = float(getattr(r, "Causal_WinPct_A", 0.5) or 0.5)
        c_win_B = float(getattr(r, "Causal_WinPct_B", 0.5) or 0.5)
        c_gd_A = float(getattr(r, "Causal_GD_GP_A", 0.0) or 0.0)
        c_gd_B = float(getattr(r, "Causal_GD_GP_B", 0.0) or 0.0)

        b2b_A = float(getattr(r, "Is_B2B_A", 0.0) or 0.0)
        b2b_B = float(getattr(r, "Is_B2B_B", 0.0) or 0.0)
        rows.append(
            {
                "Elo_diff": eA - eB,
                "Prior_xGoalsPct_diff": p_xgp_A - p_xgp_B,
                "Prior_FenwickPct_diff": p_fen_A - p_fen_B,
                "Prior_xGD_diff": (p_xgf_A - p_xga_A) - (p_xgf_B - p_xga_B),
                "Prior_GD_diff": (p_gf_A - p_ga_A) - (p_gf_B - p_ga_B),
                "Top3_Points_per_game_diff": p_top3_A - p_top3_B,
                "Depth_Points_per_game_diff": float(getattr(r, "Depth_Points_per_game_A", 0.0) or 0.0) - float(getattr(r, "Depth_Points_per_game_B", 0.0) or 0.0),
                "Depth_xGoals_per_game_diff": float(getattr(r, "Depth_xGoals_per_game_A", 0.0) or 0.0) - float(getattr(r, "Depth_xGoals_per_game_B", 0.0) or 0.0),
                "Team_GSAx_per_game_diff": p_gsax_A - p_gsax_B,
                "Starter_SavePct_diff": p_sv_A - p_sv_B,
                "Blended_WinPct_diff": ((1.0 - wA) * prior_win_A + wA * c_win_A) - ((1.0 - wB) * prior_win_B + wB * c_win_B),
                "Blended_GD_GP_diff": ((1.0 - wA) * 0.65 * (p_gf_A - p_ga_A) + wA * c_gd_A) - ((1.0 - wB) * 0.65 * (p_gf_B - p_ga_B) + wB * c_gd_B),
                "EWMA_WinPct_diff": float(getattr(r, "EWMA_WinPct_A", 0.5) or 0.5) - float(getattr(r, "EWMA_WinPct_B", 0.5) or 0.5),
                "EWMA_GD_diff": float(getattr(r, "EWMA_GD_A", 0.0) or 0.0) - float(getattr(r, "EWMA_GD_B", 0.0) or 0.0),
                "Rolling10_WinPct_diff": float(getattr(r, "Rolling10_WinPct_A", 0.5) or 0.5) - float(getattr(r, "Rolling10_WinPct_B", 0.5) or 0.5),
                "Rolling10_GoalDiff_diff": float(getattr(r, "Rolling10_GoalDiff_A", 0.0) or 0.0) - float(getattr(r, "Rolling10_GoalDiff_B", 0.0) or 0.0),
                "Streak_diff": float(getattr(r, "Current_Streak_A", 0.0) or 0.0) - float(getattr(r, "Current_Streak_B", 0.0) or 0.0),
                "Last_Game_Won_diff": float(getattr(r, "Last_Game_Won_A", 0.0) or 0.0) - float(getattr(r, "Last_Game_Won_B", 0.0) or 0.0),
                "Is_B2B_Home": b2b_A,
                "Is_B2B_Away": b2b_B,
                "Is_B2B_diff": b2b_A - b2b_B,
                "Rest_Days_diff": float(getattr(r, "Rest_Days_A", 2.0) or 0.0) - float(getattr(r, "Rest_Days_B", 2.0) or 0.0),
                "Is_3in4_diff": float(getattr(r, "Is_3in4_A", 0.0) or 0.0) - float(getattr(r, "Is_3in4_B", 0.0) or 0.0),
                "Games_Last7_diff": float(getattr(r, "Games_Last7_A", 0.0) or 0.0) - float(getattr(r, "Games_Last7_B", 0.0) or 0.0),
                "Travel_Last7_diff": (float(getattr(r, "Travel_Last7_A", 0.0) or 0.0) / 1000.0) - (float(getattr(r, "Travel_Last7_B", 0.0) or 0.0) / 1000.0),
                "Rest_x_Travel_diff": float(getattr(r, "Rest_x_Travel_A", 0.0) or 0.0) - float(getattr(r, "Rest_x_Travel_B", 0.0) or 0.0),
                "Games_Last7_ge4_diff": float(getattr(r, "Games_Last7_ge4_A", 0.0) or 0.0) - float(getattr(r, "Games_Last7_ge4_B", 0.0) or 0.0),
                "Games_Last5_ge3_diff": float(getattr(r, "Games_Last5_ge3_A", 0.0) or 0.0) - float(getattr(r, "Games_Last5_ge3_B", 0.0) or 0.0),
                "Rest_x_Games7_ge4_diff": float(getattr(r, "Rest_x_Games7_ge4_A", 0.0) or 0.0) - float(getattr(r, "Rest_x_Games7_ge4_B", 0.0) or 0.0),
                "TZ_Shift_diff": float(getattr(r, "TZ_Shift_A", 0.0) or 0.0) - float(getattr(r, "TZ_Shift_B", 0.0) or 0.0),
                "RoadTrip_Length_diff": float(getattr(r, "RoadTrip_Length_A", 0.0) or 0.0) - float(getattr(r, "RoadTrip_Length_B", 0.0) or 0.0),
            }
        )

    X_pair = pd.DataFrame(rows)[list(feature_names or CAUSAL_PAIRWISE_FEATURES)].fillna(0.0)
    y_pair = df_games["Won"].astype(int).to_numpy()
    seasons = df_games["season_year"].astype(int).to_numpy()
    return df_games, X_pair, y_pair, seasons


def train_pairwise_game_model(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
    *,
    lr_weight: float = 0.70,
    feature_subset: list[str] | None = None,
    playoff_weight: float = 1.0,
) -> dict[str, Any]:
    """Train leak-free causal Pairwise XGBoost + L2-LogReg ensemble on Elo, prior MoneyPuck, form & B2B rest."""
    if not HAS_GAME_ML:
        raise RuntimeError("Missing ML dependencies (xgboost, scikit-learn, joblib).")
    if df_team_hist.empty or df_master_logs.empty:
        raise RuntimeError("Cannot train pairwise game model: missing team history or NHL game logs.")

    df_team_norm = normalize_team_frame_per_game(df_team_hist)
    base_features = _select_top_team_features(df_team_norm, max_features=20)
    df_games_pair, X_pair, y_pair, seasons = _build_causal_training_frame(
        df_team_norm, df_stars, df_goalies, df_master_logs
    )

    if feature_subset is not None:
        subset_set = set(feature_subset)
        feature_names = [f for f in CAUSAL_PAIRWISE_FEATURES if f in subset_set]
        if not feature_names:
            raise ValueError("feature_subset matched none of CAUSAL_PAIRWISE_FEATURES.")
        X_pair = X_pair[feature_names]
    else:
        feature_names = list(CAUSAL_PAIRWISE_FEATURES)

    if playoff_weight != 1.0 and "GameType" in df_games_pair.columns:
        weights = np.where(df_games_pair["GameType"].to_numpy() == 3, float(playoff_weight), 1.0)
    else:
        weights = np.ones(len(y_pair), dtype=float)
    mcw = min(25, max(1, len(y_pair) // 50))
    xgb_kwargs = {
        "n_estimators": 140,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": mcw,
        "reg_lambda": 8.0,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    def _fit_lr(X_mat: pd.DataFrame, y_vec: np.ndarray):
        pipe = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(C=0.05, max_iter=1000, random_state=42),
        )
        pipe.fit(X_mat, y_vec)
        return pipe

    optimal_T = 1.0
    oof_ll_raw = 0.693
    oof_ll_cal = 0.693
    oof_acc = 0.5
    oof_brier = 0.25
    oof_auc = 0.5

    if len(y_pair) >= 50 and len(np.unique(y_pair)) == 2:
        unique_seasons = np.unique(seasons)
        if GroupKFold is not None and len(unique_seasons) >= 3:
            cv_splits = list(GroupKFold(n_splits=min(len(unique_seasons), 5)).split(X_pair, y_pair, groups=seasons))
        else:
            cv_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_pair, y_pair))

        oof_probs = np.zeros(len(y_pair))
        for tr_idx, va_idx in cv_splits:
            fold_xgb = xgb.XGBClassifier(**xgb_kwargs)
            fold_xgb.fit(X_pair.iloc[tr_idx], y_pair[tr_idx], sample_weight=weights[tr_idx])
            p_x = fold_xgb.predict_proba(X_pair.iloc[va_idx])[:, 1]
            if len(np.unique(y_pair[tr_idx])) == 2:
                fold_lr = _fit_lr(X_pair.iloc[tr_idx], y_pair[tr_idx])
                p_l = fold_lr.predict_proba(X_pair.iloc[va_idx])[:, 1]
                oof_probs[va_idx] = (1.0 - lr_weight) * p_x + lr_weight * p_l
            else:
                oof_probs[va_idx] = p_x

        oof_ll_raw = float(log_loss(y_pair, oof_probs))
        res = scipy.optimize.minimize(
            lambda t: log_loss(y_pair, apply_temperature(oof_probs, T=float(t[0]))),
            x0=[1.05],
            bounds=[(1.0, 15.0)],
        )
        optimal_T = float(res.x[0]) if res.success else 1.0
        calibrated_oof = apply_temperature(oof_probs, T=optimal_T)
        oof_ll_cal = float(log_loss(y_pair, calibrated_oof))
        oof_acc = float(accuracy_score(y_pair, (calibrated_oof >= 0.5).astype(int)))
        oof_brier = float(brier_score_loss(y_pair, calibrated_oof))
        oof_auc = float(roc_auc_score(y_pair, calibrated_oof))

    clf = xgb.XGBClassifier(**xgb_kwargs)
    clf.fit(X_pair, y_pair, sample_weight=weights)
    lr_clf = _fit_lr(X_pair, y_pair) if len(np.unique(y_pair)) == 2 else None

    xgb_imp = np.asarray(clf.feature_importances_, dtype=float)
    xgb_imp = xgb_imp / max(1e-9, float(np.sum(xgb_imp)))
    if lr_clf is not None:
        lr_coef = np.abs(lr_clf.named_steps["logisticregression"].coef_[0])
        lr_imp = lr_coef / max(1e-9, float(np.sum(lr_coef)))
        comb_imp = (1.0 - lr_weight) * xgb_imp + lr_weight * lr_imp
    else:
        comb_imp = xgb_imp

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": comb_imp})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    top_feat_list = [
        {"feature": str(r["feature"]), "importance": round(float(r["importance"]), 4)}
        for _, r in imp_df.iterrows()
    ]

    return {
        "model": clf,
        "lr_model": lr_clf,
        "lr_weight": lr_weight if lr_clf is not None else 0.0,
        "base_features": base_features,
        "feature_names": feature_names,
        "optimal_T": round(optimal_T, 3),
        "metrics": {
            "games_trained": int(len(y_pair)),
            "oof_log_loss_raw": round(oof_ll_raw, 4),
            "oof_log_loss_calibrated": round(oof_ll_cal, 4),
            "oof_accuracy": round(oof_acc, 4),
            "oof_brier": round(oof_brier, 4),
            "oof_auc": round(oof_auc, 4),
            "optimal_T": round(optimal_T, 3),
        },
        "feature_importance": top_feat_list,
    }


def save_pairwise_bundle(bundle: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_pairwise_bundle(path: str) -> dict[str, Any]:
    if not HAS_GAME_ML:
        raise RuntimeError("Missing ML dependencies (xgboost, scikit-learn, joblib).")
    return joblib.load(path)



def _apply_roster_star_to_stat(stat: dict[str, Any], roster_star: dict[str, Any] | None) -> None:
    """Override a live profile's Top-3 rates with roster-aware per-game values."""
    if not isinstance(stat, dict) or not isinstance(roster_star, dict):
        return
    pg = float(roster_star.get("Top3_Points_per_game", stat.get("Top3_Points_per_game", 0.0)) or 0.0)
    xg = float(roster_star.get("Top3_xGoals_per_game", stat.get("Top3_xGoals_per_game", 0.0)) or 0.0)
    gp = max(1, int(stat.get("games_played", 0) or 1))
    stat["Top3_Points_per_game"] = pg
    stat["Top3_xGoals_per_game"] = xg
    stat["Top3_Points"] = pg * gp
    stat["Top3_xGoals"] = xg * gp
    for key in ("Depth_Points_per_game", "Depth_xGoals_per_game"):
        if key in roster_star:
            stat[key] = float(roster_star.get(key, stat.get(key, 0.0)) or 0.0)


def apply_roster_stars_to_profiles(
    profiles: dict[str, dict[str, Any]],
    roster_stars: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Apply roster-aware Top-3 overrides in place; return per-team Top-3 deltas."""
    deltas: dict[str, float] = {}
    if not roster_stars or not isinstance(profiles, dict):
        return deltas
    for team, rs in roster_stars.items():
        stat = profiles.get(team)
        if not stat:
            continue
        before = float(stat.get("Top3_Points_per_game", 0.0) or 0.0)
        _apply_roster_star_to_stat(stat, rs)
        deltas[str(team)] = float(stat.get("Top3_Points_per_game", 0.0) or 0.0) - before
    return deltas



def build_live_team_profiles(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    active_mp_year: int,
    latest_team_context: dict[str, dict[str, Any]],
    base_features: list[str],
    playoff_teams_df: pd.DataFrame | None = None,
    roster_stars: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Assemble current 32-team shrunk causal profiles for inference and defense multipliers."""
    df_norm = normalize_team_frame_per_game(df_team_hist)
    available_seasons = sorted(int(s) for s in df_norm["season"].unique()) if not df_norm.empty else [active_mp_year]
    available_year = active_mp_year if active_mp_year in available_seasons else available_seasons[-1]
    prior_candidates = [s for s in available_seasons if s < available_year]
    prior_year = prior_candidates[-1] if prior_candidates else available_year

    df_curr = df_norm[df_norm["season"] == available_year].drop_duplicates(subset=["team"]).copy()
    mp_idx = df_norm.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"])

    # Optional playoff stats blending if active playoff CSV exists
    play_clean = pd.DataFrame()
    if playoff_teams_df is not None and not playoff_teams_df.empty:
        play_raw = (
            playoff_teams_df[playoff_teams_df["situation"] == "all"].copy()
            if "situation" in playoff_teams_df.columns
            else playoff_teams_df.copy()
        )
        play_raw["team"] = play_raw["team"].apply(normalize_team_abbrev)
        play_clean = normalize_team_frame_per_game(play_raw)

    profiles: dict[str, dict[str, Any]] = {}
    stars_curr = df_stars[df_stars["season"] == available_year].set_index("team") if not df_stars.empty else pd.DataFrame()
    goalies_curr = df_goalies[df_goalies["season"] == available_year].set_index("team") if not df_goalies.empty else pd.DataFrame()
    st_idx = df_stars.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"]) if not df_stars.empty else pd.DataFrame()
    go_idx = df_goalies.drop_duplicates(subset=["season", "team"]).set_index(["season", "team"]) if not df_goalies.empty else pd.DataFrame()

    for _, row in df_curr.iterrows():
        team = str(row["team"])
        stat: dict[str, Any] = {f: float(row.get(f, 0.0) or 0.0) for f in base_features}
        gp_reg = max(1.0, float(row.get("games_played", 1.0) or 1.0))

        if not play_clean.empty and team in play_clean["team"].values:
            p_row = play_clean[play_clean["team"] == team].iloc[0]
            gp_play = float(p_row.get("games_played", 0.0) or 0.0)
            tot_gp = max(1.0, gp_reg + gp_play)
            for feat in base_features:
                p_val = float(p_row.get(feat, stat[feat]) or 0.0)
                if _is_team_rate_column(feat):
                    stat[feat] = (stat[feat] * gp_reg + p_val * gp_play) / tot_gp
                else:
                    stat[feat] = stat[feat] + p_val

        # Prior MoneyPuck baseline (with ARI->UTA continuity) blended smoothly with live MoneyPuck GP
        w_mp = gp_reg / (gp_reg + 20.0) if prior_year != available_year else 1.0
        p_xgp = _lookup_prior_table(mp_idx, prior_year, team, "xGoalsPercentage", 0.5)
        p_fen = _lookup_prior_table(mp_idx, prior_year, team, "fenwickPercentage", 0.5)
        p_xgf = _lookup_prior_table(mp_idx, prior_year, team, "xGoalsFor_per_game", 3.1)
        p_xga = _lookup_prior_table(mp_idx, prior_year, team, "xGoalsAgainst_per_game", 3.1)
        p_gf = _lookup_prior_table(mp_idx, prior_year, team, "goalsFor_per_game", 3.1)
        p_ga = _lookup_prior_table(mp_idx, prior_year, team, "goalsAgainst_per_game", 3.1)

        stat["Prior_GF"] = p_gf
        stat["Prior_GA"] = p_ga
        stat["Prior_xGF"] = p_xgf
        stat["Prior_xGA"] = p_xga
        stat["Prior_ShotsFor"] = _lookup_prior_table(mp_idx, prior_year, team, "shotsOnGoalFor_per_game", 30.0)
        stat["Prior_ShotsAgainst"] = _lookup_prior_table(mp_idx, prior_year, team, "shotsOnGoalAgainst_per_game", 30.0)
        stat["Prior_Attempts"] = _lookup_prior_table(mp_idx, prior_year, team, "shotAttemptsFor_per_game", 60.0)
        stat["Prior_DangerFor"] = _lookup_prior_table(mp_idx, prior_year, team, "highDangerShotsFor_per_game", 8.0)
        stat["Prior_DangerAgainst"] = _lookup_prior_table(mp_idx, prior_year, team, "highDangerShotsAgainst_per_game", 8.0)
        stat["Prior_Starter_SavePct"] = _lookup_prior_table(go_idx, prior_year, team, "Starter_SavePct", 0.905)
        stat["Prior_Team_GSAx"] = _lookup_prior_table(go_idx, prior_year, team, "Team_GSAx_per_game", 0.0)

        c_xgp = float(row.get("xGoalsPercentage", p_xgp) or p_xgp)
        c_fen = float(row.get("fenwickPercentage", p_fen) or p_fen)
        c_xgd = float(row.get("xGoalsFor_per_game", p_xgf) or p_xgf) - float(row.get("xGoalsAgainst_per_game", p_xga) or p_xga)
        c_gd = float(row.get("goalsFor_per_game", p_gf) or p_gf) - float(row.get("goalsAgainst_per_game", p_ga) or p_ga)

        stat["Prior_xGoalsPct"] = (1.0 - w_mp) * p_xgp + w_mp * c_xgp
        stat["Prior_FenwickPct"] = (1.0 - w_mp) * p_fen + w_mp * c_fen
        stat["Prior_xGD"] = (1.0 - w_mp) * (p_xgf - p_xga) + w_mp * c_xgd
        stat["Prior_GD"] = (1.0 - w_mp) * (p_gf - p_ga) + w_mp * c_gd

        if not stars_curr.empty and team in stars_curr.index:
            t3_pts = float(stars_curr.loc[team, "Top3_Points"])
            t3_xg = float(stars_curr.loc[team, "Top3_xGoals"])
            c_t3_pts_pg = float(
                stars_curr.loc[team, "Top3_Points_per_game"]
                if "Top3_Points_per_game" in stars_curr.columns
                else (t3_pts / gp_reg)
            )
            c_t3_xg_pg = float(
                stars_curr.loc[team, "Top3_xGoals_per_game"]
                if "Top3_xGoals_per_game" in stars_curr.columns
                else (t3_xg / gp_reg)
            )
            p_t3_pts_pg = _lookup_prior_table(st_idx, prior_year, team, "Top3_Points_per_game", c_t3_pts_pg)
            p_t3_xg_pg = _lookup_prior_table(st_idx, prior_year, team, "Top3_xGoals_per_game", c_t3_xg_pg)
            stat["Top3_Points"] = t3_pts
            stat["Top3_xGoals"] = t3_xg
            stat["Top3_Points_per_game"] = (1.0 - w_mp) * p_t3_pts_pg + w_mp * c_t3_pts_pg
            stat["Top3_xGoals_per_game"] = (1.0 - w_mp) * p_t3_xg_pg + w_mp * c_t3_xg_pg
        else:
            stat["Top3_Points"] = 0.0
            stat["Top3_xGoals"] = 0.0
            stat["Top3_Points_per_game"] = _lookup_prior_table(st_idx, prior_year, team, "Top3_Points_per_game", 0.0)
            stat["Top3_xGoals_per_game"] = _lookup_prior_table(st_idx, prior_year, team, "Top3_xGoals_per_game", 0.0)

        if not stars_curr.empty and "Depth_Points_per_game" in getattr(stars_curr, "columns", []):
            c_depth = float(stars_curr.loc[team, "Depth_Points_per_game"])
            p_depth = _lookup_prior_table(st_idx, prior_year, team, "Depth_Points_per_game", c_depth)
            stat["Depth_Points_per_game"] = (1.0 - w_mp) * p_depth + w_mp * c_depth
            if "Depth_xGoals_per_game" in getattr(stars_curr, "columns", []):
                c_dxg = float(stars_curr.loc[team, "Depth_xGoals_per_game"])
                p_dxg = _lookup_prior_table(st_idx, prior_year, team, "Depth_xGoals_per_game", c_dxg)
                stat["Depth_xGoals_per_game"] = (1.0 - w_mp) * p_dxg + w_mp * c_dxg

        if not goalies_curr.empty and team in goalies_curr.index:
            gsax = float(goalies_curr.loc[team, "Team_GSAx"])
            c_gsax_pg = float(
                goalies_curr.loc[team, "Team_GSAx_per_game"]
                if "Team_GSAx_per_game" in goalies_curr.columns
                else (gsax / gp_reg)
            )
            c_sv = float(goalies_curr.loc[team, "Starter_SavePct"])
            p_gsax_pg = _lookup_prior_table(go_idx, prior_year, team, "Team_GSAx_per_game", c_gsax_pg)
            p_sv = _lookup_prior_table(go_idx, prior_year, team, "Starter_SavePct", c_sv)
            stat["Team_GSAx"] = gsax
            stat["Team_GSAx_per_game"] = (1.0 - w_mp) * p_gsax_pg + w_mp * c_gsax_pg
            stat["Starter_SavePct"] = (1.0 - w_mp) * p_sv + w_mp * c_sv
        else:
            stat["Team_GSAx"] = 0.0
            stat["Team_GSAx_per_game"] = _lookup_prior_table(go_idx, prior_year, team, "Team_GSAx_per_game", 0.0)
            stat["Starter_SavePct"] = _lookup_prior_table(go_idx, prior_year, team, "Starter_SavePct", 0.900)

        ctx = latest_team_context.get(team, {})
        gp_logs = float(ctx.get("Causal_GP", gp_reg) or gp_reg)
        w_logs = gp_logs / (gp_logs + 20.0)
        prior_win = 0.5 + 0.65 * ((p_xgp - 0.5) * 1.4)
        c_win = float(ctx.get("Causal_WinPct", 0.5))
        c_gd_pg = float(ctx.get("Causal_GD_GP", 0.0))

        stat["Elo"] = float(ctx.get("Causal_Elo", 1500.0))
        stat["Causal_Elo"] = stat["Elo"]
        stat["Blended_WinPct"] = (1.0 - w_logs) * prior_win + w_logs * c_win
        stat["Blended_GD_GP"] = (1.0 - w_logs) * 0.65 * (p_gf - p_ga) + w_logs * c_gd_pg
        stat["EWMA_WinPct"] = float(ctx.get("EWMA_WinPct", 0.5))
        stat["EWMA_GD"] = float(ctx.get("EWMA_GD", 0.0))
        stat["Last_Game_Won"] = int(ctx.get("Last_Game_Won", 0))
        stat["Streak"] = int(ctx.get("Current_Streak", 0))
        stat["Last_Game_Date"] = str(ctx.get("Last_Game_Date", "") or "")
        stat["Recent_Game_Dates"] = list(ctx.get("Recent_Game_Dates") or ([stat["Last_Game_Date"]] if stat["Last_Game_Date"] else []))
        stat["Recent_Game_Venues"] = list(ctx.get("Recent_Game_Venues") or [])
        stat["Recent_Is_Home"] = list(ctx.get("Recent_Is_Home") or [])
        stat["Rolling10_WinPct"] = float(ctx.get("Rolling10_WinPct", 0.5))
        stat["Rolling10_GoalDiff"] = float(ctx.get("Rolling10_GoalDiff", 0.0))
        stat["games_played"] = int(row.get("games_played", 0) or 0)
        if roster_stars and team in roster_stars:
            _apply_roster_star_to_stat(stat, roster_stars.get(team))
        stat["goalsAgainst"] = float(row.get("goalsAgainst", 0.0) or 0.0)
        stat["xGoalsAgainst"] = float(row.get("xGoalsAgainst", 0.0) or 0.0)
        stat["highDangerxGoalsAgainst"] = float(row.get("highDangerxGoalsAgainst", 0.0) or 0.0)
        profiles[team] = stat

    return profiles, df_curr


def _schedule_fatigue_for_date(
    stat: dict[str, Any],
    game_date: str | None,
    extra_dates: list[str] | set[str] | None = None,
) -> tuple[float, float, float]:
    """Compute (Is_B2B, Rest_Days, Is_3in4) entering game_date including scheduled prior-day games."""
    if not game_date:
        return 0.0, 2.0, 0.0
    g_dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(g_dt):
        return 0.0, 2.0, 0.0

    raw_dates: list[str] = []
    if stat.get("Last_Game_Date"):
        raw_dates.append(str(stat["Last_Game_Date"]))
    for d in stat.get("Recent_Game_Dates") or []:
        if d:
            raw_dates.append(str(d))
    for d in extra_dates or []:
        if d:
            raw_dates.append(str(d))

    prior_dts = []
    for raw in raw_dates:
        dt = pd.to_datetime(str(raw).strip(), errors="coerce")
        if pd.notna(dt) and dt < g_dt:
            prior_dts.append(dt)
    if not prior_dts:
        return 0.0, 2.0, 0.0

    unique_prior = sorted(set(prior_dts))
    diff_days = int((g_dt - unique_prior[-1]).days)
    rest_days = float(max(0, min(7, diff_days - 1)))
    is_b2b = 1.0 if rest_days == 0.0 else 0.0
    is_3in4 = 1.0 if sum(1 for d in unique_prior if 1 <= int((g_dt - d).days) <= 3) >= 2 else 0.0
    return is_b2b, rest_days, is_3in4


def _rest_metrics_for_date(
    stat: dict[str, Any],
    game_date: str | None,
    extra_dates: list[str] | set[str] | None = None,
) -> tuple[float, float]:
    """Compute (Is_B2B, Rest_Days) entering game_date; neutral (0.0, 2.0) when game_date omitted."""
    is_b2b, rest_days, _ = _schedule_fatigue_for_date(stat, game_date, extra_dates=extra_dates)
    return is_b2b, rest_days


def _fatigue_metrics_for_date(
    stat: dict[str, Any],
    game_date: str | None,
    extra_dates: list[str] | set[str] | None = None,
    venue_team: str | None = None,
) -> dict[str, float]:
    """Return the 5 raw fatigue metrics for an upcoming game from a live team profile."""
    out = {
        "Games_Last7": 0.0,
        "Games_Last5": 0.0,
        "Travel_Last7": 0.0,
        "TZ_Shift": 0.0,
        "RoadTrip_Length": 0.0,
        "Rest_x_Travel": 0.0,
        "Games_Last7_ge4": 0.0,
        "Games_Last5_ge3": 0.0,
        "Rest_x_Games7_ge4": 0.0,
    }
    if not game_date:
        return out
    g_dt = pd.to_datetime(game_date, errors="coerce")
    if pd.isna(g_dt):
        return out

    dates_list = list(stat.get("Recent_Game_Dates") or [])
    venues_list = list(stat.get("Recent_Game_Venues") or [])
    is_home_list = list(stat.get("Recent_Is_Home") or [])

    # Build completed (date, venue, is_home) list, aligned.
    completed: list[tuple[pd.Timestamp, str | None, Any]] = []
    if venues_list:
        n = min(len(dates_list), len(venues_list))
        dd = dates_list[-n:] if n > 0 else []
        vv = venues_list[-n:] if n > 0 else []
        hh = is_home_list[-n:] if len(is_home_list) >= n else [None] * n
        for d, v, h in zip(dd, vv, hh):
            dt = pd.to_datetime(str(d).strip(), errors="coerce") if d is not None else pd.NaT
            if pd.notna(dt):
                completed.append((dt, (str(v).strip().upper() if v else None), h))
    else:
        for d in dates_list:
            dt = pd.to_datetime(str(d).strip(), errors="coerce") if d is not None else pd.NaT
            if pd.notna(dt):
                completed.append((dt, None, None))

    if not completed and stat.get("Last_Game_Date"):
        dt = pd.to_datetime(str(stat["Last_Game_Date"]).strip(), errors="coerce")
        if pd.notna(dt):
            completed.append((dt, None, None))

    completed.sort(key=lambda x: x[0])
    prior_completed = [x for x in completed if x[0] < g_dt]

    extra_prior_dts: list[pd.Timestamp] = []
    for raw in extra_dates or []:
        dt = pd.to_datetime(str(raw).strip(), errors="coerce") if raw is not None else pd.NaT
        if pd.notna(dt) and dt < g_dt:
            extra_prior_dts.append(dt)

    all_prior_dts = sorted(set([x[0] for x in prior_completed] + extra_prior_dts))
    games_last7 = sum(1 for d in all_prior_dts if 1 <= int((g_dt - d).days) <= 7)
    out["Games_Last7"] = float(games_last7)
    games_last5 = sum(1 for d in all_prior_dts if 1 <= int((g_dt - d).days) <= 5)
    out["Games_Last5"] = float(games_last5)
    out["Games_Last7_ge4"] = 1.0 if games_last7 >= 4 else 0.0
    out["Games_Last5_ge3"] = 1.0 if games_last5 >= 3 else 0.0

    # Travel among prior completed games within 7 days.
    recent_completed = [x for x in prior_completed if 1 <= int((g_dt - x[0]).days) <= 7]
    travel = 0.0
    for k in range(1, len(recent_completed)):
        v_prev = recent_completed[k - 1][1]
        v_curr = recent_completed[k][1]
        if v_prev and v_curr:
            travel += travel_miles(v_prev, v_curr)
    last_prior_venue = prior_completed[-1][1] if prior_completed else None
    if last_prior_venue and venue_team:
        travel += travel_miles(last_prior_venue, venue_team)
    out["Travel_Last7"] = float(travel)

    if last_prior_venue and venue_team:
        out["TZ_Shift"] = float(abs(tz_shift_hours(last_prior_venue, venue_team, game_date)))

    # Road trip: trailing consecutive away games ending at last prior completed game.
    road = 0
    for dt, v, h in reversed(prior_completed):
        is_home = False
        if h is None:
            break
        try:
            is_home = int(h or 0) == 1 or str(h).strip().lower() in ("true", "1", "1.0")
        except Exception:
            is_home = False
        if is_home:
            break
        road += 1
        if road >= 7:
            break
    out["RoadTrip_Length"] = float(min(road, 7))

    # Rest days (schedule-aware, including extra dates).
    if all_prior_dts:
        diff_days = int((g_dt - all_prior_dts[-1]).days)
        rest_days = float(max(0, min(7, diff_days - 1)))
    else:
        rest_days = 2.0
    out["Rest_x_Travel"] = float(rest_days * (travel / 1000.0))
    out["Rest_x_Games7_ge4"] = float(rest_days * out["Games_Last7_ge4"])

    return out



def _shrink_prob_for_churn(home_prob: float, churn_home: float, churn_away: float) -> float:
    """Shrink a home win probability toward 0.5 for major roster churn (max 25%)."""
    churn = max(float(churn_home or 0.0), float(churn_away or 0.0))
    churn = min(1.0, max(0.0, churn))
    if churn <= 0.0:
        return float(home_prob)
    factor = 1.0 - 0.25 * churn
    return 0.5 + (float(home_prob) - 0.5) * factor


def _roster_move_names(roster_moves: dict[str, Any] | None, team: str) -> tuple[list[str], list[str]]:
    """Return (players_in, players_out) for a team from a diff_rosters payload."""
    info = ((roster_moves or {}).get("teams") or {}).get(team) or {}
    ins = [str(x) for x in (info.get("in") or []) if x][:5]
    outs = [str(x) for x in (info.get("out") or []) if x][:5]
    return ins, outs

def _pairwise_win_prob(
    team_a: str,
    team_b: str,
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
    game_date: str | None = None,
    extra_dates_a: list[str] | set[str] | None = None,
    extra_dates_b: list[str] | set[str] | None = None,
    venue_team: str | None = None,
) -> float:
    if team_a not in profiles or team_b not in profiles:
        return 0.5
    stat_a, stat_b = profiles[team_a], profiles[team_b]
    b2b_a, rest_a, f3in4_a = _schedule_fatigue_for_date(stat_a, game_date, extra_dates=extra_dates_a)
    b2b_b, rest_b, f3in4_b = _schedule_fatigue_for_date(stat_b, game_date, extra_dates=extra_dates_b)

    if game_date and venue_team:
        metrics_a = _fatigue_metrics_for_date(stat_a, game_date, extra_dates=extra_dates_a, venue_team=venue_team)
        metrics_b = _fatigue_metrics_for_date(stat_b, game_date, extra_dates=extra_dates_b, venue_team=venue_team)
    else:
        metrics_a = None
        metrics_b = None

    feature_names: list[str] = bundle["feature_names"]
    row_dict: dict[str, list[float]] = {}
    for feat in feature_names:
        if feat == "Is_B2B_Home":
            row_dict[feat] = [b2b_a]
        elif feat == "Is_B2B_Away":
            row_dict[feat] = [b2b_b]
        elif feat == "Is_B2B_diff":
            row_dict[feat] = [b2b_a - b2b_b]
        elif feat == "Rest_Days_diff":
            row_dict[feat] = [rest_a - rest_b]
        elif feat == "Is_3in4_diff":
            row_dict[feat] = [f3in4_a - f3in4_b]
        elif feat == "Games_Last7_diff":
            va = metrics_a["Games_Last7"] if metrics_a else 0.0
            vb = metrics_b["Games_Last7"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "Travel_Last7_diff":
            va = (metrics_a["Travel_Last7"] / 1000.0) if metrics_a else 0.0
            vb = (metrics_b["Travel_Last7"] / 1000.0) if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "TZ_Shift_diff":
            va = metrics_a["TZ_Shift"] if metrics_a else 0.0
            vb = metrics_b["TZ_Shift"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "RoadTrip_Length_diff":
            va = metrics_a["RoadTrip_Length"] if metrics_a else 0.0
            vb = metrics_b["RoadTrip_Length"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "Rest_x_Travel_diff":
            va = metrics_a["Rest_x_Travel"] if metrics_a else 0.0
            vb = metrics_b["Rest_x_Travel"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "Games_Last7_ge4_diff":
            va = metrics_a["Games_Last7_ge4"] if metrics_a else 0.0
            vb = metrics_b["Games_Last7_ge4"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "Games_Last5_ge3_diff":
            va = metrics_a["Games_Last5_ge3"] if metrics_a else 0.0
            vb = metrics_b["Games_Last5_ge3"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        elif feat == "Rest_x_Games7_ge4_diff":
            va = metrics_a["Rest_x_Games7_ge4"] if metrics_a else 0.0
            vb = metrics_b["Rest_x_Games7_ge4"] if metrics_b else 0.0
            row_dict[feat] = [va - vb]
        else:
            base_key = feat[:-5] if feat.endswith("_diff") else feat
            row_dict[feat] = [float(stat_a.get(base_key, 0.0)) - float(stat_b.get(base_key, 0.0))]

    X_m = pd.DataFrame(row_dict)[feature_names].fillna(0.0)
    raw_prob = float(bundle["model"].predict_proba(X_m)[0][1])
    lr_model = bundle.get("lr_model")
    lr_weight = float(bundle.get("lr_weight", 0.0) or 0.0)
    if lr_model is not None and lr_weight > 0.0:
        lr_prob = float(lr_model.predict_proba(X_m)[0][1])
        raw_prob = (1.0 - lr_weight) * raw_prob + lr_weight * lr_prob
    return float(apply_temperature(raw_prob, T=float(bundle["optimal_T"])))


def compute_power_rankings_and_defense(
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compute home-neutral AI Power Ratings across all teams and 70/30 Defense Multiplier."""
    teams = sorted(profiles.keys())
    if not teams:
        return [], {}

    avg_win_probs: dict[str, float] = {}
    ai_power_dict: dict[str, float] = {}
    denom = max(1, len(teams) - 1)

    for t1 in teams:
        total_p = 0.0
        for t2 in teams:
            if t2 == t1:
                continue
            p_home = _pairwise_win_prob(t1, t2, profiles, bundle)
            p_away = 1.0 - _pairwise_win_prob(t2, t1, profiles, bundle)
            total_p += 0.5 * (p_home + p_away)
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
                "elo": round(float(st.get("Causal_Elo", 1500.0)), 1),
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


def _safe_int(val: Any) -> int | None:
    try:
        if val is None or val == "":
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


def extract_cached_odds_lookup(
    payload: dict[str, Any] | None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Extract cached per-game moneyline odds keyed by (date, away, home) from data.json."""
    cached: dict[tuple[str, str, str], dict[str, Any]] = {}
    _meta = (payload or {}).get("meta") or {}
    _global_as_of = _meta.get("line_markets_as_of")
    if not _global_as_of:
        _global_as_of = (_meta.get("odds_usage") or {}).get("line_markets_as_of")
    for g in (payload or {}).get("games") or []:
        date_str = str(g.get("date") or "").strip()
        away_abbr = normalize_team_abbrev(g.get("away"))
        home_abbr = normalize_team_abbrev(g.get("home"))
        if not date_str or not away_abbr or not home_abbr:
            continue

        raw_h = g.get("home_odds")
        if raw_h is None and isinstance(g.get("home_edge"), dict):
            raw_h = g["home_edge"].get("odds")
        raw_a = g.get("away_odds")
        if raw_a is None and isinstance(g.get("away_edge"), dict):
            raw_a = g["away_edge"].get("odds")

        try:
            home_odds = float(raw_h) if raw_h is not None else 0.0
        except (TypeError, ValueError):
            home_odds = 0.0
        try:
            away_odds = float(raw_a) if raw_a is not None else 0.0
        except (TypeError, ValueError):
            away_odds = 0.0

        if home_odds <= 1.0 and away_odds <= 1.0:
            continue

        entry: dict[str, Any] = {"bookie": g.get("bookie") or "Sharp"}
        if g.get("bookie_key"):
            entry["bookie_key"] = g.get("bookie_key")
        cached_books = g.get("book_odds")
        if isinstance(cached_books, dict) and cached_books:
            entry["books"] = cached_books
        for key_src, key_dst in (("spread_market", "spread"), ("total_market", "total")):
            val = g.get(key_src)
            if isinstance(val, dict) and val:
                entry[key_dst] = val
        # also support direct cached keys
        for key in ("spread", "total"):
            if key not in entry and isinstance(g.get(key), dict):
                entry[key] = g.get(key)
        if home_odds > 1.0:
            entry[home_abbr] = home_odds
        if away_odds > 1.0:
            entry[away_abbr] = away_odds
        if _global_as_of and "line_markets_as_of" not in entry:
            entry["line_markets_as_of"] = _global_as_of
        cached[(date_str, away_abbr, home_abbr)] = entry
    return cached


def _fetch_nhl_odds_lookup(
    odds_api_key: str,
    session: Any,
    markets: str = "h2h",
    meta_out: dict[str, Any] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Fetch live Pinnacle / best NHL h2h moneyline odds when ODDS_API_KEY env var is present."""
    if not odds_api_key:
        return {}
    url = (
        f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
        f"?bookmakers={ODDS_API_BOOKMAKERS}&markets={markets}&oddsFormat=decimal"
    )
    try:
        res = session.get(url, timeout=15, headers={"x-api-key": odds_api_key})
        raw_headers = getattr(res, "headers", None) or {}
        headers = {str(k).lower(): v for k, v in raw_headers.items()} if hasattr(raw_headers, "items") else {}
        if res.status_code != 200:
            if meta_out is not None:
                meta_out.update(
                    {
                        "api_ok": False,
                        "status_code": res.status_code,
                        "requests_last": _safe_int(headers.get("x-requests-last")),
                        "requests_used": _safe_int(headers.get("x-requests-used")),
                        "requests_remaining": _safe_int(headers.get("x-requests-remaining")),
                    }
                )
            return {}
        data = res.json()
    except Exception:
        if meta_out is not None:
            meta_out.update({"api_ok": False, "status_code": None})
        return {}

    odds_map: dict[tuple[str, str], dict[str, Any]] = {}
    for game in data:
        h_full = game.get("home_team", "")
        a_full = game.get("away_team", "")
        home_abbr = TEAM_FULL_TO_ABBREV.get(h_full)
        away_abbr = TEAM_FULL_TO_ABBREV.get(a_full)
        if not home_abbr or not away_abbr:
            continue

        bookies = game.get("bookmakers", [])
        h2h_books: dict[str, dict[str, Any]] = {}
        spread_books: dict[str, dict[str, Any]] = {}
        total_books: dict[str, dict[str, Any]] = {}
        for book in bookies:
            b_key = str(book.get("key") or "").strip().lower()
            b_title = str(book.get("title") or b_key or "").strip()
            if not b_key or not b_title:
                continue
            for market in book.get("markets", []):
                mkey = market.get("key")
                if mkey == "h2h":
                    h_odds, a_odds = 0.0, 0.0
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == h_full:
                            h_odds = float(outcome.get("price", 0.0) or 0.0)
                        elif outcome.get("name") == a_full:
                            a_odds = float(outcome.get("price", 0.0) or 0.0)
                    if h_odds > 1.0 and a_odds > 1.0:
                        h2h_books[b_key] = {"title": b_title, "home": h_odds, "away": a_odds}
                elif mkey == "spreads":
                    home_point = None
                    away_point = None
                    h_odds, a_odds = 0.0, 0.0
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        pt = outcome.get("point")
                        price = float(outcome.get("price", 0.0) or 0.0)
                        if name == h_full:
                            h_odds = price
                            home_point = pt
                        elif name == a_full:
                            a_odds = price
                            away_point = pt
                    if h_odds > 1.0 and a_odds > 1.0:
                        spread_books[b_key] = {
                            "title": b_title,
                            "home": h_odds,
                            "away": a_odds,
                            "home_point": home_point,
                            "away_point": away_point,
                        }
                elif mkey == "totals":
                    point = None
                    over_odds, under_odds = 0.0, 0.0
                    for outcome in market.get("outcomes", []):
                        name = str(outcome.get("name", "")).lower()
                        pt = outcome.get("point")
                        price = float(outcome.get("price", 0.0) or 0.0)
                        if "over" in name:
                            over_odds = price
                            point = pt
                        elif "under" in name:
                            under_odds = price
                            if point is None:
                                point = pt
                    if over_odds > 1.0 and under_odds > 1.0:
                        total_books[b_key] = {
                            "title": b_title,
                            "over": over_odds,
                            "under": under_odds,
                            "point": point,
                        }
        if not h2h_books:
            continue

        default_key = "pinnacle" if "pinnacle" in h2h_books else next(iter(h2h_books))
        default_line = h2h_books[default_key]
        entry: dict[str, Any] = {
            home_abbr: default_line["home"],
            away_abbr: default_line["away"],
            "bookie": default_line["title"],
            "bookie_key": default_key,
            "books": h2h_books,
        }

        # spread market
        if "spreads" in str(markets).split(",") and spread_books:
            points = [b.get("home_point") for b in spread_books.values() if b.get("home_point") is not None]
            main_point = None
            if points:
                # prefer -1.5
                if -1.5 in [float(p) for p in points if p is not None]:
                    main_point = -1.5
                else:
                    from collections import Counter

                    c = Counter(float(p) for p in points)
                    main_point = c.most_common(1)[0][0]
            # filter books on main line
            main_spread_books = {}
            for bk, info in spread_books.items():
                try:
                    if main_point is None or float(info.get("home_point")) == float(main_point):
                        main_spread_books[bk] = info
                except (TypeError, ValueError):
                    main_spread_books[bk] = info
            if not main_spread_books:
                main_spread_books = spread_books
            # pinnacle
            pin = main_spread_books.get("pinnacle") or (main_spread_books.get(next(iter(main_spread_books))) if main_spread_books else None)
            # best prices
            best_home = None
            best_away = None
            for bk, info in main_spread_books.items():
                if best_home is None or info["home"] > best_home["odds"]:
                    best_home = {"odds": info["home"], "book": info["title"]}
                if best_away is None or info["away"] > best_away["odds"]:
                    best_away = {"odds": info["away"], "book": info["title"]}
            entry["spread"] = {
                "line": float(main_point) if main_point is not None else None,
                "books": main_spread_books,
                "pinnacle": ({"home": pin["home"], "away": pin["away"]} if pin else None),
                "best_home": best_home,
                "best_away": best_away,
                "as_of": datetime.now(timezone.utc).isoformat(),
            }

        # total market
        if "totals" in str(markets).split(",") and total_books:
            points = [b.get("point") for b in total_books.values() if b.get("point") is not None]
            main_point = None
            if points:
                float_points = [float(p) for p in points]
                if 5.5 in float_points:
                    main_point = 5.5
                elif 6.5 in float_points:
                    main_point = 6.5
                else:
                    from collections import Counter

                    c = Counter(float_points)
                    main_point = c.most_common(1)[0][0]
            main_total_books = {}
            for bk, info in total_books.items():
                try:
                    if main_point is None or float(info.get("point")) == float(main_point):
                        main_total_books[bk] = info
                except (TypeError, ValueError):
                    main_total_books[bk] = info
            if not main_total_books:
                main_total_books = total_books
            pin = main_total_books.get("pinnacle") or (main_total_books.get(next(iter(main_total_books))) if main_total_books else None)
            best_over = None
            best_under = None
            for bk, info in main_total_books.items():
                if best_over is None or info["over"] > best_over["odds"]:
                    best_over = {"odds": info["over"], "book": info["title"]}
                if best_under is None or info["under"] > best_under["odds"]:
                    best_under = {"odds": info["under"], "book": info["title"]}
            entry["total"] = {
                "line": float(main_point) if main_point is not None else None,
                "books": main_total_books,
                "pinnacle": ({"over": pin["over"], "under": pin["under"], "point": pin.get("point")} if pin else None),
                "best_over": best_over,
                "best_under": best_under,
                "as_of": datetime.now(timezone.utc).isoformat(),
            }

        odds_map[(away_abbr, home_abbr)] = entry

    line_as_of = None
    for _e in odds_map.values():
        if isinstance(_e, dict):
            if isinstance(_e.get("spread"), dict) and _e["spread"].get("as_of"):
                line_as_of = _e["spread"]["as_of"]
                break
            if isinstance(_e.get("total"), dict) and _e["total"].get("as_of"):
                line_as_of = _e["total"]["as_of"]
                break
    if meta_out is not None:
        meta_out.update(
            {
                "api_ok": True,
                "status_code": res.status_code,
                "fetched_games": len(odds_map),
                "markets": markets,
                "line_markets_as_of": line_as_of,
                "requests_last": _safe_int(headers.get("x-requests-last")),
                "requests_used": _safe_int(headers.get("x-requests-used")),
                "requests_remaining": _safe_int(headers.get("x-requests-remaining")),
            }
        )
    return odds_map


def predict_upcoming_games(
    schedule_games: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
    odds_api_key: str = "",
    session: Any | None = None,
    team_availability: dict[str, dict[str, Any]] | None = None,
    cached_odds: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    odds_meta_out: dict[str, Any] | None = None,
    profiles_unadjusted: dict[str, dict[str, Any]] | None = None,
    team_churn: dict[str, float] | None = None,
    roster_moves: dict[str, Any] | None = None,
    line_bundle: dict[str, Any] | None = None,
    league_total_prior: float | None = None,
    odds_markets: str = "h2h",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate single-game win probabilities and optional active playoff series projections."""
    churn_map: dict[str, float] = dict(team_churn or {})

    def _safe_num(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            f = float(v)
            if not __import__("math").isfinite(f):
                return default
            return f
        except (TypeError, ValueError):
            return default

    odds_map = (
        _fetch_nhl_odds_lookup(odds_api_key, session, markets=odds_markets, meta_out=odds_meta_out)
        if (odds_api_key and session)
        else {}
    )

    scheduled_dates_by_team: dict[str, list[str]] = {}
    for g in schedule_games:
        d_str = str(g.get("date") or "")
        if not d_str:
            continue
        scheduled_dates_by_team.setdefault(g["home"], []).append(d_str)
        scheduled_dates_by_team.setdefault(g["away"], []).append(d_str)

    predictions: list[dict[str, Any]] = []
    seen_series: set[tuple[str, str, int]] = set()
    playoff_series: list[dict[str, Any]] = []

    for g in schedule_games:
        home = g["home"]
        away = g["away"]
        g_date = g.get("date")
        extra_h = scheduled_dates_by_team.get(home, [])
        extra_a = scheduled_dates_by_team.get(away, [])
        home_prob = _pairwise_win_prob(
            home,
            away,
            profiles,
            bundle,
            game_date=g_date,
            extra_dates_a=extra_h,
            extra_dates_b=extra_a,
            venue_team=g["home"],
        )
        away_prob = 1.0 - home_prob

        if profiles_unadjusted is None:
            shadow_home_prob = home_prob
        else:
            shadow_home_prob = _pairwise_win_prob(
                home,
                away,
                profiles_unadjusted,
                bundle,
                game_date=g_date,
                extra_dates_a=extra_h,
                extra_dates_b=extra_a,
                venue_team=g["home"],
            )
        away_prob_unadjusted = 1.0 - shadow_home_prob

        churn_h = float((churn_map or {}).get(home, 0.0) or 0.0)
        churn_a = float((churn_map or {}).get(away, 0.0) or 0.0)
        home_in, home_out = _roster_move_names(roster_moves, home)
        away_in, away_out = _roster_move_names(roster_moves, away)
        home_prob = _shrink_prob_for_churn(home_prob, churn_h, churn_a)
        shadow_home_prob = _shrink_prob_for_churn(shadow_home_prob, churn_h, churn_a)
        away_prob = 1.0 - home_prob
        away_prob_unadjusted = 1.0 - shadow_home_prob

        avail_h = (team_availability or {}).get(home, {})
        avail_a = (team_availability or {}).get(away, {})
        home_depth_eff = avail_h.get("depth_eff", 1.0)
        away_depth_eff = avail_a.get("depth_eff", 1.0)
        st_h = profiles.get(home, {})
        st_a = profiles.get(away, {})
        b2b_h, rest_h, f3in4_h = _schedule_fatigue_for_date(st_h, g_date, extra_dates=extra_h)
        b2b_a, rest_a, f3in4_a = _schedule_fatigue_for_date(st_a, g_date, extra_dates=extra_a)

        game_odds = odds_map.get((away, home)) or (cached_odds or {}).get((str(g_date or ""), away, home)) or {}
        home_edge = evaluate_kelly_edge(home_prob, game_odds.get(home), min_edge_pct=2.0)
        away_edge = evaluate_kelly_edge(away_prob, game_odds.get(away), min_edge_pct=2.0)
        # (availability already resolved above)

        # --- line models (puck line + total goals) ---
        spread_model_out = None
        total_model_out = None
        spread_market = game_odds.get("spread") if isinstance(game_odds, dict) else None
        total_market = game_odds.get("total") if isinstance(game_odds, dict) else None

        # build line feature vector from profiles
        _spread_model_out = None
        _total_model_out = None
        _spread_market = spread_market
        _total_market = total_market
        _home_spread_edge = None
        _away_spread_edge = None
        _over_edge = None
        _under_edge = None
        if line_bundle is not None:
            try:
                import src.line_models as _lm

                ltp = 6.24 if league_total_prior is None else float(league_total_prior)
                st_h = profiles.get(home, {}) or {}
                st_a = profiles.get(away, {}) or {}
                prior_gd_h = _safe_num(st_h.get("Prior_GD", st_h.get("Prior_GD", 0.0)))
                prior_gd_a = _safe_num(st_a.get("Prior_GD", 0.0))
                prior_xgd_h = _safe_num(st_h.get("Prior_xGD", 0.0))
                prior_xgd_a = _safe_num(st_a.get("Prior_xGD", 0.0))
                prior_gf_h = _safe_num(st_h.get("Prior_GF", 3.1))
                prior_ga_h = _safe_num(st_h.get("Prior_GA", 3.1))
                prior_gf_a = _safe_num(st_a.get("Prior_GF", 3.1))
                prior_ga_a = _safe_num(st_a.get("Prior_GA", 3.1))

                gf_h = _safe_num(st_h.get("goalsFor_per_game", prior_gf_h))
                ga_h = _safe_num(st_h.get("goalsAgainst_per_game", prior_ga_h))
                gf_a = _safe_num(st_a.get("goalsFor_per_game", prior_gf_a))
                ga_a = _safe_num(st_a.get("goalsAgainst_per_game", prior_ga_a))

                prior_xgf_h = _safe_num(st_h.get("Prior_xGF", 3.1))
                prior_xga_h = _safe_num(st_h.get("Prior_xGA", 3.1))
                prior_xgf_a = _safe_num(st_a.get("Prior_xGF", 3.1))
                prior_xga_a = _safe_num(st_a.get("Prior_xGA", 3.1))
                exp_margin = (prior_gf_h - prior_ga_h) - (prior_gf_a - prior_ga_a)
                exp_margin_xg = (prior_xgf_h - prior_xga_h) - (prior_xgf_a - prior_xga_a)
                exp_total = ((prior_gf_h + prior_ga_a) + (prior_gf_a + prior_ga_h)) / 2.0
                exp_total_xg = ((prior_xgf_h + prior_xga_a) + (prior_xgf_a + prior_xga_h)) / 2.0
                cur_exp_margin = (gf_h - ga_h) - (gf_a - ga_a)
                cur_exp_total = (gf_h + ga_h + gf_a + ga_a) / 2.0
                cur_gp_h = _safe_num(st_h.get("games_played", 0.0))
                cur_gp_a = _safe_num(st_a.get("games_played", 0.0))
                cur_gp_min = min(cur_gp_h, cur_gp_a)
                k = 10.0
                cur_exp_margin_s = (cur_gp_min * cur_exp_margin + k * exp_margin) / (cur_gp_min + k)
                cur_exp_total_s = (cur_gp_min * cur_exp_total + k * exp_total) / (cur_gp_min + k)

                feats = {}
                # initialize all feature names to 0
                for fn in line_bundle.get("feature_names", []):
                    feats[fn] = 0.0
                # base pairwise-style diffs
                feats["Elo_diff"] = _safe_num(st_h.get("Causal_Elo", 1500.0)) - _safe_num(st_a.get("Causal_Elo", 1500.0))
                feats["Prior_GD_diff"] = exp_margin
                feats["Prior_xGD_diff"] = exp_margin_xg
                feats["Blended_GD_GP_diff"] = _safe_num(st_h.get("Blended_GD_GP", 0.0)) - _safe_num(st_a.get("Blended_GD_GP", 0.0))
                feats["EWMA_GD_diff"] = _safe_num(st_h.get("EWMA_GD", 0.0)) - _safe_num(st_a.get("EWMA_GD", 0.0))
                feats["Rolling10_GoalDiff_diff"] = _safe_num(st_h.get("Rolling10_GoalDiff", 0.0)) - _safe_num(st_a.get("Rolling10_GoalDiff", 0.0))
                feats["Top3_Points_per_game_diff"] = _safe_num(st_h.get("Top3_Points_per_game", 0.0)) - _safe_num(st_a.get("Top3_Points_per_game", 0.0))
                feats["Team_GSAx_per_game_diff"] = _safe_num(st_h.get("Team_GSAx_per_game", 0.0)) - _safe_num(st_a.get("Team_GSAx_per_game", 0.0))
                feats["Starter_SavePct_diff"] = _safe_num(st_h.get("Starter_SavePct", 0.905)) - _safe_num(st_a.get("Starter_SavePct", 0.905))
                # line aggregates
                feats["exp_margin"] = exp_margin
                feats["exp_margin_xg"] = exp_margin_xg
                feats["exp_total"] = exp_total
                feats["exp_total_xg"] = exp_total_xg
                feats["pace"] = (_safe_num(st_h.get("Prior_Attempts", 60.0)) + _safe_num(st_a.get("Prior_Attempts", 60.0))) / 2.0
                feats["shots_sum"] = _safe_num(st_h.get("Prior_ShotsFor", 30.0)) + _safe_num(st_a.get("Prior_ShotsFor", 30.0))
                feats["shots_against_sum"] = _safe_num(st_h.get("Prior_ShotsAgainst", 30.0)) + _safe_num(st_a.get("Prior_ShotsAgainst", 30.0))
                feats["danger_sum"] = _safe_num(st_h.get("Prior_DangerFor", 8.0)) + _safe_num(st_a.get("Prior_DangerFor", 8.0))
                feats["danger_against_sum"] = _safe_num(st_h.get("Prior_DangerAgainst", 8.0)) + _safe_num(st_a.get("Prior_DangerAgainst", 8.0))
                feats["save_pct_sum"] = _safe_num(st_h.get("Prior_Starter_SavePct", 0.905)) + _safe_num(st_a.get("Prior_Starter_SavePct", 0.905))
                feats["gsax_sum"] = _safe_num(st_h.get("Prior_Team_GSAx", 0.0)) + _safe_num(st_a.get("Prior_Team_GSAx", 0.0))
                # causal current
                feats["cur_exp_margin"] = cur_exp_margin
                feats["cur_exp_total"] = cur_exp_total
                feats["cur_exp_margin_s"] = cur_exp_margin_s
                feats["cur_exp_total_s"] = cur_exp_total_s
                feats["cur_gp_min"] = cur_gp_min
                feats["league_total_prior"] = ltp
                # prior xg total approximate
                feats["exp_total_xg"] = exp_total

                spread_line = spread_market.get("line") if isinstance(spread_market, dict) else None
                if spread_line is None:
                    spread_line = -1.5
                sp = _lm.predict_spread(line_bundle, feats, float(spread_line))
                spread_model_out = {
                    "line": spread_line,
                    "home_cover_prob": round(float(sp["home_cover_prob"]) * 100.0, 1),
                    "away_cover_prob": round(float(sp["away_cover_prob"]) * 100.0, 1),
                    "push_prob": round(float(sp.get("push_prob", 0.0)) * 100.0, 1),
                    "exp_margin": round(float(sp.get("exp_margin", exp_margin)), 4),
                    "validated": True,
                    "metrics": (line_bundle.get("spread", {}) or {}).get("metrics", {}),
                }

                total_line = total_market.get("line") if isinstance(total_market, dict) else None
                if total_line is None:
                    total_line = 5.5
                tp = _lm.predict_total(line_bundle, feats, float(total_line))
                total_model_out = {
                    "line": total_line,
                    "over_prob": round(float(tp["over"]) * 100.0, 1),
                    "under_prob": round(float(tp["under"]) * 100.0, 1),
                    "push_prob": round(float(tp.get("push", 0.0)) * 100.0, 1),
                    "exp_total": round(float(tp.get("exp_total", exp_total)), 4),
                    "validated": False,
                    "experimental": True,
                    "stake_enabled": False,
                    "metrics": (line_bundle.get("total", {}) or {}).get("metrics", {}),
                }

                # edges for markets
                if isinstance(spread_market, dict):
                    home_price = None
                    away_price = None
                    if isinstance(spread_market.get("best_home"), dict):
                        home_price = spread_market["best_home"].get("odds")
                    if isinstance(spread_market.get("best_away"), dict):
                        away_price = spread_market["best_away"].get("odds")
                    if home_price is None and isinstance(spread_market.get("pinnacle"), dict):
                        home_price = spread_market["pinnacle"].get("home")
                    if away_price is None and isinstance(spread_market.get("pinnacle"), dict):
                        away_price = spread_market["pinnacle"].get("away")
                    home_cover_frac = float(sp["home_cover_prob"])
                    away_cover_frac = float(sp["away_cover_prob"])
                    home_spread_edge = _lm.evaluate_line_edge(home_cover_frac, home_price, min_edge_pct=2.0, stake_enabled=True)
                    away_spread_edge = _lm.evaluate_line_edge(away_cover_frac, away_price, min_edge_pct=2.0, stake_enabled=True)
                else:
                    home_spread_edge = None
                    away_spread_edge = None

                if isinstance(total_market, dict):
                    over_price = None
                    under_price = None
                    if isinstance(total_market.get("best_over"), dict):
                        over_price = total_market["best_over"].get("odds")
                    if isinstance(total_market.get("best_under"), dict):
                        under_price = total_market["best_under"].get("odds")
                    if over_price is None and isinstance(total_market.get("pinnacle"), dict):
                        over_price = total_market["pinnacle"].get("over")
                    if under_price is None and isinstance(total_market.get("pinnacle"), dict):
                        under_price = total_market["pinnacle"].get("under")
                    over_edge = _lm.evaluate_line_edge(float(tp["over"]), over_price, min_edge_pct=2.0, stake_enabled=False)
                    under_edge = _lm.evaluate_line_edge(float(tp["under"]), under_price, min_edge_pct=2.0, stake_enabled=False)
                else:
                    over_edge = None
                    under_edge = None
            except Exception:
                spread_model_out = None
                total_model_out = None
                home_spread_edge = None
                away_spread_edge = None
                over_edge = None
                under_edge = None

        # attach to prediction dict later; store in locals for now
        _spread_model_out = spread_model_out
        _total_model_out = total_model_out
        _spread_market = spread_market
        _total_market = total_market
        _home_spread_edge = home_spread_edge if line_bundle is not None else None
        _away_spread_edge = away_spread_edge if line_bundle is not None else None
        _over_edge = over_edge if line_bundle is not None else None
        _under_edge = under_edge if line_bundle is not None else None

        predictions.append(
            {
                "id": g["id"],
                "date": g["date"],
                "is_today": g["is_today"],
                "start_time_utc": g.get("start_time_utc"),
                "start_time_et": g["start_time_et"],
                "game_state": g["game_state"],
                "game_type": g["game_type"],
                "away": away,
                "home": home,
                "away_score": g.get("away_score"),
                "home_score": g.get("home_score"),
                "away_prob": round(away_prob * 100.0, 1),
                "home_prob": round(home_prob * 100.0, 1),
                "home_prob_unadjusted": round(shadow_home_prob * 100.0, 1),
                "away_prob_unadjusted": round(away_prob_unadjusted * 100.0, 1),
                "home_depth_eff": round(float(home_depth_eff), 4),
                "away_depth_eff": round(float(away_depth_eff), 4),
                "home_roster_churn": round(float(churn_h), 3),
                "away_roster_churn": round(float(churn_a), 3),
                "home_roster_in": home_in,
                "home_roster_out": home_out,
                "away_roster_in": away_in,
                "away_roster_out": away_out,
                "favored": home if home_prob >= away_prob else away,
                "fav_prob": round(max(home_prob, away_prob) * 100.0, 1),
                "away_streak": int(st_a.get("Streak", 0)),
                "home_streak": int(st_h.get("Streak", 0)),
                "away_b2b": bool(b2b_a >= 0.5),
                "home_b2b": bool(b2b_h >= 0.5),
                "away_3in4": bool(f3in4_a >= 0.5),
                "home_3in4": bool(f3in4_h >= 0.5),
                "away_rest": int(round(rest_a)),
                "home_rest": int(round(rest_h)),
                "away_elo": round(float(st_a.get("Causal_Elo", 1500.0)), 1),
                "home_elo": round(float(st_h.get("Causal_Elo", 1500.0)), 1),
                "away_gsax": round(float(st_a.get("Team_GSAx", 0.0)), 1),
                "home_gsax": round(float(st_h.get("Team_GSAx", 0.0)), 1),
                "away_top3_pts": int(round(float(st_a.get("Top3_Points", 0.0)))),
                "home_top3_pts": int(round(float(st_h.get("Top3_Points", 0.0)))),
                "away_goalie": avail_a.get("goalie"),
                "home_goalie": avail_h.get("goalie"),
                "away_injuries": [r["player"] + f" ({r['status']})" for r in (avail_a.get("out_skaters") or [])[:4]],
                "home_injuries": [r["player"] + f" ({r['status']})" for r in (avail_h.get("out_skaters") or [])[:4]],
                "bookie": game_odds.get("bookie"),
                "book_odds": game_odds.get("books") or {},
                "bookie_key": game_odds.get("bookie_key"),
                "away_odds": round(float(game_odds[away]), 3) if game_odds.get(away) else None,
                "home_odds": round(float(game_odds[home]), 3) if game_odds.get(home) else None,
                "away_edge": away_edge,
                "home_edge": home_edge,
                "spread_model": _spread_model_out,
                "total_model": _total_model_out,
                "spread_market": _spread_market,
                "total_market": _total_market,
                "home_spread_edge": _home_spread_edge,
                "away_spread_edge": _away_spread_edge,
                "over_edge": _over_edge,
                "under_edge": _under_edge,
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
                p_home = _pairwise_win_prob(t1, t2, profiles, bundle)
                p_away = 1.0 - _pairwise_win_prob(t2, t1, profiles, bundle)
                p_series_t1 = series_prob_22111(p_home, p_away, w1, w2)
                exp_rem = expected_remaining_games_22111(p_home, p_away, w1, w2)
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
