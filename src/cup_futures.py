"""Stanley Cup Futures Monte Carlo Simulator & Portfolio Kelly Engine.

Supersedes the older notebook bracket/futures logic by pairing a dedicated
long-horizon pairwise matchup model (free of short-term B2B/streak noise) with
NHL standings, remaining schedule simulation (including OT loser points and RW/ROW
tiebreakers), true 2-2-1-1-1 Best-of-7 series math, sharp-market shrinkage, and
simultaneous mutually-exclusive Portfolio Kelly sizing.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize

from src.data_sync import (
    TEAM_FULL_TO_ABBREV,
    _fetch_with_retries,
    create_http_session,
    get_nhl_season_context,
    normalize_team_abbrev,
    sync_game_logs,
    sync_moneypuck_data,
)
from src.report_builder import write_json_report
from src.team_game_model import (
    _pairwise_win_prob,
    _safe_int,
    build_live_team_profiles,
    build_team_star_goalie_history,
    compute_power_rankings_and_defense,
    extract_latest_team_form,
    load_active_goalie_lookup,
    save_pairwise_bundle,
    series_prob_22111,
    train_pairwise_game_model,
)

FUTURES_FEATURES: list[str] = [
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
]

FUTURES_ODDS_BOOKMAKERS = "pinnacle,lowvig,betonlineag,draftkings,fanduel,betmgm,betrivers,espnbet,bovada,playnow_ca"

NHL_DIVISIONS: dict[str, tuple[str, str]] = {
    # Eastern — Atlantic
    "BOS": ("Eastern", "Atlantic"),
    "BUF": ("Eastern", "Atlantic"),
    "DET": ("Eastern", "Atlantic"),
    "FLA": ("Eastern", "Atlantic"),
    "MTL": ("Eastern", "Atlantic"),
    "OTT": ("Eastern", "Atlantic"),
    "TBL": ("Eastern", "Atlantic"),
    "TOR": ("Eastern", "Atlantic"),
    # Eastern — Metropolitan
    "CAR": ("Eastern", "Metropolitan"),
    "CBJ": ("Eastern", "Metropolitan"),
    "NJD": ("Eastern", "Metropolitan"),
    "NYI": ("Eastern", "Metropolitan"),
    "NYR": ("Eastern", "Metropolitan"),
    "PHI": ("Eastern", "Metropolitan"),
    "PIT": ("Eastern", "Metropolitan"),
    "WSH": ("Eastern", "Metropolitan"),
    # Western — Central
    "CHI": ("Western", "Central"),
    "COL": ("Western", "Central"),
    "DAL": ("Western", "Central"),
    "MIN": ("Western", "Central"),
    "NSH": ("Western", "Central"),
    "STL": ("Western", "Central"),
    "UTA": ("Western", "Central"),
    "WPG": ("Western", "Central"),
    # Western — Pacific
    "ANA": ("Western", "Pacific"),
    "CGY": ("Western", "Pacific"),
    "EDM": ("Western", "Pacific"),
    "LAK": ("Western", "Pacific"),
    "SEA": ("Western", "Pacific"),
    "SJS": ("Western", "Pacific"),
    "VAN": ("Western", "Pacific"),
    "VGK": ("Western", "Pacific"),
}

CONFERENCE_DIVISIONS: dict[str, tuple[str, str]] = {
    "Eastern": ("Atlantic", "Metropolitan"),
    "Western": ("Central", "Pacific"),
}


def _normalize_conference(raw: Any, fallback: str) -> str:
    txt = str(raw or "").strip()
    if "East" in txt:
        return "Eastern"
    if "West" in txt:
        return "Western"
    return fallback


def _normalize_division(raw: Any, fallback: str) -> str:
    txt = str(raw or "").strip()
    for div in ("Atlantic", "Metropolitan", "Central", "Pacific"):
        if div.lower() in txt.lower():
            return div
    return fallback


def fetch_nhl_standings(session: Any) -> tuple[dict[str, dict[str, Any]], float]:
    """Fetch current NHL standings and empirical OT/SO rate."""
    res = _fetch_with_retries(session, "https://api-web.nhle.com/v1/standings/now", timeout=15, attempts=2)
    standings: dict[str, dict[str, Any]] = {}
    if res is None or res.status_code != 200:
        return standings, 0.23

    total_ot_losses = 0
    total_gp = 0
    for row in res.json().get("standings", []):
        raw_abbrev = row.get("teamAbbrev")
        if isinstance(raw_abbrev, dict):
            raw_abbrev = raw_abbrev.get("default")
        team = normalize_team_abbrev(raw_abbrev)
        if not team:
            continue

        fallback_conf, fallback_div = NHL_DIVISIONS.get(team, ("Western", "Central"))
        conf = _normalize_conference(row.get("conferenceName"), fallback_conf)
        div = _normalize_division(row.get("divisionName"), fallback_div)
        gp = int(row.get("gamesPlayed", 0) or 0)
        wins = int(row.get("wins", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        ot_losses = int(row.get("otLosses", 0) or 0)
        points = int(row.get("points", wins * 2 + ot_losses) or 0)
        rw = int(row.get("regulationWins", 0) or 0)
        row_wins = int(row.get("regulationPlusOtWins", wins) or 0)
        goal_diff = int(row.get("goalDifferential", 0) or 0)

        standings[team] = {
            "team": team,
            "conference": conf,
            "division": div,
            "gp": gp,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "points": points,
            "rw": rw,
            "row": row_wins,
            "goal_diff": goal_diff,
        }
        total_ot_losses += ot_losses
        total_gp += gp

    if total_gp >= 40:
        ot_rate = float(np.clip(total_ot_losses / max(1.0, total_gp / 2.0), 0.18, 0.28))
    else:
        ot_rate = 0.23
    return standings, round(ot_rate, 4)


def fetch_remaining_regular_schedule(
    season: str,
    teams: list[str],
    session: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch deduplicated remaining regular-season games and any active playoff series."""
    seen_reg: set[int] = set()
    remaining_games: list[dict[str, Any]] = []
    series_map: dict[tuple[int, str, str], dict[str, Any]] = {}

    for team in teams:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
        res = _fetch_with_retries(session, url, timeout=12, attempts=2)
        if res is None or res.status_code != 200:
            continue

        for game in res.json().get("games", []):
            gid = int(game.get("id", 0) or 0)
            game_type = int(game.get("gameType", 0) or 0)
            state = str(game.get("gameState") or "FUT").upper()
            away = normalize_team_abbrev((game.get("awayTeam") or {}).get("abbrev"))
            home = normalize_team_abbrev((game.get("homeTeam") or {}).get("abbrev"))
            if not away or not home:
                continue

            if game_type == 2 and state in ("FUT", "PRE", "LIVE", "CRIT"):
                if gid > 0:
                    if gid in seen_reg:
                        continue
                    seen_reg.add(gid)
                remaining_games.append(
                    {
                        "id": gid,
                        "date": str(game.get("gameDate") or ""),
                        "away": away,
                        "home": home,
                        "game_state": state,
                    }
                )
            elif game_type == 3:
                ss = game.get("seriesStatus") or {}
                if not ss:
                    continue
                t1 = normalize_team_abbrev(ss.get("topSeedTeamAbbrev") or home)
                t2 = normalize_team_abbrev(ss.get("bottomSeedTeamAbbrev") or away)
                rnd = int(ss.get("round", 1) or 1)
                w1 = int(ss.get("topSeedWins", 0) or 0)
                w2 = int(ss.get("bottomSeedWins", 0) or 0)
                if not t1 or not t2:
                    continue
                pair = tuple(sorted((t1, t2)))
                s_key = (rnd, pair[0], pair[1])
                series_map[s_key] = {
                    "round": rnd,
                    "top_seed": t1,
                    "top_wins": w1,
                    "bottom_seed": t2,
                    "bottom_wins": w2,
                }

    active_series = sorted(series_map.values(), key=lambda s: (s["round"], s["top_seed"]))
    return remaining_games, active_series


def fetch_cup_futures_odds(
    odds_api_key: str,
    session: Any,
    meta_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch 1-credit Stanley Cup outright futures and devig sharp market probabilities."""
    empty = {
        "api_ok": False,
        "best_odds": {},
        "best_bookie": {},
        "market_probs": {},
        "market_overround_pct": None,
        "devig_source": "none",
    }
    if not odds_api_key or session is None:
        return empty

    url = (
        f"https://api.the-odds-api.com/v4/sports/icehockey_nhl_championship_winner/odds/"
        f"?apiKey={odds_api_key}&bookmakers={FUTURES_ODDS_BOOKMAKERS}&markets=outrights&oddsFormat=decimal"
    )
    try:
        res = session.get(url, timeout=15)
        raw_headers = getattr(res, "headers", None) or {}
        headers = {str(k).lower(): v for k, v in raw_headers.items()} if hasattr(raw_headers, "items") else {}
        if meta_out is not None:
            meta_out.update(
                {
                    "status_code": res.status_code,
                    "requests_last": _safe_int(headers.get("x-requests-last")),
                    "requests_used": _safe_int(headers.get("x-requests-used")),
                    "requests_remaining": _safe_int(headers.get("x-requests-remaining")),
                }
            )
        if res.status_code != 200:
            if meta_out is not None:
                meta_out["api_ok"] = False
            return empty
        data = res.json()
    except Exception:
        if meta_out is not None:
            meta_out["api_ok"] = False
        return empty

    best_odds: dict[str, float] = {}
    best_bookie: dict[str, str] = {}
    book_devig_probs: dict[str, dict[str, float]] = {}
    book_overrounds: dict[str, float] = {}

    for event in data or []:
        for bookie in event.get("bookmakers", []):
            b_key = str(bookie.get("key") or "")
            b_title = str(bookie.get("title") or b_key or "Bookie")
            for market in bookie.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                raw_implied: dict[str, float] = {}
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name") or "").strip()
                    abbr = TEAM_FULL_TO_ABBREV.get(name) or normalize_team_abbrev(name)
                    if not abbr or len(abbr) > 4:
                        continue
                    try:
                        price = float(outcome.get("price", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    if price <= 1.0:
                        continue
                    if abbr not in best_odds or price > best_odds[abbr]:
                        best_odds[abbr] = round(price, 3)
                        best_bookie[abbr] = b_title
                    raw_implied[abbr] = max(raw_implied.get(abbr, 0.0), 1.0 / price)

                implied_sum = sum(raw_implied.values())
                if len(raw_implied) >= 2 and implied_sum > 0.0:
                    book_devig_probs[b_key] = {t: v / implied_sum for t, v in raw_implied.items()}
                    book_overrounds[b_key] = (implied_sum - 1.0) * 100.0

    # Devig market consensus: prefer Pinnacle when well-populated, else average full boards
    market_probs: dict[str, float] = {}
    market_overround_pct: float | None = None
    devig_source = "none"
    if book_devig_probs:
        max_quoted = max(len(v) for v in book_devig_probs.values())
        min_required = max(2, int(0.75 * max_quoted))
        complete_books = {k: v for k, v in book_devig_probs.items() if len(v) >= min_required}
        if not complete_books:
            complete_books = book_devig_probs

        if "pinnacle" in complete_books:
            market_probs = dict(complete_books["pinnacle"])
            market_overround_pct = round(book_overrounds["pinnacle"], 2)
            devig_source = "pinnacle"
        else:
            all_teams = {t for probs in complete_books.values() for t in probs}
            raw_avg = {
                t: sum(probs.get(t, 0.0) for probs in complete_books.values()) / len(complete_books)
                for t in all_teams
            }
            tot = sum(raw_avg.values())
            if tot > 0.0:
                market_probs = {t: v / tot for t, v in raw_avg.items()}
                market_overround_pct = round(
                    sum(book_overrounds[k] for k in complete_books) / len(complete_books),
                    2,
                )
                devig_source = "consensus"

    if meta_out is not None:
        meta_out["api_ok"] = True
        meta_out["quoted_teams"] = len(best_odds)
        meta_out["devig_source"] = devig_source

    return {
        "api_ok": True,
        "best_odds": best_odds,
        "best_bookie": best_bookie,
        "market_probs": market_probs,
        "market_overround_pct": market_overround_pct,
        "devig_source": devig_source,
    }


def train_futures_game_model(
    df_team_hist: pd.DataFrame,
    df_stars: pd.DataFrame,
    df_goalies: pd.DataFrame,
    df_master_logs: pd.DataFrame,
) -> dict[str, Any]:
    """Train the long-horizon futures pairwise model using durable team strength features only."""
    return train_pairwise_game_model(
        df_team_hist,
        df_stars,
        df_goalies,
        df_master_logs,
        feature_subset=FUTURES_FEATURES,
        playoff_weight=2.5,
    )


def build_matchup_matrices(
    teams: list[str],
    profiles: dict[str, dict[str, Any]],
    futures_bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute N x N single-game home win probs and 2-2-1-1-1 Best-of-7 series matrices."""
    n = len(teams)
    p_home_mat = np.full((n, n), 0.5, dtype=float)
    for i, t_hi in enumerate(teams):
        for j, t_lo in enumerate(teams):
            if i == j:
                continue
            p_home_mat[i, j] = _pairwise_win_prob(t_hi, t_lo, profiles, futures_bundle)

    series_mat = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_h = float(p_home_mat[i, j])
            p_a = float(1.0 - p_home_mat[j, i])
            series_mat[i, j] = series_prob_22111(p_h, p_a)

    return p_home_mat, series_mat


def _play_round_vectorized(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    sort_score: np.ndarray,
    series_mat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve one playoff round across n_sims choosing 2-2-1-1-1 home ice by regular-season rank."""
    n_sims = len(idx_a)
    rows = np.arange(n_sims)
    a_higher = sort_score[rows, idx_a] >= sort_score[rows, idx_b]
    hi = np.where(a_higher, idx_a, idx_b)
    lo = np.where(a_higher, idx_b, idx_a)
    p_hi = series_mat[hi, lo]
    return np.where(rng.random(n_sims) < p_hi, hi, lo)


def simulate_season_and_cup(
    standings: dict[str, dict[str, Any]],
    remaining_games: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    futures_bundle: dict[str, Any],
    *,
    ot_rate: float = 0.23,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Simulate the remaining NHL regular season and 4-round 2-2-1-1-1 Stanley Cup playoffs."""
    teams = sorted(set(NHL_DIVISIONS.keys()) | set(standings.keys()))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    p_home_mat, series_mat = build_matchup_matrices(teams, profiles, futures_bundle)
    rng = np.random.default_rng(seed)

    pts = np.zeros((n_sims, n_teams), dtype=float)
    rw = np.zeros((n_sims, n_teams), dtype=float)
    row_wins = np.zeros((n_sims, n_teams), dtype=float)
    wins = np.zeros((n_sims, n_teams), dtype=float)

    div_map: dict[tuple[str, str], list[int]] = {
        ("Eastern", "Atlantic"): [],
        ("Eastern", "Metropolitan"): [],
        ("Western", "Central"): [],
        ("Western", "Pacific"): [],
    }
    for i, t in enumerate(teams):
        st = standings.get(t, {})
        pts[:, i] = float(st.get("points", 0))
        rw[:, i] = float(st.get("rw", 0))
        row_wins[:, i] = float(st.get("row", st.get("wins", 0)))
        wins[:, i] = float(st.get("wins", 0))

        fb_conf, fb_div = NHL_DIVISIONS.get(t, ("Western", "Central"))
        conf = _normalize_conference(st.get("conference"), fb_conf)
        div = _normalize_division(st.get("division"), fb_div)
        div_map.setdefault((conf, div), []).append(i)

    valid_games = [
        (team_idx[g["away"]], team_idx[g["home"]])
        for g in (remaining_games or [])
        if g.get("away") in team_idx and g.get("home") in team_idx and g.get("away") != g.get("home")
    ]
    n_games = len(valid_games)
    if n_games > 0:
        a_arr = np.asarray([g[0] for g in valid_games], dtype=int)
        h_arr = np.asarray([g[1] for g in valid_games], dtype=int)
        p_vec = p_home_mat[h_arr, a_arr]

        u_win = rng.random((n_sims, n_games))
        u_ot = rng.random((n_sims, n_games))
        u_so = rng.random((n_sims, n_games))
        home_won = u_win < p_vec
        is_ot = u_ot < float(ot_rate)
        is_row = (~is_ot) | (u_so >= 0.30)

        for g_idx in range(n_games):
            ai = a_arr[g_idx]
            hi = h_arr[g_idx]
            hw = home_won[:, g_idx]
            aw = ~hw
            ot_g = is_ot[:, g_idx]
            reg_g = ~ot_g
            row_g = is_row[:, g_idx]

            hw_f = hw.astype(float)
            aw_f = aw.astype(float)
            ot_f = ot_g.astype(float)

            pts[:, hi] += 2.0 * hw_f + (aw_f * ot_f)
            pts[:, ai] += 2.0 * aw_f + (hw_f * ot_f)

            wins[:, hi] += hw_f
            wins[:, ai] += aw_f

            reg_f = reg_g.astype(float)
            rw[:, hi] += hw_f * reg_f
            rw[:, ai] += aw_f * reg_f

            row_f = row_g.astype(float)
            row_wins[:, hi] += hw_f * row_f
            row_wins[:, ai] += aw_f * row_f

    # Encode exact NHL tiebreaker precedence: Points > RW > ROW > Wins > uniform jitter
    sort_score = (
        pts * 1e7
        + rw * 1e5
        + row_wins * 1e3
        + wins * 10.0
        + rng.uniform(0.0, 1.0, size=(n_sims, n_teams))
    )

    make_playoffs_cnt = np.zeros(n_teams, dtype=float)
    win_div_cnt = np.zeros(n_teams, dtype=float)
    conf_final_cnt = np.zeros(n_teams, dtype=float)
    cup_final_cnt = np.zeros(n_teams, dtype=float)
    cup_win_cnt = np.zeros(n_teams, dtype=float)

    rows = np.arange(n_sims)
    conf_champs: dict[str, np.ndarray] = {}

    for conf in ("Eastern", "Western"):
        d1_name, d2_name = CONFERENCE_DIVISIONS[conf]
        d1_idx = np.asarray(div_map[(conf, d1_name)], dtype=int)
        d2_idx = np.asarray(div_map[(conf, d2_name)], dtype=int)

        d1_order = d1_idx[np.argsort(-sort_score[:, d1_idx], axis=1)]
        d2_order = d2_idx[np.argsort(-sort_score[:, d2_idx], axis=1)]

        d1_win = d1_order[:, 0]
        d1_2 = d1_order[:, 1]
        d1_3 = d1_order[:, 2]

        d2_win = d2_order[:, 0]
        d2_2 = d2_order[:, 1]
        d2_3 = d2_order[:, 2]

        wc_pool = np.concatenate([d1_order[:, 3:], d2_order[:, 3:]], axis=1)
        wc_scores = sort_score[rows[:, None], wc_pool]
        wc_order = np.take_along_axis(wc_pool, np.argsort(-wc_scores, axis=1), axis=1)
        wc1 = wc_order[:, 0]
        wc2 = wc_order[:, 1]

        for col in (d1_win, d1_2, d1_3, d2_win, d2_2, d2_3, wc1, wc2):
            np.add.at(make_playoffs_cnt, col, 1.0)
        np.add.at(win_div_cnt, d1_win, 1.0)
        np.add.at(win_div_cnt, d2_win, 1.0)

        d1_is_top = sort_score[rows, d1_win] >= sort_score[rows, d2_win]
        opp_d1 = np.where(d1_is_top, wc2, wc1)
        opp_d2 = np.where(d1_is_top, wc1, wc2)

        # Round 1
        r1_d1_top = _play_round_vectorized(d1_win, opp_d1, sort_score, series_mat, rng)
        r1_d1_low = _play_round_vectorized(d1_2, d1_3, sort_score, series_mat, rng)
        r1_d2_top = _play_round_vectorized(d2_win, opp_d2, sort_score, series_mat, rng)
        r1_d2_low = _play_round_vectorized(d2_2, d2_3, sort_score, series_mat, rng)

        # Round 2 (Division Finals)
        cf_1 = _play_round_vectorized(r1_d1_top, r1_d1_low, sort_score, series_mat, rng)
        cf_2 = _play_round_vectorized(r1_d2_top, r1_d2_low, sort_score, series_mat, rng)

        np.add.at(conf_final_cnt, cf_1, 1.0)
        np.add.at(conf_final_cnt, cf_2, 1.0)

        # Round 3 (Conference Final)
        conf_champs[conf] = _play_round_vectorized(cf_1, cf_2, sort_score, series_mat, rng)
        np.add.at(cup_final_cnt, conf_champs[conf], 1.0)

    # Round 4 (Stanley Cup Final)
    cup_winners = _play_round_vectorized(
        conf_champs["Eastern"],
        conf_champs["Western"],
        sort_score,
        series_mat,
        rng,
    )
    np.add.at(cup_win_cnt, cup_winners, 1.0)

    mean_pts = pts.mean(axis=0)
    results: dict[str, dict[str, float]] = {}
    for i, t in enumerate(teams):
        results[t] = {
            "proj_points": round(float(mean_pts[i]), 1),
            "make_playoffs_prob": float(make_playoffs_cnt[i] / n_sims),
            "win_division_prob": float(win_div_cnt[i] / n_sims),
            "make_conf_final_prob": float(conf_final_cnt[i] / n_sims),
            "make_cup_final_prob": float(cup_final_cnt[i] / n_sims),
            "win_cup_prob": float(cup_win_cnt[i] / n_sims),
        }
    return results


def compute_exact_playoff_cup_probs(
    active_series: list[dict[str, Any]],
    standings: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    futures_bundle: dict[str, Any],
) -> dict[str, dict[str, float]] | None:
    """Exact 2-2-1-1-1 bracket tree Cup probabilities once NHL playoff rounds start."""
    if not active_series:
        return None
    teams = sorted(set(NHL_DIVISIONS.keys()) | set(standings.keys()))
    team_idx = {t: i for i, t in enumerate(teams)}
    p_home_mat, series_mat = build_matchup_matrices(teams, profiles, futures_bundle)

    min_round = min(int(s.get("round", 1)) for s in active_series)
    cur_series = [s for s in active_series if int(s.get("round", 1)) == min_round]

    def _standings_rank(t: str) -> tuple[int, int, int, int]:
        st = standings.get(t, {})
        return (int(st.get("points", 0)), int(st.get("rw", 0)), int(st.get("row", 0)), int(st.get("wins", 0)))

    def _div_sort_key(s: dict[str, Any]) -> tuple[int, str, int, str]:
        t_top = s["top_seed"]
        st = standings.get(t_top, {})
        fb_c, fb_d = NHL_DIVISIONS.get(t_top, ("Western", "Central"))
        c_name = _normalize_conference(st.get("conference"), fb_c)
        d_name = _normalize_division(st.get("division"), fb_d)
        # Keep Eastern divisions (Atlantic, Metropolitan) paired, then Western (Central, Pacific)
        c_ord = 0 if c_name == "Eastern" else 1
        return (c_ord, d_name, -sum(_standings_rank(t_top)), t_top)

    ordered = sorted(cur_series, key=_div_sort_key)

    round_trees: list[dict[str, float]] = []
    alive_teams: set[str] = set()
    for s in ordered:
        t1, t2 = s["top_seed"], s["bottom_seed"]
        w1, w2 = int(s.get("top_wins", 0)), int(s.get("bottom_wins", 0))
        if t1 not in team_idx or t2 not in team_idx:
            continue
        i1, i2 = team_idx[t1], team_idx[t2]
        p_t1 = series_prob_22111(float(p_home_mat[i1, i2]), float(1.0 - p_home_mat[i2, i1]), w1, w2)
        round_trees.append({t1: p_t1, t2: 1.0 - p_t1})
        alive_teams.update((t1, t2))

    if not round_trees:
        return None

    def _merge_trees(b1: dict[str, float], b2: dict[str, float]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for ta, pa in b1.items():
            for tb, pb in b2.items():
                w_prob = pa * pb
                if _standings_rank(ta) >= _standings_rank(tb):
                    p_a_win = float(series_mat[team_idx[ta], team_idx[tb]])
                else:
                    p_a_win = float(1.0 - series_mat[team_idx[tb], team_idx[ta]])
                merged[ta] = merged.get(ta, 0.0) + w_prob * p_a_win
                merged[tb] = merged.get(tb, 0.0) + w_prob * (1.0 - p_a_win)
        return merged

    trees = list(round_trees)
    conf_final_probs: dict[str, float] = {}
    cup_final_probs: dict[str, float] = {}

    if min_round == 3 and len(trees) == 2:
        conf_final_probs = {t: 1.0 for tr in trees for t in tr}
        cup_final_probs = {t: p for tr in trees for t, p in tr.items()}
    elif min_round == 4 and len(trees) == 1:
        cup_final_probs = {t: 1.0 for t in trees[0]}

    while len(trees) > 1:
        if len(trees) == 4:
            conf_final_probs = {t: p for tr in trees for t, p in tr.items()}
        next_trees = []
        for i in range(0, len(trees), 2):
            if i + 1 < len(trees):
                next_trees.append(_merge_trees(trees[i], trees[i + 1]))
            else:
                next_trees.append(trees[i])
        trees = next_trees
        if len(trees) == 2:
            cup_final_probs = {t: p for tr in trees for t, p in tr.items()}

    final_probs = trees[0]
    results: dict[str, dict[str, float]] = {}
    for t in teams:
        st = standings.get(t, {})
        in_play = t in alive_teams
        results[t] = {
            "proj_points": float(st.get("points", 0)),
            "make_playoffs_prob": 1.0 if in_play else 0.0,
            "win_division_prob": 0.0,
            "make_conf_final_prob": float(conf_final_probs.get(t, 0.0)),
            "make_cup_final_prob": float(cup_final_probs.get(t, 0.0)),
            "win_cup_prob": float(final_probs.get(t, 0.0)),
        }
    return results


def _solve_mutually_exclusive_kelly(
    probs: list[float],
    decimal_odds: list[float],
    *,
    kelly_fraction: float = 0.125,
    max_team_pct: float = 0.025,
    max_total_pct: float = 0.08,
) -> list[float]:
    """Optimize simultaneous fractional Kelly stakes over mutually exclusive futures."""
    k = len(probs)
    if k == 0:
        return []

    p_arr = np.asarray(probs, dtype=float)
    b_arr = np.asarray([o - 1.0 for o in decimal_odds], dtype=float)
    p_lose = max(0.0, 1.0 - float(np.sum(p_arr)))

    naive_full = np.maximum(0.0, (p_arr * b_arr - (1.0 - p_arr)) / b_arr)
    x0 = naive_full.copy()
    tot0 = float(np.sum(x0))
    if tot0 > 0.50:
        x0 *= 0.50 / tot0
    x0 = np.clip(x0 * 0.5, 0.0005, 0.25)

    def neg_log_growth(f: np.ndarray) -> float:
        sum_f = float(np.sum(f))
        w_win = np.maximum(1e-12, 1.0 - sum_f + f * (b_arr + 1.0))
        w_lose = max(1e-12, 1.0 - sum_f)
        growth = float(np.sum(p_arr * np.log(w_win)))
        if p_lose > 0.0:
            growth += p_lose * float(np.log(w_lose))
        return -growth

    res = scipy.optimize.minimize(
        neg_log_growth,
        x0,
        method="SLSQP",
        bounds=[(0.0, 0.90)] * k,
        constraints=[{"type": "ineq", "fun": lambda f: 0.90 - float(np.sum(f))}],
        options={"ftol": 1e-10, "maxiter": 300},
    )
    opt_f = np.maximum(0.0, res.x if res.success else x0)

    scaled = np.minimum(opt_f * float(kelly_fraction), float(max_team_pct))
    tot_scaled = float(np.sum(scaled))
    if tot_scaled > float(max_total_pct) and tot_scaled > 0.0:
        scaled *= float(max_total_pct) / tot_scaled

    return [float(x) for x in scaled]


def compute_futures_recommendations(
    sim_results: dict[str, dict[str, float]],
    standings: dict[str, dict[str, Any]],
    market_data: dict[str, Any],
    *,
    today_date: date | str,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.125,
    max_team_pct: float = 0.025,
    max_total_pct: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Blend simulated Cup odds with sharp market consensus and compute Portfolio Kelly stakes."""
    if isinstance(today_date, str):
        today = datetime.strptime(today_date[:10], "%Y-%m-%d").date()
    else:
        today = today_date

    cup_year = today.year + (1 if today.month >= 7 else 0)
    cup_date = date(cup_year, 6, 20)
    months_to_cup = max(0.5, (cup_date - today).days / 30.0)
    min_edge_pct = round(3.0 + 0.85 * months_to_cup, 1)

    gp_vals = [float(s.get("gp", 0)) for s in standings.values() if s.get("gp") is not None]
    avg_gp = float(np.mean(gp_vals)) if gp_vals else 0.0
    model_weight = round(float(np.clip(0.35 + 0.30 * (avg_gp / 82.0), 0.35, 0.65)), 3)

    best_odds_map: dict[str, float] = market_data.get("best_odds") or {}
    best_bookie_map: dict[str, str] = market_data.get("best_bookie") or {}
    market_probs_map: dict[str, float] = market_data.get("market_probs") or {}
    has_market_probs = len(market_probs_map) > 0

    teams = sorted(sim_results.keys())
    raw_blended: dict[str, float] = {}
    for t in teams:
        p_model = float(sim_results[t].get("win_cup_prob", 0.0))
        if has_market_probs:
            p_mkt = float(market_probs_map.get(t, 0.0))
            raw_blended[t] = model_weight * p_model + (1.0 - model_weight) * p_mkt
        else:
            raw_blended[t] = p_model

    blend_tot = sum(raw_blended.values())
    if blend_tot > 0.0:
        blended_probs = {t: v / blend_tot for t, v in raw_blended.items()}
    else:
        blended_probs = {t: 0.0 for t in teams}

    # Filter qualified +EV candidates clearing the time-adjusted edge hurdle
    cand_teams: list[str] = []
    cand_probs: list[float] = []
    cand_odds: list[float] = []
    for t in teams:
        b_prob = blended_probs[t]
        b_odd = float(best_odds_map.get(t, 0.0) or 0.0)
        if b_odd > 1.0 and b_prob > 0.0:
            edge_pct = ((b_prob * b_odd) - 1.0) * 100.0
            if edge_pct >= min_edge_pct:
                cand_teams.append(t)
                cand_probs.append(b_prob)
                cand_odds.append(b_odd)

    kelly_weights = _solve_mutually_exclusive_kelly(
        cand_probs,
        cand_odds,
        kelly_fraction=kelly_fraction,
        max_team_pct=max_team_pct,
        max_total_pct=max_total_pct,
    )
    weight_by_team = {t: w for t, w in zip(cand_teams, kelly_weights)}

    rows: list[dict[str, Any]] = []
    for t in teams:
        sim = sim_results[t]
        st = standings.get(t, {})
        fb_conf, fb_div = NHL_DIVISIONS.get(t, ("Western", "Central"))
        conf = _normalize_conference(st.get("conference"), fb_conf)
        div = _normalize_division(st.get("division"), fb_div)

        p_model = float(sim.get("win_cup_prob", 0.0))
        p_mkt = float(market_probs_map[t]) if t in market_probs_map else None
        p_blend = float(blended_probs.get(t, p_model))
        b_odd = best_odds_map.get(t)
        bookie = best_bookie_map.get(t)

        edge_pct = round(((p_blend * float(b_odd)) - 1.0) * 100.0, 2) if b_odd and float(b_odd) > 1.0 else None
        stake_pct = float(weight_by_team.get(t, 0.0))
        rec_bet = round(stake_pct * float(bankroll), 2)
        fair_odds = round(1.0 / p_blend, 2) if p_blend > 1e-4 else None

        rows.append(
            {
                "team": t,
                "conference": conf,
                "division": div,
                "gp": int(st.get("gp", 0)),
                "points": int(st.get("points", 0)),
                "rw": int(st.get("rw", 0)),
                "proj_points": float(sim.get("proj_points", 0.0)),
                "make_playoffs_pct": round(float(sim.get("make_playoffs_prob", 0.0)) * 100.0, 1),
                "win_division_pct": round(float(sim.get("win_division_prob", 0.0)) * 100.0, 1),
                "make_conf_final_pct": round(float(sim.get("make_conf_final_prob", 0.0)) * 100.0, 1),
                "make_cup_final_pct": round(float(sim.get("make_cup_final_prob", 0.0)) * 100.0, 1),
                "model_cup_pct": round(p_model * 100.0, 2),
                "market_cup_pct": round(p_mkt * 100.0, 2) if p_mkt is not None else None,
                "blended_cup_pct": round(p_blend * 100.0, 2),
                "fair_odds": fair_odds,
                "best_odds": round(float(b_odd), 2) if b_odd is not None else None,
                "best_bookie": bookie,
                "edge_pct": edge_pct,
                "kelly_pct": round(stake_pct * 100.0, 2),
                "rec_bet": rec_bet,
                "qualified": t in weight_by_team and rec_bet > 0.0,
            }
        )

    rows.sort(key=lambda r: (r["blended_cup_pct"], r["model_cup_pct"], r["proj_points"]), reverse=True)
    for idx, r in enumerate(rows, 1):
        r["rank"] = idx

    recommended_bets = [
        {
            "team": r["team"],
            "blended_cup_pct": r["blended_cup_pct"],
            "fair_odds": r["fair_odds"],
            "best_odds": r["best_odds"],
            "best_bookie": r["best_bookie"],
            "edge_pct": r["edge_pct"],
            "kelly_pct": r["kelly_pct"],
            "rec_bet": r["rec_bet"],
        }
        for r in rows
        if r["qualified"]
    ]
    recommended_bets.sort(key=lambda b: b["rec_bet"], reverse=True)

    summary = {
        "avg_gp": round(avg_gp, 1),
        "model_weight": model_weight,
        "market_weight": round(1.0 - model_weight, 3) if has_market_probs else 0.0,
        "months_to_cup": round(months_to_cup, 2),
        "min_edge_pct": min_edge_pct,
        "kelly_fraction": kelly_fraction,
        "max_team_pct": round(max_team_pct * 100.0, 2),
        "max_total_pct": round(max_total_pct * 100.0, 2),
        "bankroll": round(float(bankroll), 2),
        "market_overround_pct": market_data.get("market_overround_pct"),
        "devig_source": market_data.get("devig_source", "none"),
        "recommended_count": len(recommended_bets),
        "total_rec_stake": round(sum(b["rec_bet"] for b in recommended_bets), 2),
        "recommended_bets": recommended_bets,
    }
    return rows, summary


def run_futures_pipeline(
    *,
    data_dir: str = "./data",
    models_dir: str = "./models",
    docs_dir: str = "./docs",
    skip_network: bool = False,
    odds_api_key: str | None = None,
    n_sims: int = 10000,
    bankroll: float = 1000.0,
) -> dict[str, Any]:
    """Weekly training + Monte Carlo season/playoff simulation + Stanley Cup +EV futures export."""
    archive_dir = os.path.join(data_dir, "moneypuck")
    logs_dir = os.path.join(data_dir, "nhl_game_logs")
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    context = get_nhl_season_context()
    session = None if skip_network else create_http_session()
    odds_key = (odds_api_key if odds_api_key is not None else os.environ.get("ODDS_API_KEY", "")).strip()

    print("=== HOCKEYPICKS AI — WEEKLY STANLEY CUP FUTURES PIPELINE ===")
    print(f"Date (ET): {context['today_str']} | Season: {context['nhl_season_label']} ({context['nhl_season']})")

    if skip_network:
        available = [
            y
            for y in sorted(context["historical_mp_years"], reverse=True)
            if os.path.exists(os.path.join(archive_dir, f"{y}_regular_skaters.csv"))
        ]
        active_mp_year = available[0] if available else context["mp_year"]
    else:
        sync_stats = sync_moneypuck_data(archive_dir, context, session=session)
        active_mp_year = int(sync_stats["active_mp_year"])

    max_age = 1e9 if skip_network else 4.0
    df_master_logs, _, log_stats = sync_game_logs(
        logs_dir, context, session=session, max_active_age_hours=max_age, allow_network=not skip_network
    )
    latest_team_context = extract_latest_team_form(df_master_logs)

    df_team_hist, df_stars, df_goalies = build_team_star_goalie_history(
        archive_dir, context["historical_mp_years"]
    )
    futures_bundle = train_futures_game_model(df_team_hist, df_stars, df_goalies, df_master_logs)
    save_pairwise_bundle(futures_bundle, os.path.join(models_dir, "futures_game_bundle.joblib"))
    print(
        f"Futures Pairwise Model -> {futures_bundle['metrics']['games_trained']} games | "
        f"Features: {len(futures_bundle['feature_names'])} | Optimal T={futures_bundle['optimal_T']} | "
        f"OOF LogLoss: {futures_bundle['metrics']['oof_log_loss_calibrated']} | AUC={futures_bundle['metrics']['oof_auc']:.4f}"
    )

    playoff_path = os.path.join(archive_dir, f"{active_mp_year}_playoffs_teams.csv")
    playoff_df = pd.read_csv(playoff_path) if os.path.exists(playoff_path) and os.path.getsize(playoff_path) > 0 else None
    profiles, _ = build_live_team_profiles(
        df_team_hist=df_team_hist,
        df_stars=df_stars,
        df_goalies=df_goalies,
        active_mp_year=active_mp_year,
        latest_team_context=latest_team_context,
        base_features=futures_bundle["base_features"],
        playoff_teams_df=playoff_df,
    )
    active_goalies = load_active_goalie_lookup(archive_dir, active_mp_year)
    _rankings, _ = compute_power_rankings_and_defense(profiles, futures_bundle)

    if skip_network or session is None:
        standings = {
            t: {
                "team": t,
                "conference": conf,
                "division": div,
                "gp": int(profiles.get(t, {}).get("games_played", 0) or 0),
                "wins": 0,
                "losses": 0,
                "ot_losses": 0,
                "points": 0,
                "rw": 0,
                "row": 0,
            }
            for t, (conf, div) in NHL_DIVISIONS.items()
        }
        ot_rate = 0.23
        remaining_games: list[dict[str, Any]] = []
        active_series: list[dict[str, Any]] = []
        odds_meta: dict[str, Any] = {}
        market_data = {"api_ok": False, "best_odds": {}, "best_bookie": {}, "market_probs": {}, "market_overround_pct": None, "devig_source": "none"}
    else:
        standings, ot_rate = fetch_nhl_standings(session)
        teams_list = sorted(set(NHL_DIVISIONS.keys()) | set(standings.keys()))
        remaining_games, active_series = fetch_remaining_regular_schedule(context["nhl_season"], teams_list, session)
        odds_meta = {}
        market_data = fetch_cup_futures_odds(odds_key, session, meta_out=odds_meta)

    sim_mode = "monte_carlo_season"
    sim_results: dict[str, dict[str, float]] | None = None
    if not remaining_games and active_series:
        sim_results = compute_exact_playoff_cup_probs(active_series, standings, profiles, futures_bundle)
        if sim_results is not None:
            sim_mode = "exact_playoff_bracket"
    if sim_results is None:
        sim_results = simulate_season_and_cup(
            standings,
            remaining_games,
            profiles,
            futures_bundle,
            ot_rate=ot_rate,
            n_sims=n_sims,
        )

    rows, summary = compute_futures_recommendations(
        sim_results,
        standings,
        market_data,
        today_date=context["today_str"],
        bankroll=bankroll,
    )

    now_utc = datetime.now(timezone.utc)
    now_et = context["now_et"]
    payload = {
        "meta": {
            "status": "live",
            "updated_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "updated_et": now_et.strftime("%Y-%m-%d %I:%M %p ET"),
            "today_date": context["today_str"],
            "nhl_season": context["nhl_season"],
            "nhl_season_label": context["nhl_season_label"],
            "sim_mode": sim_mode,
            "n_sims": int(n_sims),
            "remaining_regular_games": len(remaining_games),
            "active_playoff_series_count": len(active_series),
            "empirical_ot_rate": ot_rate,
            "odds_enabled": bool(market_data.get("api_ok")),
            "odds_api": odds_meta,
            "model_metrics": futures_bundle.get("metrics", {}),
            "model_features": futures_bundle.get("feature_names", []),
            "game_log_rows": int(log_stats.get("total_rows", 0)),
        },
        "summary": summary,
        "teams": rows,
        "active_goalies_count": len(active_goalies),
    }

    out_path = os.path.join(docs_dir, "cup_futures.json")
    write_json_report(payload, out_path)
    print(
        f"Futures Simulation ({sim_mode}, {n_sims} trials, {len(remaining_games)} reg games left, OT rate={ot_rate:.1%}) -> "
        f"Top Favorite: {rows[0]['team']} ({rows[0]['blended_cup_pct']:.1f}% Cup) | "
        f"+EV Bets (>=+{summary['min_edge_pct']}%): {summary['recommended_count']} "
        f"(${summary['total_rec_stake']:.2f} total)"
    )
    if market_data.get("api_ok"):
        print(
            f"Futures Odds API -> {odds_meta.get('quoted_teams', 0)} teams ({summary['devig_source']} devig, "
            f"vig={summary['market_overround_pct']}%) | cost={odds_meta.get('requests_last')}, "
            f"used={odds_meta.get('requests_used')}, remaining={odds_meta.get('requests_remaining')}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train & evaluate weekly Stanley Cup futures model.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--models-dir", default="./models")
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args(argv)

    run_futures_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        docs_dir=args.docs_dir,
        skip_network=args.skip_network,
        n_sims=args.n_sims,
        bankroll=args.bankroll,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
