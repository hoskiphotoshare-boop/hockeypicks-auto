"""Live NHL injury report (ESPN + CBS fallback) and starting goalie (Daily Faceoff) ingestion."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

from src.data_sync import (
    CHROME_HEADERS,
    TEAM_FULL_TO_ABBREV,
    PlayerMatcher,
    _fetch_with_retries,
    clean_name,
    create_http_session,
    normalize_team_abbrev,
)

CBS_TEAM_CITY_TO_ABBREV = {
    "anaheim": "ANA",
    "boston": "BOS",
    "buffalo": "BUF",
    "calgary": "CGY",
    "carolina": "CAR",
    "chicago": "CHI",
    "colorado": "COL",
    "columbus": "CBJ",
    "dallas": "DAL",
    "detroit": "DET",
    "edmonton": "EDM",
    "florida": "FLA",
    "los angeles": "LAK",
    "minnesota": "MIN",
    "montreal": "MTL",
    "nashville": "NSH",
    "new jersey": "NJD",
    "n.y. islanders": "NYI",
    "ny islanders": "NYI",
    "new york islanders": "NYI",
    "n.y. rangers": "NYR",
    "ny rangers": "NYR",
    "new york rangers": "NYR",
    "ottawa": "OTT",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "san jose": "SJS",
    "seattle": "SEA",
    "st. louis": "STL",
    "st louis": "STL",
    "tampa bay": "TBL",
    "toronto": "TOR",
    "utah": "UTA",
    "vancouver": "VAN",
    "vegas": "VGK",
    "washington": "WSH",
    "winnipeg": "WPG",
}

_OUT_STATUS_TOKENS = ("injured reserve", "ltir", "ir-lt", "ir-nr", "out", "suspend")
_DTD_STATUS_TOKENS = ("day-to-day", "day to day", "gtd", "game-time", "questionable", "probable")


def classify_injury_status(status: Any, type_abbr: Any = "") -> tuple[str, bool, bool]:
    """Normalize injury severity into (short_label, is_out, is_dtd)."""
    raw = str(status or "").strip()
    abbr = str(type_abbr or "").strip().upper()
    low = raw.lower()

    if abbr in ("IR", "IR-LT", "IR-NR", "LTIR", "O", "OUT", "SUSP") or any(tok in low for tok in _OUT_STATUS_TOKENS):
        label = "IR" if ("reserve" in low or abbr.startswith("IR") or "LTIR" in abbr) else (
            "Susp" if "suspend" in low or abbr == "SUSP" else "Out"
        )
        return label, True, False

    if abbr in ("DD", "DTD", "GTD") or any(tok in low for tok in _DTD_STATUS_TOKENS):
        return "DTD", False, True

    return (raw[:12] if raw else "Injured"), False, False


def _parse_espn_injuries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tblock in payload.get("injuries") or []:
        t_display = str(tblock.get("displayName") or "").strip()
        block_team = TEAM_FULL_TO_ABBREV.get(t_display, "UNK")
        for item in tblock.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            ath_team = normalize_team_abbrev((athlete.get("team") or {}).get("abbreviation"))
            team = block_team if block_team != "UNK" else ath_team
            pos = str((athlete.get("position") or {}).get("abbreviation") or "").strip()
            type_obj = item.get("type") or {}
            status_label, is_out, is_dtd = classify_injury_status(
                item.get("status") or type_obj.get("description"),
                type_obj.get("abbreviation"),
            )
            records.append(
                {
                    "player": name,
                    "name_clean": clean_name(name),
                    "team": team,
                    "position": pos,
                    "status": status_label,
                    "raw_status": str(item.get("status") or ""),
                    "is_out": is_out,
                    "is_dtd": is_dtd,
                    "note": str(item.get("shortComment") or ""),
                }
            )
    return records


def _resolve_cbs_team_header(text: str) -> str:
    low = clean_name(text)
    if low in CBS_TEAM_CITY_TO_ABBREV:
        return CBS_TEAM_CITY_TO_ABBREV[low]
    for full_name, abbr in TEAM_FULL_TO_ABBREV.items():
        if full_name.lower() == low:
            return abbr
    return normalize_team_abbrev(text)


def _parse_cbs_injuries(html_text: str) -> list[dict[str, Any]]:
    if BeautifulSoup is None or not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    records: list[dict[str, Any]] = []
    for wrapper in soup.select(".TableBaseWrapper"):
        hdr = wrapper.select_one(".TeamLogoNameLockup-name") or wrapper.select_one("h4, .TeamName, a")
        team = _resolve_cbs_team_header(hdr.get_text(" ", strip=True) if hdr else "")
        for tr in wrapper.select("tbody tr"):
            tds = tr.select("td")
            if len(tds) < 5:
                continue
            long_el = tds[0].select_one(".CellPlayerName--long")
            raw_name = (long_el or tds[0]).get_text(" ", strip=True)
            pos = tds[1].get_text(" ", strip=True)
            inj_type = tds[3].get_text(" ", strip=True)
            raw_status = tds[4].get_text(" ", strip=True)
            if not raw_name:
                continue
            status_label, is_out, is_dtd = classify_injury_status(raw_status)
            records.append(
                {
                    "player": raw_name,
                    "name_clean": clean_name(raw_name),
                    "team": team,
                    "position": pos,
                    "status": status_label,
                    "raw_status": raw_status,
                    "is_out": is_out,
                    "is_dtd": is_dtd,
                    "note": f"{inj_type}: {raw_status}".strip(": "),
                }
            )
    return records


def fetch_nhl_injuries(
    session: Any | None = None,
    *,
    allow_network: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch league-wide NHL injuries from ESPN JSON API with CBS Sports HTML fallback."""
    if not allow_network:
        return [], "skipped_offline"

    http = session or create_http_session()
    espn_url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
    res = _fetch_with_retries(http, espn_url, headers=CHROME_HEADERS, timeout=12, attempts=2)
    if res is not None and res.status_code == 200:
        try:
            records = _parse_espn_injuries(res.json())
            if records:
                return records, "espn"
        except Exception:
            pass

    cbs_url = "https://www.cbssports.com/nhl/injuries/"
    cres = _fetch_with_retries(http, cbs_url, headers=CHROME_HEADERS, timeout=12, attempts=2)
    if cres is not None and cres.status_code == 200:
        try:
            records = _parse_cbs_injuries(cres.text)
            if records:
                return records, "cbssports"
        except Exception:
            pass

    return [], "unavailable"


def _parse_dailyfaceoff_goalies(html_text: str, today_str: str) -> dict[str, dict[str, Any]]:
    if not html_text:
        return {}
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html_text, re.S)
    if not match:
        return {}
    try:
        nd = json.loads(match.group(1))
    except Exception:
        return {}

    pp = (nd.get("props") or {}).get("pageProps") or {}
    games = pp.get("data") or []
    if not isinstance(games, list):
        return {}

    goalies: dict[str, dict[str, Any]] = {}
    date_matches = [g for g in games if isinstance(g, dict) and str(g.get("date") or "") == today_str]
    target_games = date_matches if date_matches else (
        games if not today_str or str(pp.get("date") or "") == today_str else []
    )

    for g in target_games:
        for prefix in ("away", "home"):
            team = normalize_team_abbrev(g.get(f"{prefix}TeamAbbreviation"))
            g_name = str(g.get(f"{prefix}GoalieName") or "").strip()
            if not team or team == "UNK" or not g_name:
                continue
            conf_raw = g.get(f"{prefix}NewsStrengthName")
            conf = str(conf_raw).strip() if conf_raw else "Expected"
            try:
                sv_pct = float(g.get(f"{prefix}GoalieSavePercentage") or 0.0)
            except (TypeError, ValueError):
                sv_pct = 0.0
            try:
                gaa = float(g.get(f"{prefix}GoalieGoalsAgainstAvg") or 0.0)
            except (TypeError, ValueError):
                gaa = 0.0
            goalies[team] = {
                "team": team,
                "goalie_name": g_name,
                "name_clean": clean_name(g_name),
                "confirmation": conf,
                "sv_pct": sv_pct if 0.70 <= sv_pct <= 0.99 else None,
                "gaa": round(gaa, 2) if gaa > 0 else None,
                "details": str(g.get(f"{prefix}NewsDetails") or ""),
            }
    return goalies


def fetch_starting_goalies(
    today_str: str,
    session: Any | None = None,
    *,
    allow_network: bool = True,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Fetch projected/confirmed NHL starting goalies for today from Daily Faceoff."""
    if not allow_network:
        return {}, "skipped_offline"

    http = session or create_http_session()
    url = "https://www.dailyfaceoff.com/starting-goalies"
    res = _fetch_with_retries(http, url, headers=CHROME_HEADERS, timeout=12, attempts=2)
    if res is not None and res.status_code == 200:
        goalies = _parse_dailyfaceoff_goalies(res.text, today_str=today_str)
        if goalies:
            return goalies, "dailyfaceoff"
        return {}, "empty_slate"
    return {}, "unavailable"


def build_injury_matcher(injury_records: list[dict[str, Any]]) -> PlayerMatcher:
    """Index injury records in a PlayerMatcher for accent/nickname-safe skater lookups."""
    entries = [
        {
            "player_id": 0,
            "name": str(rec.get("player", "")),
            "name_clean": str(rec.get("name_clean") or clean_name(rec.get("player", ""))),
            "team": normalize_team_abbrev(rec.get("team")),
            "position": str(rec.get("position", "")),
            "payload": rec,
        }
        for rec in (injury_records or [])
        if rec.get("player")
    ]
    return PlayerMatcher(entries)


def apply_live_availability_to_profiles(
    *,
    profiles: dict[str, dict[str, Any]],
    starting_goalies: dict[str, dict[str, Any]],
    injury_records: list[dict[str, Any]],
    active_goalies_by_team: dict[str, list[dict[str, Any]]] | None = None,
    skater_baseline_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Adjust live team matchup profiles for confirmed/expected starters and Out/IR star absences.

    Returns per-team UI summary metadata (starter goalie, status, key injured skaters).
    """
    injury_matcher = build_injury_matcher(injury_records)
    team_summary: dict[str, dict[str, Any]] = {}

    injuries_by_team: dict[str, list[dict[str, Any]]] = {}
    for rec in injury_records or []:
        t = normalize_team_abbrev(rec.get("team"))
        if t != "UNK":
            injuries_by_team.setdefault(t, []).append(rec)

    # Pre-index team Top-3 scorer baseline share if skater pool is supplied
    skater_top_by_team: dict[str, list[dict[str, Any]]] = {}
    if skater_baseline_df is not None and not skater_baseline_df.empty and "team" in skater_baseline_df.columns:
        work = skater_baseline_df.copy()
        work["team"] = work["team"].apply(normalize_team_abbrev)
        score_col = "base_pred" if "base_pred" in work.columns else (
            "base_pred_regular" if "base_pred_regular" in work.columns else None
        )
        if score_col:
            work["_rank_val"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
            for t_key, grp in work.groupby("team"):
                top_rows = grp.sort_values("_rank_val", ascending=False).head(6)
                skater_top_by_team[str(t_key)] = [
                    {
                        "name": str(r.get("name", "")),
                        "position": str(r.get("position", "")),
                        "val": max(0.0, float(r.get("_rank_val", 0.0))),
                    }
                    for _, r in top_rows.iterrows()
                ]

    all_teams = sorted(set(profiles.keys()) | set(starting_goalies.keys()) | set(injuries_by_team.keys()))
    for team in all_teams:
        g_info = starting_goalies.get(team)
        t_inj = injuries_by_team.get(team, [])
        out_skaters = [r for r in t_inj if r.get("is_out") and str(r.get("position", "")).upper() != "G"]
        dtd_skaters = [r for r in t_inj if r.get("is_dtd") and str(r.get("position", "")).upper() != "G"]

        starter_record = None
        if team in profiles:
            st = profiles[team]
            # 1. Starting Goalie Adjustment
            if g_info:
                g_name = g_info["goalie_name"]
                mp_goalies = (active_goalies_by_team or {}).get(team, [])
                matched_mp = None
                if mp_goalies:
                    gm = PlayerMatcher(
                        [
                            {
                                "player_id": int(mg.get("player_id", 0) or 0),
                                "name": str(mg.get("name", "")),
                                "team": team,
                                "position": "G",
                                "payload": mg,
                            }
                            for mg in mp_goalies
                        ]
                    )
                    matched_mp = gm.match(
                        name=g_name,
                        team=team,
                        position="G",
                        allow_initial_fallback=True,
                        require_same_team=True,
                        require_compatible_pos=True,
                    )

                base_sv = float(st.get("Starter_SavePct", 0.900))
                starter_sv = g_info.get("sv_pct")
                shots_faced = 0.0
                if matched_mp is not None and float(matched_mp.get("ongoal", 0.0)) > 0:
                    shots_faced = float(matched_mp["ongoal"])
                    mp_sv = float(matched_mp["save_pct"])
                    # Blend MoneyPuck season starter SV% with starter's own SV% shrunk toward 0.902
                    if starter_sv is None or shots_faced >= 60.0:
                        starter_sv = mp_sv
                    w_starter = shots_faced / (shots_faced + 180.0)
                    st["Starter_SavePct"] = round((0.902 * (1.0 - w_starter)) + (starter_sv * w_starter), 4)
                    # Blend Team_GSAx_per_game toward starter's personal GSAx/GP
                    mp_gp = max(1.0, float(matched_mp.get("games_played", 1.0)))
                    starter_gsax_pg = float(matched_mp.get("gsax", 0.0)) / mp_gp
                    st["Team_GSAx_per_game"] = round(
                        (0.5 * float(st.get("Team_GSAx_per_game", 0.0))) + (0.5 * starter_gsax_pg),
                        4,
                    )
                elif starter_sv is not None:
                    # Backup/rookie with DFO stats only -> lightly nudge toward confirmed SV%
                    w_dfo = 0.25 if g_info.get("confirmation", "").lower() == "confirmed" else 0.15
                    st["Starter_SavePct"] = round((base_sv * (1.0 - w_dfo)) + (starter_sv * w_dfo), 4)

                starter_record = {
                    "name": g_name,
                    "confirmation": g_info.get("confirmation", "Expected"),
                    "sv_pct": round(float(g_info.get("sv_pct") or st.get("Starter_SavePct", 0.900)), 3),
                }

            # 2. Injured Top-3 Star-Power Damping (healthy Top-3 vs season Top-3, clamped [0.65, 1.0])
            top_sk = skater_top_by_team.get(team, [])
            if len(top_sk) >= 3 and out_skaters:
                orig_top3_sum = sum(x["val"] for x in top_sk[:3])
                healthy_sk = [
                    x
                    for x in top_sk
                    if not (
                        (
                            inj := injury_matcher.match(
                                name=x["name"],
                                team=team,
                                position=x["position"],
                                require_same_team=True,
                                require_compatible_pos=True,
                            )
                        )
                        and inj.get("is_out")
                    )
                ]
                if orig_top3_sum > 1e-6 and len(healthy_sk) < len(top_sk):
                    new_top3_sum = sum(x["val"] for x in healthy_sk[:3])
                    ratio = float(min(1.0, max(0.65, new_top3_sum / orig_top3_sum)))
                    if ratio < 0.999:
                        # Blend 50% of missing star drop with team baseline (teammates step into PP1/top line)
                        eff_scale = 0.5 + 0.5 * ratio
                        st["Top3_Points_per_game"] = float(st.get("Top3_Points_per_game", 0.0)) * eff_scale
                        st["Top3_xGoals_per_game"] = float(st.get("Top3_xGoals_per_game", 0.0)) * eff_scale

        team_summary[team] = {
            "goalie": starter_record,
            "out_skaters": [
                {"player": r["player"], "position": r["position"], "status": r["status"]}
                for r in out_skaters
            ],
            "dtd_skaters": [
                {"player": r["player"], "position": r["position"], "status": r["status"]}
                for r in dtd_skaters
            ],
        }

    return team_summary
