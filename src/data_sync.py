"""Data synchronization for MoneyPuck CSVs, NHL API logs/schedules, and Tim Hortons picks."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz
import requests

try:
    import cloudscraper
except ImportError:  # pragma: no cover - optional fallback in offline minimal test envs
    cloudscraper = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


ET_TZ = pytz.timezone("America/New_York")

CHROME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/json,text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TEAM_ABBREV_MAP = {
    "S.J": "SJS",
    "L.A": "LAK",
    "N.J": "NJD",
    "T.B": "TBL",
    "ANH": "ANA",
    "PHX": "ARI",
}

NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

TEAM_FULL_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Utah": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}


def normalize_team_abbrev(team: Any) -> str:
    """Normalize MoneyPuck/NHL team abbreviations into standard 3-letter codes."""
    if not isinstance(team, str):
        return "UNK"
    cleaned = team.strip().upper()
    return TEAM_ABBREV_MAP.get(cleaned, cleaned)


def get_nhl_season_context(now_et: datetime | None = None) -> dict[str, Any]:
    """Compute active NHL and MoneyPuck season identifiers from Eastern Time."""
    if now_et is None:
        now_et = datetime.now(ET_TZ)
    elif now_et.tzinfo is None:
        now_et = ET_TZ.localize(now_et)

    start_year = now_et.year if now_et.month >= 8 else now_et.year - 1
    nhl_season = f"{start_year}{start_year + 1}"
    historical_mp_years = list(range(2021, start_year + 1))
    historical_nhl_seasons = [f"{y}{y + 1}" for y in range(2021, start_year + 1)]

    return {
        "now_et": now_et,
        "today_str": now_et.strftime("%Y-%m-%d"),
        "tomorrow_str": (now_et + timedelta(days=1)).strftime("%Y-%m-%d"),
        "start_year": start_year,
        "nhl_season": nhl_season,
        "nhl_season_label": f"{start_year}-{str(start_year + 1)[-2:]}",
        "mp_year": start_year,
        "historical_mp_years": historical_mp_years,
        "historical_nhl_seasons": historical_nhl_seasons,
    }


def create_http_session() -> Any:
    """Create an HTTP session with Chrome headers and optional Cloudflare bypass."""
    if cloudscraper is not None:
        try:
            scraper = cloudscraper.create_scraper()
            scraper.headers.update(CHROME_HEADERS)
            return scraper
        except Exception:
            pass
    session = requests.Session()
    session.headers.update(CHROME_HEADERS)
    return session


def _fetch_with_retries(
    session: Any,
    url: str,
    timeout: int = 20,
    attempts: int = 3,
    headers: dict[str, str] | None = None,
) -> requests.Response | None:
    """Perform a resilient GET request with brief exponential backoff."""
    for attempt in range(attempts):
        try:
            res = session.get(url, headers=headers, timeout=timeout)
            if res.status_code in (200, 404):
                return res
            if res.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            return res
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def _build_moneypuck_targets(years: list[int]) -> list[tuple[str, str, int]]:
    """Build deterministic list of MoneyPuck season summary CSV targets."""
    targets: list[tuple[str, str, int]] = []
    for year in years:
        for split in ("regular", "playoffs"):
            for entity in ("skaters", "teams", "goalies"):
                filename = f"{year}_{split}_{entity}.csv"
                url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/{split}/{entity}.csv"
                targets.append((filename, url, year))
    return targets


def _discover_extra_moneypuck_urls(session: Any) -> dict[str, str]:
    """Optionally scour MoneyPuck data page for mirrored seasonSummary URLs."""
    discovered: dict[str, str] = {}
    if BeautifulSoup is None:
        return discovered
    res = _fetch_with_retries(session, "https://moneypuck.com/data.htm", timeout=15, attempts=2)
    if res is None or res.status_code != 200:
        return discovered
    soup = BeautifulSoup(res.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.endswith(".csv"):
            continue
        if href.startswith("/"):
            full_url = f"https://moneypuck.com{href}"
        elif not href.startswith("http"):
            full_url = f"https://moneypuck.com/{href}"
        else:
            full_url = href
        if "seasonSummary" in full_url:
            parts = full_url.rstrip("/").split("/")
            if len(parts) >= 3:
                filename = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
                discovered[filename] = full_url
    return discovered


def sync_moneypuck_data(
    archive_dir: str,
    context: dict[str, Any],
    session: Any | None = None,
    max_active_age_hours: float = 6.0,
) -> dict[str, Any]:
    """Download/cache MoneyPuck skaters, teams, and goalies CSVs cleanly."""
    os.makedirs(archive_dir, exist_ok=True)
    http = session or create_http_session()
    mp_year = context["mp_year"]
    targets = _build_moneypuck_targets(context["historical_mp_years"])
    extra_urls = _discover_extra_moneypuck_urls(http)

    downloaded = 0
    cached = 0
    unavailable_optional = 0
    errors = 0

    for filename, url, year in targets:
        url = extra_urls.get(filename, url)
        filepath = os.path.join(archive_dir, filename)
        is_active = year >= mp_year - 1 if context["now_et"].month <= 9 else year >= mp_year

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            if not is_active:
                cached += 1
                continue
            age_hours = (time.time() - os.path.getmtime(filepath)) / 3600.0
            if age_hours < max_active_age_hours:
                cached += 1
                continue

        res = _fetch_with_retries(http, url, timeout=25, attempts=3)
        if res is not None and res.status_code == 200 and len(res.content) > 200:
            with open(filepath, "wb") as f:
                f.write(res.content)
            downloaded += 1
            time.sleep(0.35)
        elif res is not None and res.status_code == 404 and year == mp_year:
            unavailable_optional += 1
        elif os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            cached += 1
        else:
            if year < mp_year:
                errors += 1
            else:
                unavailable_optional += 1

    # Determine the latest regular season MoneyPuck year actually available on disk
    available_reg_years = []
    for y in sorted(context["historical_mp_years"], reverse=True):
        sk_path = os.path.join(archive_dir, f"{y}_regular_skaters.csv")
        tm_path = os.path.join(archive_dir, f"{y}_regular_teams.csv")
        if os.path.exists(sk_path) and os.path.exists(tm_path):
            available_reg_years.append(y)
    active_mp_year = available_reg_years[0] if available_reg_years else mp_year

    return {
        "downloaded": downloaded,
        "cached": cached,
        "unavailable_optional": unavailable_optional,
        "errors": errors,
        "active_mp_year": active_mp_year,
    }


def _compute_team_streaks(df_team_season: pd.DataFrame) -> pd.DataFrame:
    """Sort games chronologically and assign pre-game Last_Game_Won and Current_Streak."""
    if df_team_season.empty:
        return df_team_season
    df = df_team_season.sort_values("Date").reset_index(drop=True)
    streaks: list[int] = []
    last_won_list: list[int] = []
    current_streak = 0
    last_won = 0

    for won in df["Won"]:
        last_won_list.append(last_won)
        streaks.append(current_streak)
        last_won = int(won)
        if last_won == 1:
            current_streak = current_streak + 1 if current_streak > 0 else 1
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1

    df["Last_Game_Won"] = last_won_list
    df["Current_Streak"] = streaks
    return df


def _extract_post_game_streak_state(df_logs: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Compute each team's live entering-tonight streak after its most recent completed game."""
    context_map: dict[str, dict[str, int]] = {}
    if df_logs.empty:
        return context_map

    df_sorted = df_logs.sort_values("Date")
    for team, group in df_sorted.groupby("Team"):
        last_row = group.iloc[-1]
        last_won = int(last_row["Won"])
        prev_streak = int(last_row["Current_Streak"])
        if last_won == 1:
            next_streak = prev_streak + 1 if prev_streak > 0 else 1
        else:
            next_streak = prev_streak - 1 if prev_streak < 0 else -1
        context_map[str(team)] = {
            "Last_Game_Won": last_won,
            "Current_Streak": next_streak,
        }
    return context_map


def sync_game_logs(
    logs_dir: str,
    context: dict[str, Any],
    session: Any | None = None,
    max_active_age_hours: float = 6.0,
    allow_network: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, int]]:
    """Download/cache NHL API season schedules & compute leak-free team streaks."""
    os.makedirs(logs_dir, exist_ok=True)
    http = (session or create_http_session()) if allow_network else None
    current_season = context["nhl_season"]
    seasons = context["historical_nhl_seasons"]

    teams_with_ari = sorted(set(NHL_TEAMS) | {"ARI"})
    all_logs: list[pd.DataFrame] = []
    downloaded = 0
    cached = 0
    errors = 0

    for team in teams_with_ari:
        for season in seasons:
            if team == "UTA" and season < "20242025":
                continue
            if team == "ARI" and season >= "20242025":
                continue

            file_path = os.path.join(logs_dir, f"{team}_{season}_gamelog.csv")
            is_current = season == current_season
            use_cache = False

            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                if not is_current:
                    use_cache = True
                else:
                    age_hours = (time.time() - os.path.getmtime(file_path)) / 3600.0
                    if age_hours < max_active_age_hours:
                        use_cache = True

            df_team_season = pd.DataFrame()
            if use_cache:
                try:
                    df_team_season = pd.read_csv(file_path)
                    cached += 1
                except Exception:
                    df_team_season = pd.DataFrame()

            if df_team_season.empty and not use_cache and allow_network:
                url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
                res = _fetch_with_retries(http, url, timeout=12, attempts=2)
                if res is not None and res.status_code == 200:
                    games = []
                    for game in res.json().get("games", []):
                        if game.get("gameState") not in ("FINAL", "OFF", "CRIT"):
                            continue
                        game_type = game.get("gameType")
                        if game_type not in (2, 3):
                            continue
                        home_team = normalize_team_abbrev(game.get("homeTeam", {}).get("abbrev"))
                        away_team = normalize_team_abbrev(game.get("awayTeam", {}).get("abbrev"))
                        home_score = int(game.get("homeTeam", {}).get("score", 0) or 0)
                        away_score = int(game.get("awayTeam", {}).get("score", 0) or 0)

                        # Map historical ARI records under ARI cleanly while preserving matchup abbrevs
                        is_home = team == home_team
                        opponent = away_team if is_home else home_team
                        team_score = home_score if is_home else away_score
                        opp_score = away_score if is_home else home_score
                        won = 1 if team_score > opp_score else 0

                        games.append(
                            {
                                "Season": season,
                                "Team": team,
                                "Date": game.get("gameDate"),
                                "GameType": game_type,
                                "Opponent": opponent,
                                "Is_Home": 1 if is_home else 0,
                                "Team_Score": team_score,
                                "Opp_Score": opp_score,
                                "Won": won,
                            }
                        )
                    df_team_season = _compute_team_streaks(pd.DataFrame(games))
                    if not df_team_season.empty:
                        df_team_season.to_csv(file_path, index=False)
                        downloaded += 1
                elif os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    df_team_season = pd.read_csv(file_path)
                    cached += 1
                elif res is not None and res.status_code != 404:
                    errors += 1

            if not df_team_season.empty:
                df_team_season["Season"] = df_team_season["Season"].astype(str)
                all_logs.append(df_team_season)

    df_master = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    latest_context = _extract_post_game_streak_state(df_master)
    stats = {"downloaded": downloaded, "cached": cached, "errors": errors, "total_rows": len(df_master)}
    return df_master, latest_context, stats


def _format_start_et(utc_str: str | None) -> str:
    """Convert ISO-8601 UTC startTimeUTC into readable Eastern Time string."""
    if not utc_str:
        return "TBD"
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ET_TZ)
        return dt_et.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return "TBD"


def fetch_nhl_schedule(
    context: dict[str, Any],
    session: Any | None = None,
) -> dict[str, Any]:
    """Fetch today and tomorrow NHL games safely across list-typed gameWeek entries."""
    http = session or create_http_session()
    dates_to_check = [context["today_str"], context["tomorrow_str"]]
    games_out: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()

    for date_str in dates_to_check:
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        res = _fetch_with_retries(http, url, timeout=12, attempts=2)
        if res is None or res.status_code != 200:
            continue
        payload = res.json()
        for week in payload.get("gameWeek", []):
            if week.get("date") != date_str:
                continue
            for game in week.get("games", []):
                gid = game.get("id") or f"{date_str}-{game.get('awayTeam', {}).get('abbrev')}-{game.get('homeTeam', {}).get('abbrev')}"
                if gid in seen_ids:
                    continue
                seen_ids.add(gid)
                away = normalize_team_abbrev(game.get("awayTeam", {}).get("abbrev"))
                home = normalize_team_abbrev(game.get("homeTeam", {}).get("abbrev"))
                game_type = int(game.get("gameType") or 2)
                games_out.append(
                    {
                        "id": gid,
                        "date": date_str,
                        "is_today": date_str == context["today_str"],
                        "away": away,
                        "home": home,
                        "game_type": game_type,
                        "game_state": game.get("gameState", "FUT"),
                        "start_time_utc": game.get("startTimeUTC"),
                        "start_time_et": _format_start_et(game.get("startTimeUTC")),
                        "away_score": game.get("awayTeam", {}).get("score"),
                        "home_score": game.get("homeTeam", {}).get("score"),
                        "series_status": game.get("seriesStatus") or None,
                    }
                )

    today_games = [g for g in games_out if g["is_today"]]
    tomorrow_games = [g for g in games_out if not g["is_today"]]
    upcoming_today = [g for g in today_games if g.get("game_state") in ("FUT", "PRE")]
    slate_games = today_games if upcoming_today else (tomorrow_games or today_games or games_out)

    daily_matchups: dict[str, str] = {}
    for g in slate_games:
        daily_matchups[g["away"]] = g["home"]
        daily_matchups[g["home"]] = g["away"]

    # Determine active mode: Playoffs (3) if any scheduled playoff game or Apr-Jun series exist
    has_playoff_games = any(g["game_type"] == 3 for g in games_out)
    active_game_type = 3 if has_playoff_games else 2
    active_mode = "Playoffs" if active_game_type == 3 else "Regular Season"

    return {
        "games": games_out,
        "today_games": today_games,
        "daily_matchups": daily_matchups,
        "active_game_type": active_game_type,
        "active_mode": active_mode,
    }


def clean_name(name: Any) -> str:
    """Normalize accented/unicode player names to lowercase ASCII for matching."""
    import unicodedata

    if not isinstance(name, str):
        return ""
    cleaned = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def fetch_live_club_skaters(
    teams: set[str] | list[str],
    nhl_season: str,
    game_type: int,
    session: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch live NHL API skater GP and goals for the current season & gameType (2 or 3)."""
    http = session or create_http_session()
    live_stats: dict[str, dict[str, Any]] = {}

    for team in sorted(set(normalize_team_abbrev(t) for t in teams if t and t != "UNK")):
        url = f"https://api-web.nhle.com/v1/club-stats/{team}/{nhl_season}/{game_type}"
        res = _fetch_with_retries(http, url, timeout=8, attempts=2)
        if res is None or res.status_code != 200:
            continue
        for sk in res.json().get("skaters", []):
            first = sk.get("firstName", {}).get("default", "")
            last = sk.get("lastName", {}).get("default", "")
            full_name = f"{first} {last}".strip()
            key = clean_name(full_name)
            if key:
                live_stats[key] = {
                    "live_gp": int(sk.get("gamesPlayed", 0) or 0),
                    "live_goals": int(sk.get("goals", 0) or 0),
                    "live_points": int(sk.get("points", 0) or 0),
                    "live_shots": int(sk.get("shots", 0) or 0),
                    "team": team,
                }
    return live_stats


def fetch_tims_picks(session: Any | None = None) -> tuple[pd.DataFrame, str]:
    """Fetch active Tim Hortons NHL Hockey Challenge candidate groups."""
    http = session or create_http_session()
    headers = {
        "Referer": "https://hockeychallengehelper.com/",
        "Origin": "https://hockeychallengehelper.com",
        "Accept": "application/json, text/plain, */*",
    }
    url = "https://api.hockeychallengehelper.com/api/picks"
    res = _fetch_with_retries(http, url, headers=headers, timeout=15, attempts=2)
    if res is None:
        return pd.DataFrame(), "network_error"
    if res.status_code == 404:
        return pd.DataFrame(), "empty_404"
    if res.status_code != 200:
        return pd.DataFrame(), f"http_{res.status_code}"

    rows = []
    try:
        data = res.json()
    except Exception:
        return pd.DataFrame(), "invalid_json"

    for g_idx, group in enumerate(data.get("playerLists", []), 1):
        for p in group.get("players", []):
            first = p.get("firstName", "")
            last = p.get("lastName", "")
            full = f"{first} {last}".strip()
            rows.append(
                {
                    "group": g_idx,
                    "firstName": first,
                    "lastName": last,
                    "player_name": full,
                    "team": normalize_team_abbrev(p.get("team")),
                    "opponentTeam": normalize_team_abbrev(p.get("opponentTeam", "UNK")),
                    "position": p.get("position", ""),
                    "name_clean": clean_name(full),
                }
            )

    if not rows:
        return pd.DataFrame(), "empty_lists"
    return pd.DataFrame(rows), "ok"
