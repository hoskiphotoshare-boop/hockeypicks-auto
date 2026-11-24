"""Data synchronization for MoneyPuck CSVs, NHL API logs/schedules, and Tim Hortons picks."""

from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
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
    "SJ": "SJS",
    "L.A": "LAK",
    "LA": "LAK",
    "N.J": "NJD",
    "NJ": "NJD",
    "T.B": "TBL",
    "TB": "TBL",
    "ANH": "ANA",
    "PHX": "ARI",
    "UTAH": "UTA",
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
        "yesterday_str": (now_et - timedelta(days=1)).strftime("%Y-%m-%d"),
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
    seasons_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, int]]:
    """Download/cache NHL API season schedules & compute leak-free team streaks."""
    os.makedirs(logs_dir, exist_ok=True)
    http = (session or create_http_session()) if allow_network else None
    current_season = context["nhl_season"]
    seasons = seasons_filter if seasons_filter is not None else context["historical_nhl_seasons"]

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
    yesterday_str = context.get("yesterday_str")
    today_str = context["today_str"]
    tomorrow_str = context["tomorrow_str"]
    dates_to_check = [d for d in (yesterday_str, today_str, tomorrow_str) if d]
    games_out: list[dict[str, Any]] = []
    previous_games: list[dict[str, Any]] = []
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
                entry = {
                    "id": gid,
                    "date": date_str,
                    "is_today": date_str == today_str,
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
                if date_str == yesterday_str:
                    previous_games.append(entry)
                else:
                    games_out.append(entry)

    today_games = [g for g in games_out if g["is_today"]]
    tomorrow_games = [g for g in games_out if g.get("date") == tomorrow_str]
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
        "previous_games": previous_games,
        "daily_matchups": daily_matchups,
        "active_game_type": active_game_type,
        "active_mode": active_mode,
    }


def clean_name(name: Any) -> str:
    """Normalize accented/unicode player names to lowercase ASCII for matching."""
    if not isinstance(name, str):
        return ""
    cleaned = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


_FIRST_NAME_CANONICAL: dict[str, str] = {
    "nicholas": "nick",
    "nicolas": "nick",
    "nic": "nick",
    "mitchell": "mitch",
    "alexander": "alex",
    "alexei": "alexey",
    "matthew": "matt",
    "mathew": "matt",
    "joshua": "josh",
    "zachary": "zach",
    "zack": "zach",
    "zackary": "zach",
    "jacob": "jake",
    "joseph": "joe",
    "thomas": "tom",
    "tommy": "tom",
    "samuel": "sam",
    "sammy": "sam",
    "maxime": "max",
    "maxim": "max",
    "maxwell": "max",
    "christopher": "chris",
    "patrick": "pat",
    "yegor": "egor",
    "evgenii": "evgeny",
    "evgeni": "evgeny",
    "artyom": "artem",
    "matvey": "matvei",
    "jonathan": "jon",
    "jonny": "jon",
    "michael": "mike",
    "mikey": "mike",
    "daniel": "dan",
    "danny": "dan",
    "benjamin": "ben",
    "william": "will",
    "anthony": "tony",
    "cameron": "cam",
    "timothy": "tim",
    "timmy": "tim",
    "philip": "phil",
    "phillip": "phil",
    "philippe": "phil",
}

_MULTI_WORD_FIRST_PREFIXES: dict[str, str] = {
    "benoit olivier": "bo",
    "pierre olivier": "po",
    "john jason": "jj",
}


def _normalize_pos_group(position: Any) -> str:
    pos = str(position or "").strip().upper()
    if pos in ("D", "DEF", "DEFENSE"):
        return "D"
    if pos in ("G", "GOALIE"):
        return "G"
    if pos in ("F", "C", "L", "R", "LW", "RW", "W", "FOR", "FORWARD", "FORWARDS"):
        return "F"
    return ""


def _split_name_parts_with_last(name: Any) -> tuple[str, str, str, list[str]]:
    """Return (canonical_first, collapsed_last_alpha, canonical_full_key, raw_tokens)."""
    cleaned = clean_name(name)
    if not cleaned:
        return "", "", "", []
    # Collapse apostrophe prefixes (e.g. O'Connor -> oconnor, D'Astous -> dastous, L'Heureux -> lheureux, K'Andre -> kandre)
    cleaned_no_apo = re.sub(r"\b([a-z])['\u2019]([a-z])", r"\1\2", cleaned)
    alpha_spaced = re.sub(r"[.\u2019'\-]", " ", cleaned_no_apo)
    alpha_spaced = re.sub(r"[^a-z0-9 ]", "", alpha_spaced)
    tokens = re.sub(r"\s+", " ", alpha_spaced).strip().split()
    if not tokens:
        return "", "", "", []
    if len(tokens) >= 2 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
        tokens = [tokens[0] + tokens[1]] + tokens[2:]
    if len(tokens) >= 3 and f"{tokens[0]} {tokens[1]}" in _MULTI_WORD_FIRST_PREFIXES:
        tokens = [_MULTI_WORD_FIRST_PREFIXES[f"{tokens[0]} {tokens[1]}"]] + tokens[2:]
    first = _FIRST_NAME_CANONICAL.get(tokens[0], tokens[0])
    last = "".join(tokens[1:]) if len(tokens) > 1 else tokens[0]
    return first, last, f"{first} {last}".strip(), tokens


def parse_player_name_parts(name: Any) -> tuple[str, str, str]:
    """Return (canonical_first, collapsed_last_alpha, canonical_full_key)."""
    first, last, canon, _ = _split_name_parts_with_last(name)
    return first, last, canon


def _consonant_skeleton(text: str) -> str:
    return re.sub(r"[aeiouy]", "", text)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def accent_drop_match(
    q_first: str,
    q_last: str,
    c_first: str,
    c_last: str,
) -> bool:
    """Detect MoneyPuck dropped-accent spellings (e.g. Lafrenire vs Lafreniere, Slafkovsk vs Slafkovsky)."""
    if not q_first or not c_first or not q_last or not c_last:
        return False

    if q_first == c_first:
        first_ok = True
    else:
        sf, lf = (q_first, c_first) if len(q_first) <= len(c_first) else (c_first, q_first)
        first_ok = (
            len(sf) >= 2
            and (len(lf) - len(sf)) <= 2
            and _is_subsequence(sf, lf)
            and _consonant_skeleton(sf) == _consonant_skeleton(lf)
        )
    if not first_ok:
        return False

    if q_last == c_last:
        return True
    sl, ll = (q_last, c_last) if len(q_last) <= len(c_last) else (c_last, q_last)
    if len(sl) < 2 or (len(ll) - len(sl)) > 3:
        return False
    if not _is_subsequence(sl, ll):
        return False
    return _consonant_skeleton(sl) == _consonant_skeleton(ll) or SequenceMatcher(None, sl, ll).ratio() >= 0.80


_SURNAME_PARTICLES = {"st", "ste", "van", "von", "de", "del", "la", "le", "di", "mc", "mac", "saint"}


class PlayerMatcher:
    """Fast layered player lookup: playerId -> exact ASCII -> nickname/initial -> dropped-accent fuzzy."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._by_id: dict[int, Any] = {}
        self._by_exact: dict[str, list[dict[str, Any]]] = {}
        self._by_canon: dict[str, list[dict[str, Any]]] = {}
        self._by_initial_last: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._entries: list[dict[str, Any]] = []

        for idx, raw in enumerate(entries):
            pid = int(raw.get("player_id", 0) or 0)
            name = str(raw.get("name", "") or "")
            name_clean = clean_name(raw.get("name_clean") or name)
            team = normalize_team_abbrev(raw.get("team"))
            pos_grp = _normalize_pos_group(raw.get("position"))
            first, last, canon, tokens = _split_name_parts_with_last(name or name_clean)
            rec = {
                "order": idx,
                "player_id": pid,
                "name_clean": name_clean,
                "team": team,
                "pos_grp": pos_grp,
                "first": first,
                "last": last,
                "canon": canon,
                "payload": raw.get("payload", raw),
            }
            self._entries.append(rec)
            if pid > 0 and pid not in self._by_id:
                self._by_id[pid] = rec["payload"]
            if name_clean:
                self._by_exact.setdefault(name_clean, []).append(rec)
            if canon:
                self._by_canon.setdefault(canon, []).append(rec)
            if first and last:
                self._by_initial_last.setdefault((first[0], last), []).append(rec)
                # Handle two-part/hyphenated first names abbreviated to first initial + final token (e.g. Jean-Gabriel Pageau -> J. Pageau)
                if len(tokens) >= 3 and tokens[-2] not in _SURNAME_PARTICLES and tokens[-1] != last:
                    self._by_initial_last.setdefault((tokens[0][0], tokens[-1]), []).append(rec)
                # Also index raw clean first initial in case canonical alias shifted initial (e.g., Yegor vs Egor)
                raw_tok = name_clean.split()[0] if name_clean else ""
                if raw_tok and raw_tok[0] != first[0]:
                    self._by_initial_last.setdefault((raw_tok[0], last), []).append(rec)
                    if len(tokens) >= 3 and tokens[-2] not in _SURNAME_PARTICLES and tokens[-1] != last:
                        self._by_initial_last.setdefault((raw_tok[0], tokens[-1]), []).append(rec)

    @staticmethod
    def _filter_scope(
        candidates: list[dict[str, Any]],
        *,
        team_norm: str,
        pos_grp: str,
        require_same_team: bool,
        require_compatible_pos: bool,
    ) -> list[dict[str, Any]]:
        out = candidates
        if require_same_team and team_norm not in ("", "UNK"):
            out = [r for r in out if r["team"] in (team_norm, "UNK")]
        if require_compatible_pos and pos_grp:
            out = [r for r in out if not r["pos_grp"] or r["pos_grp"] == pos_grp]
        return out

    @staticmethod
    def _pick_best(
        candidates: list[dict[str, Any]],
        *,
        team: str = "",
        pos_grp: str = "",
    ) -> Any:
        if len(candidates) == 1:
            return candidates[0]["payload"]
        team_norm = normalize_team_abbrev(team) if team else "UNK"
        best = min(
            candidates,
            key=lambda r: (
                0 if (team_norm != "UNK" and r["team"] == team_norm) else 1,
                0 if (pos_grp and r["pos_grp"] == pos_grp) else 1,
                r["order"],
            ),
        )
        return best["payload"]

    def match(
        self,
        name: str = "",
        *,
        name_clean_key: str = "",
        team: str = "",
        position: str = "",
        player_id: int = 0,
        allow_initial_fallback: bool = False,
        require_same_team: bool = False,
        require_compatible_pos: bool = False,
    ) -> Any | None:
        if player_id > 0 and player_id in self._by_id:
            return self._by_id[player_id]

        team_norm = normalize_team_abbrev(team) if team else "UNK"
        pos_grp = _normalize_pos_group(position)
        exact_key = clean_name(name_clean_key or name)
        if exact_key and exact_key in self._by_exact:
            hits = self._filter_scope(
                self._by_exact[exact_key],
                team_norm=team_norm,
                pos_grp=pos_grp,
                require_same_team=require_same_team,
                require_compatible_pos=require_compatible_pos,
            )
            if hits:
                return self._pick_best(hits, team=team, pos_grp=pos_grp)

        q_first, q_last, q_canon, q_tokens = _split_name_parts_with_last(name or exact_key)
        if q_canon and q_canon in self._by_canon:
            hits = self._filter_scope(
                self._by_canon[q_canon],
                team_norm=team_norm,
                pos_grp=pos_grp,
                require_same_team=require_same_team,
                require_compatible_pos=require_compatible_pos,
            )
            if hits:
                return self._pick_best(hits, team=team, pos_grp=pos_grp)

        if q_first and q_last:
            fuzzy_hits = [
                rec
                for rec in self._entries
                if accent_drop_match(q_first, q_last, rec["first"], rec["last"])
            ]
            fuzzy_hits = self._filter_scope(
                fuzzy_hits,
                team_norm=team_norm,
                pos_grp=pos_grp,
                require_same_team=require_same_team,
                require_compatible_pos=require_compatible_pos,
            )
            if fuzzy_hits:
                return self._pick_best(fuzzy_hits, team=team, pos_grp=pos_grp)

            if allow_initial_fallback:
                raw_tokens = exact_key.split()
                initials = {q_first[0]}
                if raw_tokens:
                    initials.add(raw_tokens[0][0])
                last_keys = [q_last]
                if len(q_tokens) >= 3 and q_tokens[-2] not in _SURNAME_PARTICLES and q_tokens[-1] != q_last:
                    last_keys.append(q_tokens[-1])
                init_hits: list[dict[str, Any]] = []
                seen_orders: set[int] = set()
                for init_ch in initials:
                    for lk in last_keys:
                        for rec in self._by_initial_last.get((init_ch, lk), []):
                            if rec["order"] not in seen_orders and (len(rec["first"]) == 1 or len(q_first) == 1):
                                seen_orders.add(rec["order"])
                                init_hits.append(rec)
                if not init_hits:
                    # Also handle initial + dropped-accent last name (e.g. "A. Rty" / "F. Unger Srum")
                    for rec in self._entries:
                        if (
                            rec["order"] not in seen_orders
                            and rec["first"]
                            and rec["first"][0] in initials
                            and (len(rec["first"]) == 1 or len(q_first) == 1)
                            and accent_drop_match("xx", q_last, "xx", rec["last"])
                        ):
                            seen_orders.add(rec["order"])
                            init_hits.append(rec)
                init_hits = self._filter_scope(
                    init_hits,
                    team_norm=team_norm,
                    pos_grp=pos_grp,
                    require_same_team=require_same_team,
                    require_compatible_pos=require_compatible_pos,
                )
                if init_hits:
                    if team_norm != "UNK":
                        team_hits = [r for r in init_hits if r["team"] in (team_norm, "UNK")]
                        if team_hits:
                            return self._pick_best(team_hits, team=team_norm, pos_grp=pos_grp)
                        # Trade fallback when same-team matching isn't strictly required and candidate is unique
                        if not require_same_team and len({r["order"] for r in init_hits}) == 1:
                            return init_hits[0]["payload"]
                    elif len({r["order"] for r in init_hits}) == 1:
                        return init_hits[0]["payload"]

        return None


def build_player_value_table(archive_dir: str, years: list[int]) -> dict[str, dict[str, Any]]:
    """Aggregate per-player MoneyPuck value rates across all team rows.

    Returns a dict keyed by ``clean_name(name)`` with display name plus current
    (``max(years)``) and prior (``max(years)-1``) per-game rates and GP.
    """
    if not years:
        return {}

    years_sorted = sorted(int(y) for y in years)
    current_year = years_sorted[-1]
    prior_year = current_year - 1

    per_year: dict[int, dict[str, dict[str, float]]] = {}
    for year in years_sorted:
        path = os.path.join(archive_dir, f"{year}_regular_skaters.csv")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "situation" in df.columns:
            df = df[df["situation"].astype(str).str.strip().str.lower() == "all"].copy()
        if df.empty:
            continue

        name_col = "name" if "name" in df.columns else ("playerName" if "playerName" in df.columns else None)
        if name_col is None:
            continue

        for _, row in df.iterrows():
            raw_name = row.get(name_col)
            if not isinstance(raw_name, str) or not str(raw_name).strip():
                continue
            key = clean_name(raw_name)
            if not key:
                continue
            gp = float(pd.to_numeric(row.get("games_played", 0.0), errors="coerce") or 0.0)
            if gp <= 0.0:
                gp = 0.0
            points = float(pd.to_numeric(row.get("I_F_points", 0.0), errors="coerce") or 0.0)
            xgoals = float(pd.to_numeric(row.get("I_F_xGoals", 0.0), errors="coerce") or 0.0)
            icetime = float(pd.to_numeric(row.get("icetime", 0.0), errors="coerce") or 0.0)

            agg = per_year.setdefault(year, {}).setdefault(
                key,
                {"name": str(raw_name).strip(), "gp": 0.0, "points": 0.0, "xgoals": 0.0, "icetime": 0.0},
            )
            agg["gp"] += gp
            agg["points"] += points
            agg["xgoals"] += xgoals
            agg["icetime"] += icetime
            if not agg.get("name"):
                agg["name"] = str(raw_name).strip()

    def _rates(agg: dict[str, float] | None) -> dict[str, Any]:
        if not agg:
            return {}
        gp = float(agg.get("gp", 0.0) or 0.0)
        denom = gp if gp > 0.0 else 1.0
        return {
            "gp": float(agg.get("gp", 0.0) or 0.0),
            "points": float(agg.get("points", 0.0) or 0.0),
            "xgoals": float(agg.get("xgoals", 0.0) or 0.0),
            "icetime": float(agg.get("icetime", 0.0) or 0.0),
            "points_pg": float(agg.get("points", 0.0) or 0.0) / denom,
            "xgoals_pg": float(agg.get("xgoals", 0.0) or 0.0) / denom,
            "icetime_pg": float(agg.get("icetime", 0.0) or 0.0) / denom,
            "points_per_game": float(agg.get("points", 0.0) or 0.0) / denom,
            "xgoals_per_game": float(agg.get("xgoals", 0.0) or 0.0) / denom,
        }

    all_keys: set[str] = set()
    for y in years_sorted:
        all_keys |= set(per_year.get(y, {}).keys())

    result: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        cur_agg = per_year.get(current_year, {}).get(key)
        prior_agg = per_year.get(prior_year, {}).get(key)
        cur_rates = _rates(cur_agg)
        prior_rates = _rates(prior_agg)
        display = (cur_agg or prior_agg or {}).get("name", key)

        entry: dict[str, Any] = {
            "name": display,
            "display_name": display,
            "player_key": key,
            "current": cur_rates,
            "prior": prior_rates,
        }

        for prefix, rates in (("cur", cur_rates), ("prior", prior_rates)):
            if rates:
                entry[f"gp_{prefix}"] = rates["gp"]
                entry[f"points_{prefix}"] = rates["points"]
                entry[f"xgoals_{prefix}"] = rates["xgoals"]
                entry[f"icetime_{prefix}"] = rates["icetime"]
                entry[f"points_pg_{prefix}"] = rates["points_pg"]
                entry[f"xgoals_pg_{prefix}"] = rates["xgoals_pg"]
                entry[f"icetime_pg_{prefix}"] = rates["icetime_pg"]

        entry.setdefault("gp_cur", (cur_rates or {}).get("gp", 0.0))
        entry.setdefault("gp_prior", (prior_rates or {}).get("gp", 0.0))
        entry.setdefault("points_pg_cur", (cur_rates or {}).get("points_pg", 0.0))
        entry.setdefault("points_pg_prior", (prior_rates or {}).get("points_pg", 0.0))
        entry.setdefault("xgoals_pg_cur", (cur_rates or {}).get("xgoals_pg", 0.0))
        entry.setdefault("xgoals_pg_prior", (prior_rates or {}).get("xgoals_pg", 0.0))

        result[key] = entry

    return result


def build_roster_aware_star_table(
    rosters: dict[str, Any],
    player_values: dict[str, dict[str, Any]],
    *,
    prior_weight_gp: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Build per-team Top-3 star aggregates from live rosters + player values."""
    if not rosters or not player_values:
        return {}

    entries = []
    for key, val in player_values.items():
        if not isinstance(val, dict):
            continue
        display = val.get("name") or val.get("display_name") or key
        entries.append(
            {
                "player_id": int(val.get("player_id", 0) or 0),
                "name": str(display),
                "name_clean": key,
                "team": "",
                "position": "",
                "payload": (key, val),
            }
        )
    matcher = PlayerMatcher(entries) if entries else None

    result: dict[str, dict[str, Any]] = {}
    for team, roster in rosters.items():
        if not isinstance(roster, list) or not roster:
            continue
        team_norm = normalize_team_abbrev(team)
        matched: list[dict[str, Any]] = []

        for p in roster:
            if isinstance(p, dict):
                pname = str(p.get("name") or p.get("player") or "").strip()
                pos = str(p.get("position") or p.get("positionCode") or "").strip()
                pid = p.get("id", p.get("player_id", 0))
            else:
                pname = str(p).strip()
                pos = ""
                pid = 0

            if not pname:
                continue

            key = clean_name(pname)
            pv = player_values.get(key)
            if pv is None and matcher is not None:
                m = matcher.match(
                    name=pname,
                    team=team_norm,
                    position=pos,
                    player_id=int(pid or 0),
                    allow_initial_fallback=True,
                )
                if isinstance(m, tuple) and len(m) == 2:
                    key, pv = m

            if not isinstance(pv, dict):
                continue

            cur = pv.get("current") if isinstance(pv.get("current"), dict) else {}
            prior = pv.get("prior") if isinstance(pv.get("prior"), dict) else {}

            gp_cur = float(cur.get("gp", pv.get("gp_cur", 0.0)) or 0.0)
            gp_prior = float(prior.get("gp", pv.get("gp_prior", 0.0)) or 0.0)

            cur_pts = float(cur.get("points_pg", cur.get("points_per_game", pv.get("points_pg_cur", 0.0))) or 0.0)
            prior_pts = float(prior.get("points_pg", prior.get("points_per_game", pv.get("points_pg_prior", 0.0))) or 0.0)
            cur_xg = float(cur.get("xgoals_pg", cur.get("xgoals_per_game", pv.get("xgoals_pg_cur", 0.0))) or 0.0)
            prior_xg = float(prior.get("xgoals_pg", prior.get("xgoals_per_game", pv.get("xgoals_pg_prior", 0.0))) or 0.0)

            if gp_cur > 0.0 and gp_prior > 0.0:
                w = gp_cur / (gp_cur + float(prior_weight_gp))
                blended_pts = (1.0 - w) * prior_pts + w * cur_pts
                blended_xg = (1.0 - w) * prior_xg + w * cur_xg
            elif gp_cur > 0.0:
                blended_pts = cur_pts
                blended_xg = cur_xg
            elif gp_prior > 0.0:
                blended_pts = prior_pts
                blended_xg = prior_xg
            else:
                blended_pts = 0.0
                blended_xg = 0.0

            matched.append(
                {
                    "name": pname,
                    "key": key,
                    "blended_pts_pg": blended_pts,
                    "blended_xg_pg": blended_xg,
                }
            )

        if not matched:
            continue

        matched.sort(key=lambda x: x["blended_pts_pg"], reverse=True)
        top3 = matched[:3]
        depth = matched[3:]
        t3_pts_pg = float(sum(x["blended_pts_pg"] for x in top3))
        t3_xg_pg = float(sum(x["blended_xg_pg"] for x in top3))
        team_gp = 1.0

        result[team_norm] = {
            "Top3_Points_per_game": t3_pts_pg,
            "Top3_xGoals_per_game": t3_xg_pg,
            "Top3_Points": t3_pts_pg * team_gp,
            "Top3_xGoals": t3_xg_pg * team_gp,
            "Depth_Points_per_game": float(sum(x["blended_pts_pg"] for x in depth)),
            "Depth_xGoals_per_game": float(sum(x["blended_xg_pg"] for x in depth)),
        }

    return result



def build_roster_membership_moves(
    rosters: dict[str, Any],
    archive_dir: str,
    year: int,
    player_values: dict[str, dict[str, Any]] | None = None,
    injury_records: list[dict[str, Any]] | None = None,
    *,
    min_value: float = 0.15,
) -> dict[str, dict[str, list[str]]]:
    """Day-one roster moves: current roster vs the team's season-attributed skaters.

    Works without a prior roster snapshot (e.g. the Larkin trade on deployment
    day). Injured players are excluded from the "out" list, and low-value
    callups/rookies are filtered out.
    """
    if not rosters:
        return {}
    path = os.path.join(archive_dir, f"{year}_regular_skaters.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "name" not in df.columns or "team" not in df.columns:
        return {}
    if "situation" in df.columns:
        df = df[df["situation"].astype(str).str.strip().str.lower() == "all"].copy()
    if df.empty:
        return {}

    attributed: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        team = normalize_team_abbrev(row.get("team"))
        if not name or team == "UNK":
            continue
        gp = float(pd.to_numeric(row.get("games_played", 0.0), errors="coerce") or 0.0)
        pts = float(pd.to_numeric(row.get("I_F_points", 0.0), errors="coerce") or 0.0)
        if gp <= 0.0:
            continue
        attributed.setdefault(team, {})[clean_name(name)] = {"name": name, "ppg": pts / gp}

    injured: dict[str, set[str]] = {}
    for rec in injury_records or []:
        if not isinstance(rec, dict):
            continue
        if not (rec.get("is_out") or rec.get("is_dtd")):
            continue
        team = normalize_team_abbrev(rec.get("team"))
        key = clean_name(rec.get("player") or "")
        if team != "UNK" and key:
            injured.setdefault(team, set()).add(key)

    def _value(player_key: str, fallback: float = 0.0) -> float:
        pv = (player_values or {}).get(player_key) or {}
        cur = pv.get("current") if isinstance(pv.get("current"), dict) else {}
        prior = pv.get("prior") if isinstance(pv.get("prior"), dict) else {}
        gp_cur = float(cur.get("gp", pv.get("gp_cur", 0.0)) or 0.0)
        gp_prior = float(prior.get("gp", pv.get("gp_prior", 0.0)) or 0.0)
        cur_pg = float(cur.get("points_pg", pv.get("points_pg_cur", 0.0)) or 0.0)
        prior_pg = float(prior.get("points_pg", pv.get("points_pg_prior", 0.0)) or 0.0)
        if gp_cur > 0.0 and gp_prior > 0.0:
            w = gp_cur / (gp_cur + 20.0)
            return (1.0 - w) * prior_pg + w * cur_pg
        if gp_cur > 0.0:
            return cur_pg
        if gp_prior > 0.0:
            return prior_pg
        return fallback

    roster_by_key: dict[str, str] = {}
    for team, roster in rosters.items():
        team_norm = normalize_team_abbrev(team)
        for p in roster or []:
            if isinstance(p, dict):
                nm = str(p.get("name") or "").strip()
                if nm:
                    roster_by_key[clean_name(nm)] = team_norm

    attr_index: dict[str, tuple[str, dict[str, Any]]] = {}
    for team_key, players in attributed.items():
        for key, info in players.items():
            attr_index[key] = (team_key, info)

    def _lookup(index: dict[str, Any], key: str) -> Any:
        if key in index:
            return index[key]
        # tolerate MoneyPuck truncation (e.g. "Juraj Slafkovsk" vs "Juraj Slafkovsky")
        if len(key) >= 8:
            for other in index:
                if other.startswith(key) or key.startswith(other):
                    return index[other]
        return None

    moves: dict[str, dict[str, list[str]]] = {}
    for team, roster in rosters.items():
        if not isinstance(roster, list) or not roster:
            continue
        team_norm = normalize_team_abbrev(team)
        roster_names = {}
        for p in roster:
            if not isinstance(p, dict):
                continue
            nm = str(p.get("name") or "").strip()
            if nm:
                roster_names[clean_name(nm)] = nm
        team_attr = attributed.get(team_norm, {})
        team_inj = injured.get(team_norm, set())

        incoming: list[str] = []
        for key, display in roster_names.items():
            found = _lookup(attr_index, key)
            if not found:
                continue  # rookies / no season attribution
            owner_team, info = found
            if owner_team == team_norm:
                continue
            if _value(key, float(info.get("ppg", 0.0) or 0.0)) < min_value:
                continue
            incoming.append(display)

        outgoing: list[str] = []
        for key, info in team_attr.items():
            if key in team_inj:
                continue
            owner = _lookup(roster_by_key, key)
            if not owner or owner == team_norm:
                continue  # AHL / no longer in the NHL
            if _value(key, float(info.get("ppg", 0.0) or 0.0)) < min_value:
                continue
            outgoing.append(str(info.get("name") or key.title()))

        if incoming or outgoing:
            moves[team_norm] = {"in": sorted(incoming)[:5], "out": sorted(outgoing)[:5]}
    return moves


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
                    "player_id": int(sk.get("playerId", 0) or 0),
                    "name": full_name,
                    "live_gp": int(sk.get("gamesPlayed", 0) or 0),
                    "live_goals": int(sk.get("goals", 0) or 0),
                    "live_points": int(sk.get("points", 0) or 0),
                    "live_shots": int(sk.get("shots", 0) or 0),
                    "team": team,
                    "position": str(sk.get("positionCode", "") or ""),
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
