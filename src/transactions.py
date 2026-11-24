"""NHL roster + transaction ingestion, history archiving, and diffing."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_sync import (
    _fetch_with_retries,
    clean_name,
    create_http_session,
    normalize_team_abbrev,
)

ESPN_TEAM_ALIASES: dict[str, str] = {
    "SJ": "SJS", "TB": "TBL", "LA": "LAK", "NJ": "NJD",
    "NY": "NYR", "WAS": "WSH", "VEG": "VGK", "MON": "MTL",
    "NAS": "NSH", "CLB": "CBJ", "PHO": "ARI",
}

NHL_TEAM_ABBRS: list[str] = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]


def fetch_team_rosters(season: str | int, session: Any | None = None) -> dict[str, list[dict[str, Any]]]:
    """Fetch all 32 NHL team rosters for ``season`` (e.g. ``20262027``)."""
    http = session or create_http_session()
    out: dict[str, list[dict[str, Any]]] = {}
    for team in NHL_TEAM_ABBRS:
        url = f"https://api-web.nhle.com/v1/roster/{team}/{season}"
        res = _fetch_with_retries(http, url, timeout=12, attempts=3)
        if res is None or res.status_code != 200:
            continue
        try:
            data = res.json()
        except Exception:
            continue
        players: list[dict[str, Any]] = []
        for group in ("forwards", "defensemen", "goalies"):
            for p in data.get(group, []) or []:
                first = (p.get("firstName") or {}).get("default", "")
                last = (p.get("lastName") or {}).get("default", "")
                name = f"{first} {last}".strip()
                if not name:
                    continue
                players.append(
                    {
                        "id": p.get("id"),
                        "name": name,
                        "position": p.get("positionCode"),
                        "team": team,
                    }
                )
        if players:
            out[team] = players
    return out


def fetch_espn_transactions(pages: int = 1, session: Any | None = None) -> list[dict[str, Any]]:
    """Fetch and normalize ESPN NHL transactions (newest first)."""
    http = session or create_http_session()
    out: list[dict[str, Any]] = []
    pages = max(1, int(pages or 1))
    for page in range(1, pages + 1):
        url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/transactions"
        if page > 1:
            url = f"{url}?page={page}"
        res = _fetch_with_retries(http, url, timeout=12, attempts=3)
        if res is None or res.status_code != 200:
            break
        try:
            data = res.json()
        except Exception:
            break
        txs = data.get("transactions") or []
        if not txs:
            break
        for tr in txs:
            desc = str(tr.get("description") or "")
            low = desc.lower()
            team_raw = tr.get("team")
            if isinstance(team_raw, dict):
                team_raw = (
                    team_raw.get("abbreviation")
                    or team_raw.get("shortDisplayName")
                    or team_raw.get("displayName")
                )
            team_raw = str(team_raw or "").strip().upper()
            team = normalize_team_abbrev(ESPN_TEAM_ALIASES.get(team_raw, team_raw))
            out.append(
                {
                    "date": tr.get("date"),
                    "description": desc,
                    "team": team,
                    "is_trade": ("trade" in low) or ("acquired" in low),
                    "is_waiver": "waiver" in low,
                    "is_ir": ("injured reserve" in low) or (" ir " in f" {low} "),
                    "is_recall": "recall" in low,
                    "is_signing": "sign" in low,
                }
            )
        try:
            page_count = int(data.get("pageCount") or page)
        except (TypeError, ValueError):
            page_count = page
        if page >= page_count:
            break
    return out


def write_roster_history(
    rosters: Any,
    docs_dir: str | os.PathLike[str],
    date_str: str,
    captured_utc: str | None = None,
) -> str:
    """Persist a daily roster snapshot under ``<docs_dir>/roster_history/``."""
    base_path = Path(docs_dir)
    out_dir = base_path if base_path.name == "roster_history" else base_path / "roster_history"
    out_dir.mkdir(parents=True, exist_ok=True)

    teams = rosters
    if isinstance(rosters, dict) and "teams" in rosters and isinstance(rosters["teams"], dict):
        teams = rosters["teams"]

    payload = {
        "date": date_str,
        "captured_utc": captured_utc or datetime.now(timezone.utc).isoformat(),
        "teams": teams,
    }
    path = out_dir / f"{date_str}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def load_roster_snapshot(
    docs_dir: str | os.PathLike[str],
    date_str: str,
    max_age_minutes: float = 120.0,
) -> dict[str, Any] | None:
    """Load today's roster snapshot when it is still fresh (avoids 32 fetches per run)."""
    base_path = Path(docs_dir)
    out_dir = base_path if base_path.name == "roster_history" else base_path / "roster_history"
    path = out_dir / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    captured = snap.get("captured_utc")
    if captured:
        try:
            dt = datetime.fromisoformat(str(captured).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_min = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0
            if age_min > float(max_age_minutes):
                return None
        except ValueError:
            pass
    return snap


def load_latest_roster_history(docs_dir: str | os.PathLike[str], before_date: str | None = None) -> dict[str, Any] | None:
    """Load the most recent roster snapshot strictly before ``before_date``."""
    base_path = Path(docs_dir)
    out_dir = base_path if base_path.name == "roster_history" else base_path / "roster_history"
    if not out_dir.exists():
        return None
    candidates = []
    for f in out_dir.glob("*.json"):
        stem = f.stem
        if before_date is not None and not (stem < str(before_date)):
            continue
        candidates.append(f)
    if not candidates:
        return None
    latest = sorted(candidates)[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None


def _roster_teams(snapshot: Any) -> dict[str, list[Any]]:
    if not isinstance(snapshot, dict):
        return {}
    if "teams" in snapshot and isinstance(snapshot["teams"], dict):
        return snapshot["teams"]
    if snapshot and all(isinstance(v, list) for v in snapshot.values()):
        return snapshot
    return {}


def _player_id(player: Any) -> Any:
    if isinstance(player, dict):
        for key in ("id", "player_id", "playerId"):
            val = player.get(key)
            if val is not None:
                return val
        name = player.get("name") or player.get("player")
        return clean_name(name) if name else None
    return clean_name(str(player))


def _player_name(player: Any) -> str:
    if isinstance(player, dict):
        return str(player.get("name") or player.get("player") or "").strip()
    return str(player).strip()


def diff_rosters(prev: Any, curr: Any) -> dict[str, Any]:
    """Diff two roster snapshots by NHL id (fallback clean_name)."""
    prev_teams = _roster_teams(prev)
    curr_teams = _roster_teams(curr)

    prev_players: dict[Any, tuple[str, str, Any]] = {}
    for team, roster in prev_teams.items():
        for p in roster or []:
            pid = _player_id(p)
            if pid is None:
                continue
            prev_players[pid] = (str(team), _player_name(p), p)

    curr_players: dict[Any, tuple[str, str, Any]] = {}
    for team, roster in curr_teams.items():
        for p in roster or []:
            pid = _player_id(p)
            if pid is None:
                continue
            curr_players[pid] = (str(team), _player_name(p), p)

    moves: dict[Any, dict[str, Any]] = {}
    team_changes: dict[str, dict[str, Any]] = {}

    for pid in set(prev_players) & set(curr_players):
        prev_team = prev_players[pid][0]
        curr_team = curr_players[pid][0]
        if prev_team == curr_team:
            continue
        name = curr_players[pid][1] or prev_players[pid][1]
        moves[pid] = {"name": name, "from": prev_team, "to": curr_team}

        team_changes.setdefault(
            prev_team,
            {"in": [], "out": [], "in_ids": [], "out_ids": [], "in_players": [], "out_players": []},
        )
        team_changes.setdefault(
            curr_team,
            {"in": [], "out": [], "in_ids": [], "out_ids": [], "in_players": [], "out_players": []},
        )
        team_changes[prev_team]["out"].append(name)
        team_changes[prev_team]["out_ids"].append(pid)
        team_changes[prev_team]["out_players"].append(prev_players[pid][2])
        team_changes[curr_team]["in"].append(name)
        team_changes[curr_team]["in_ids"].append(pid)
        team_changes[curr_team]["in_players"].append(curr_players[pid][2])

    result: dict[str, Any] = dict(moves)
    result["moves"] = moves
    for team, info in team_changes.items():
        result[team] = info
    result["teams"] = team_changes
    return result



def compute_name_churn(
    roster_names: dict[str, Any] | None,
    player_values: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Per-team churn (0-1) from named roster in/out lists.

    Works even when the saved profiles are already roster-aware (the top-3
    delta is then zero), which is the case on a full-run day.
    """
    net: dict[str, float] = {}
    for team, info in (roster_names or {}).items():
        if not isinstance(info, dict) or not isinstance(team, str):
            continue
        for name in info.get("in") or []:
            net[team] = net.get(team, 0.0) + _blended_player_value(
                (player_values or {}).get(clean_name(str(name)), {})
            )
        for name in info.get("out") or []:
            net[team] = net.get(team, 0.0) - _blended_player_value(
                (player_values or {}).get(clean_name(str(name)), {})
            )
    return {team: min(1.0, abs(value)) for team, value in net.items() if value}


def _trade_descriptions(recent_trades: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for tr in recent_trades or []:
        if isinstance(tr, dict) and tr.get("is_trade"):
            desc = clean_name(str(tr.get("description") or ""))
            if desc:
                out.append(desc)
    return out


def _player_is_traded(name: str, descriptions: list[str]) -> bool:
    key = clean_name(str(name or ""))
    return bool(key) and any(key in desc for desc in descriptions)


def _blended_player_value(pv: dict[str, Any], prior_weight_gp: float = 20.0) -> float:
    """Blend prior/current points-per-game the same way the roster star table does."""
    if not isinstance(pv, dict):
        return 0.0
    cur = pv.get("current") if isinstance(pv.get("current"), dict) else {}
    prior = pv.get("prior") if isinstance(pv.get("prior"), dict) else {}
    gp_cur = float(cur.get("gp", pv.get("gp_cur", 0.0)) or 0.0)
    gp_prior = float(prior.get("gp", pv.get("gp_prior", 0.0)) or 0.0)
    cur_pg = float(cur.get("points_pg", pv.get("points_pg_cur", 0.0)) or 0.0)
    prior_pg = float(prior.get("points_pg", pv.get("points_pg_prior", 0.0)) or 0.0)
    if gp_cur > 0.0 and gp_prior > 0.0:
        w = gp_cur / (gp_cur + max(1.0, float(prior_weight_gp)))
        return (1.0 - w) * prior_pg + w * cur_pg
    if gp_cur > 0.0:
        return cur_pg
    if gp_prior > 0.0:
        return prior_pg
    return 0.0


def compute_team_churn(
    roster_moves: Any,
    player_values: dict[str, dict[str, Any]] | None = None,
    *,
    recent_trades: list[dict[str, Any]] | None = None,
    player_value_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute per-team churn (0-1) from traded roster moves."""
    if player_value_table is not None:
        player_values = player_value_table
    descriptions = _trade_descriptions(recent_trades)

    moves: dict[Any, Any] = {}
    if isinstance(roster_moves, dict):
        if "moves" in roster_moves and isinstance(roster_moves["moves"], dict):
            moves = roster_moves["moves"]
        else:
            moves = {
                k: v
                for k, v in roster_moves.items()
                if isinstance(v, dict) and "from" in v and "to" in v
            }

    def _value_of(name: str) -> float:
        return _blended_player_value((player_values or {}).get(clean_name(name), {}))

    net: dict[str, float] = {}
    for _pid, move in moves.items():
        if not isinstance(move, dict):
            continue
        name = str(move.get("name") or "")
        if not name:
            continue
        if descriptions and not _player_is_traded(name, descriptions):
            continue
        frm = move.get("from")
        to = move.get("to")
        if not frm or not to:
            continue
        val = _value_of(name)
        net[to] = net.get(to, 0.0) + val
        net[frm] = net.get(frm, 0.0) - val

    all_teams = set(NHL_TEAM_ABBRS) | set(net.keys())
    result: dict[str, float] = {}
    for team in all_teams:
        result[team] = min(1.0, abs(net.get(team, 0.0)) / 1.0)
    return result

def write_roster_moves_index(
    path: str,
    roster_names: dict[str, Any] | None,
    churn: dict[str, float] | None,
) -> None:
    """Persist named roster moves + churn so infer-only runs (no data archive) can show them."""
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "teams": {
            str(team): {
                "in": list((info or {}).get("in") or [])[:5],
                "out": list((info or {}).get("out") or [])[:5],
            }
            for team, info in ((roster_names or {}).get("teams") or {}).items()
            if isinstance(info, dict)
        },
        "churn": {
            str(team): float(value)
            for team, value in (churn or {}).items()
            if value
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_roster_moves_index(path: str) -> dict[str, Any]:
    """Load the persisted roster-moves index (empty dict when missing/invalid)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
