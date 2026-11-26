#!/usr/bin/env python3
"""Collect historical NHL player game logs for player-stat models.

Reads the MoneyPuck season skater files for the player universe, fetches one
NHL game log per player-season, and caches the parsed rows in
``data/player_game_logs/history``. The combined parquet is the training input
for the per-stat models.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_sync import _fetch_with_retries, create_http_session

CACHE_DIR = os.path.join("data", "player_game_logs", "history")
OUT_PATH = os.path.join(CACHE_DIR, "player_games.csv.gz")
STAT_COLUMNS = [
    "season", "playerId", "name", "position", "team", "gameDate", "gameId",
    "home", "opponent", "goals", "assists", "points", "sog", "pp_points", "toi_min",
]


def _toi_minutes(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw or ":" not in raw:
        return 0.0
    try:
        mm, ss = raw.split(":")[:2]
        return float(mm) + float(ss) / 60.0
    except ValueError:
        return 0.0


def parse_game_log(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for g in (payload or {}).get("gameLog", []) or []:
        if not isinstance(g, dict):
            continue
        rows.append(
            {
                "gameId": g.get("gameId"),
                "gameDate": g.get("gameDate"),
                "team": g.get("teamAbbrev"),
                "home": 1 if str(g.get("homeRoadFlag") or "").upper() == "H" else 0,
                "opponent": g.get("opponentAbbrev"),
                "goals": int(g.get("goals") or 0),
                "assists": int(g.get("assists") or 0),
                "points": int(g.get("points") or 0),
                "sog": int(g.get("shots") or g.get("shotsOnGoal") or 0),
                "pp_points": int(g.get("powerPlayPoints") or 0),
                "toi_min": round(_toi_minutes(g.get("toi")), 3),
            }
        )
    return rows


def season_universe(year: int, min_gp: int = 5) -> pd.DataFrame:
    path = os.path.join("data", "moneypuck", f"{year}_regular_skaters.csv")
    df = pd.read_csv(path)
    if "situation" in df.columns:
        df = df[df["situation"].astype(str).str.lower() == "all"]
    df = df[df["games_played"].fillna(0) >= min_gp]
    df = df[["playerId", "name", "team", "position"]].dropna(subset=["playerId"])
    df["playerId"] = df["playerId"].astype(int)
    return df.drop_duplicates(subset=["playerId"])


def cache_path(season: str, player_id: int) -> str:
    return os.path.join(CACHE_DIR, f"{season}_{player_id}.json")


def fetch_one(session: Any, season: str, row: dict[str, Any], refresh_hours: float) -> tuple[int, list[dict[str, Any]]]:
    pid = int(row["playerId"])
    path = cache_path(season, pid)
    if os.path.exists(path) and os.path.getsize(path) > 2:
        age_h = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(path)) / 3600.0
        if age_h < refresh_hours:
            try:
                return pid, json.load(open(path, encoding="utf-8"))
            except Exception:
                pass
    url = f"https://api-web.nhle.com/v1/player/{pid}/game-log/{season}/2"
    res = _fetch_with_retries(session, url, timeout=15, attempts=2)
    if res is None or getattr(res, "status_code", None) != 200:
        return pid, []
    try:
        rows = parse_game_log(res.json())
    except Exception:
        rows = []
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
    except Exception:
        pass
    return pid, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2021,2022,2023,2024,2025,2026")
    ap.add_argument("--min-gp", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--refresh-hours", type=float, default=12.0)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    session = create_http_session()
    all_rows: list[dict[str, Any]] = []
    for year in [int(y) for y in args.years.split(",") if y.strip()]:
        season = f"{year}{year + 1}"
        universe = season_universe(year, args.min_gp)
        print(f"{season}: {len(universe)} players", flush=True)
        meta = {
            int(r["playerId"]): {
                "name": r["name"], "position": r["position"],
            }
            for r in universe.to_dict("records")
        }
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(fetch_one, session, season, r, args.refresh_hours) for r in universe.to_dict("records")]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    pid, rows = fut.result()
                except Exception:
                    continue
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(universe)}", flush=True)
                for row in rows:
                    row = dict(row)
                    row["season"] = season
                    row["playerId"] = pid
                    row["name"] = meta.get(pid, {}).get("name")
                    row["position"] = meta.get(pid, {}).get("position")
                    all_rows.append(row)
    df = pd.DataFrame(all_rows, columns=STAT_COLUMNS)
    df = df.dropna(subset=["gameDate", "playerId", "goals"])
    df.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"Wrote {len(df)} player-games to {OUT_PATH} ({df['season'].nunique()} seasons)", flush=True)


if __name__ == "__main__":
    main()
