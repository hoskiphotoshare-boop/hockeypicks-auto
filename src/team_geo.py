"""Static NHL arena geography and timezone helpers for fatigue/travel features."""

from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

TEAM_GEO: dict[str, tuple[float, float, str]] = {
    "ANA": (33.8078, -117.8765, "America/Los_Angeles"),
    "ARI": (33.4242, -111.9400, "America/Phoenix"),
    "BOS": (42.3663, -71.0622, "America/New_York"),
    "BUF": (42.8750, -78.8764, "America/New_York"),
    "CAR": (35.8033, -78.7218, "America/New_York"),
    "CBJ": (39.9694, -83.0061, "America/New_York"),
    "CGY": (51.0447, -114.0547, "America/Calgary"),
    "CHI": (41.8807, -87.6742, "America/Chicago"),
    "COL": (39.7487, -105.0077, "America/Denver"),
    "DAL": (32.7905, -96.8103, "America/Chicago"),
    "DET": (42.3411, -83.0551, "America/Detroit"),
    "EDM": (53.5461, -113.4938, "America/Edmonton"),
    "FLA": (26.1584, -80.3256, "America/New_York"),
    "LAK": (34.0430, -118.2673, "America/Los_Angeles"),
    "MIN": (44.9447, -93.1010, "America/Chicago"),
    "MTL": (45.4961, -73.5693, "America/Montreal"),
    "NSH": (36.1616, -86.7780, "America/Chicago"),
    "NJD": (40.7336, -74.1711, "America/New_York"),
    "NYI": (40.7200, -73.7250, "America/New_York"),
    "NYR": (40.7505, -73.9934, "America/New_York"),
    "OTT": (45.2969, -75.9272, "America/Toronto"),
    "PHI": (39.9012, -75.1720, "America/New_York"),
    "PIT": (40.4395, -79.9896, "America/New_York"),
    "SEA": (47.6221, -122.3540, "America/Los_Angeles"),
    "SJS": (37.3327, -121.9012, "America/Los_Angeles"),
    "STL": (38.6267, -90.2026, "America/Chicago"),
    "TBL": (27.9428, -82.4518, "America/New_York"),
    "TOR": (43.6435, -79.3791, "America/Toronto"),
    "UTA": (40.7683, -111.9005, "America/Denver"),
    "VAN": (49.2778, -123.1089, "America/Vancouver"),
    "VGK": (36.1029, -115.1784, "America/Los_Angeles"),
    "WPG": (49.8928, -97.1436, "America/Winnipeg"),
    "WSH": (38.8981, -77.0209, "America/New_York"),
}


def _norm_team(team: Any) -> str | None:
    try:
        return str(team).strip().upper() if team is not None else None
    except Exception:
        return None


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(r * c)


def team_coords(team: Any) -> tuple[float, float] | None:
    t = _norm_team(team)
    if t is None or t not in TEAM_GEO:
        return None
    lat, lon, _ = TEAM_GEO[t]
    return (float(lat), float(lon))


def _resolve_tz(team_or_tz: Any) -> str | None:
    if isinstance(team_or_tz, str) and "/" in team_or_tz:
        return team_or_tz.strip()
    t = _norm_team(team_or_tz)
    if t is not None and t in TEAM_GEO:
        return TEAM_GEO[t][2]
    if isinstance(team_or_tz, str) and team_or_tz.strip():
        return team_or_tz.strip()
    return None


def tz_offset_hours(team_or_tz: Any, on_date: Any) -> int:
    tz_name = _resolve_tz(team_or_tz)
    if tz_name is None:
        return 0
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return 0
    try:
        import pandas as pd

        dd = pd.to_datetime(on_date, errors="coerce")
        if pd.isna(dd):
            return 0
        d = dd.date()
    except Exception:
        return 0
    dt = datetime.combine(d, time(12, 0), tzinfo=tz)
    offset = dt.utcoffset()
    if offset is None:
        return 0
    return int(offset.total_seconds() // 3600)


def travel_miles(team_from_venue: Any, team_to_venue: Any) -> float:
    c1 = team_coords(team_from_venue)
    c2 = team_coords(team_to_venue)
    if c1 is None or c2 is None:
        return 0.0
    return haversine_miles(c1[0], c1[1], c2[0], c2[1])


def tz_shift_hours(from_venue: Any, to_venue: Any, on_date: Any) -> int:
    off_from = tz_offset_hours(from_venue, on_date)
    off_to = tz_offset_hours(to_venue, on_date)
    # unknown teams -> 0 offsets (neutral)
    if _norm_team(from_venue) is not None and _norm_team(from_venue) not in TEAM_GEO:
        off_from = 0
    if _norm_team(to_venue) is not None and _norm_team(to_venue) not in TEAM_GEO:
        off_to = 0
    return int(off_to - off_from)
