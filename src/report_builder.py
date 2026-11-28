"""Report generation for structured docs/data.json and styled docs/latest_run.html."""

from __future__ import annotations

import copy
import html
import json
import os
from datetime import datetime
from typing import Any

import pytz


UTC_TZ = pytz.UTC
ET_TZ = pytz.timezone("America/New_York")


PUBLIC_DROP_GAME_KEYS = frozenset({
    "book_odds", "away_odds", "home_odds",
    "away_edge", "home_edge", "away_ev", "home_ev",
    "rec_bet", "kelly", "edge_pct", "best_odds", "best_bookie",
    "bookie", "bookie_key",
    "home_depth_eff", "away_depth_eff",
    "home_prob_unadjusted", "away_prob_unadjusted",
})


def sanitize_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a public copy with admin odds/ledger fields removed.

    The Worker also filters responses, but the raw docs/data.json is public in
    the repository; sanitizing at write time closes that bypass.
    """
    if not isinstance(payload, dict):
        return payload
    out = copy.deepcopy(payload)
    out.pop("calls_admin", None)
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta.pop("odds_usage", None)
    games = out.get("games")
    if isinstance(games, list):
        for game in games:
            if isinstance(game, dict):
                for key in PUBLIC_DROP_GAME_KEYS:
                    game.pop(key, None)
    return out


TEAM_KEEP_KEYS = (
    "rank", "team", "conference", "division", "gp", "points", "proj_points",
    "make_playoffs_pct", "make_conf_final_pct", "make_cup_final_pct",
    "model_cup_pct", "win_division_pct",
)
SUMMARY_KEEP_KEYS = ("avg_gp", "max_team_pct", "months_to_cup")
META_KEEP_KEYS = (
    "updated_et", "n_sims", "model_metrics", "model_features",
    "remaining_regular_games", "active_playoff_series_count", "empirical_ot_rate",
    "sim_mode", "status", "today_date", "nhl_season", "nhl_season_label",
)


def sanitize_public_cup_futures(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a public copy of cup futures with all betting fields removed."""
    if not isinstance(payload, dict):
        return payload
    out = copy.deepcopy(payload)
    teams = out.get("teams")
    if isinstance(teams, list):
        sanitized = []
        for t in teams:
            if not isinstance(t, dict):
                continue
            o = {}
            for k in TEAM_KEEP_KEYS:
                if k in t:
                    o[k] = t[k]
            sanitized.append(o)
        out["teams"] = sanitized
    summary = out.get("summary")
    if isinstance(summary, dict):
        o = {}
        for k in SUMMARY_KEEP_KEYS:
            if k in summary:
                o[k] = summary[k]
        out["summary"] = o
    meta = out.get("meta")
    if isinstance(meta, dict):
        o = {}
        for k in META_KEEP_KEYS:
            if k in meta:
                o[k] = meta[k]
        o["odds_enabled"] = False
        out["meta"] = o
    return out


def write_json_report(payload: dict[str, Any], path: str) -> None:
    """Write structured dashboard payload as formatted JSON, rejecting NaN/Infinity."""
    text = json.dumps(payload, indent=2, allow_nan=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_terminal_log_html(log_text: str, meta: dict[str, Any], path: str) -> None:
    """Render headless pipeline stdout trace as a dark-theme HTML terminal log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    escaped = html.escape(log_text)
    updated_et = html.escape(str(meta.get("updated_et", "")))
    season_label = html.escape(str(meta.get("nhl_season_label", "")))
    mode = html.escape(str(meta.get("active_mode", "")))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hockeypicks Pipeline Run Log</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      max-width: 100%;
      overflow-x: hidden;
      background: #070b14;
      color: #e2e8f0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    .wrap {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 20px 16px 40px;
    }}
    .bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: #0f172a;
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 16px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .bar a {{
      color: #10b981;
      text-decoration: none;
      font-weight: 600;
      font-size: 0.9rem;
    }}
    .meta {{
      color: #94a3b8;
      font-size: 0.85rem;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      background: #0b1120;
      border: 1px solid rgba(16, 185, 129, 0.22);
      border-radius: 10px;
      font-size: 0.82rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      max-width: 100%;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <a href="./index.html">← Back to Dashboard</a>
      <span class="meta">Season: {season_label} | Mode: {mode} | Updated: {updated_et}</span>
    </div>
    <pre>{escaped}</pre>
  </div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def build_dashboard_payload(
    *,
    context: dict[str, Any],
    sync_stats: dict[str, Any],
    log_stats: dict[str, Any],
    schedule_info: dict[str, Any],
    tims_status: str,
    skater_bundle: dict[str, Any],
    skater_scores: dict[str, Any],
    skater_baseline_label: str,
    pairwise_bundle: dict[str, Any],
    power_rankings: list[dict[str, Any]],
    game_predictions: list[dict[str, Any]],
    playoff_series: list[dict[str, Any]],
    odds_enabled: bool,
    tims_history: dict[str, Any] | None = None,
    tims_timing_advice: dict[str, Any] | None = None,
    tims_scoreboard: list[dict[str, Any]] | None = None,
    tims_scoreboard_meta: dict[str, Any] | None = None,
    availability_summary: dict[str, Any] | None = None,
    odds_usage: dict[str, Any] | None = None,
    run_mode: str = "full",
    calls_summary: dict[str, Any] | None = None,
    calls_recent: list[dict[str, Any]] | None = None,
    calls_admin: dict[str, Any] | None = None,
    tims_board_players: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the complete JSON payload consumed by docs/index.html."""
    now_utc = datetime.now(UTC_TZ)
    now_et = now_utc.astimezone(ET_TZ)

    meta = {
        "status": "live",
        "run_mode": run_mode,
        "updated_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "updated_et": now_et.strftime("%Y-%m-%d %I:%M %p ET"),
        "today_date": context["today_str"],
        "nhl_season": context["nhl_season"],
        "nhl_season_label": context["nhl_season_label"],
        "active_mp_year": sync_stats.get("active_mp_year"),
        "active_mode": schedule_info["active_mode"],
        "active_game_type": schedule_info["active_game_type"],
        "skater_baseline_source": skater_baseline_label,
        "tims_status": tims_status,
        "odds_enabled": odds_enabled,
        "odds_usage": odds_usage or {},
        "availability": availability_summary or {},
        "moneypuck_sync": sync_stats,
        "game_log_sync": log_stats,
        "models": {
            "skater": {
                "mode": skater_bundle["mode"],
                "n_half": skater_bundle["n_half"],
                **skater_bundle["metrics"],
            },
            "pairwise": pairwise_bundle["metrics"],
        },
    }

    return {
        "meta": meta,
        "tims_picks": skater_scores["tims_groups"],
        "tims_timing": tims_timing_advice or {},
        "tims_history": tims_history
        or {"total_days_logged": 0, "settled_days": 0, "pending_days": 0, "recent_days": []},
        "tims_scoreboard": tims_scoreboard or [],
        "tims_scoreboard_meta": tims_scoreboard_meta or {},
        "slate_top_scorers": skater_scores["slate_top_scorers"],
        "games": game_predictions,
        "power_rankings": power_rankings,
        "playoff_series": playoff_series,
        "calls_summary": calls_summary if calls_summary is not None else {},
        "calls_recent": calls_recent if calls_recent is not None else [],
        "calls_admin": calls_admin if calls_admin is not None else {},
        "tims_board_players": tims_board_players if tims_board_players is not None else [],
        "feature_importance": {
            "skater": skater_bundle.get("feature_importance", []),
            "pairwise": pairwise_bundle.get("feature_importance", []),
        },
    }
