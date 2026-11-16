"""Report generation for structured docs/data.json and styled docs/latest_run.html."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any

import pytz


UTC_TZ = pytz.UTC
ET_TZ = pytz.timezone("America/New_York")


def write_json_report(payload: dict[str, Any], path: str) -> None:
    """Write structured dashboard payload atomically as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
) -> dict[str, Any]:
    """Assemble the complete JSON payload consumed by docs/index.html."""
    now_utc = datetime.now(UTC_TZ)
    now_et = now_utc.astimezone(ET_TZ)

    meta = {
        "status": "live",
        "updated_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
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
        "tims_history": tims_history
        or {"total_days_logged": 0, "settled_days": 0, "pending_days": 0, "recent_days": []},
        "slate_top_scorers": skater_scores["slate_top_scorers"],
        "games": game_predictions,
        "power_rankings": power_rankings,
        "playoff_series": playoff_series,
        "feature_importance": {
            "skater": skater_bundle.get("feature_importance", []),
            "pairwise": pairwise_bundle.get("feature_importance", []),
        },
    }
