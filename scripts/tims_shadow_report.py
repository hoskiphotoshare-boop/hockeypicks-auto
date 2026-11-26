#!/usr/bin/env python3
"""Live shadow report for the Tims pick path.

Every captured day in ``docs/tims_history.json`` stores the full candidate pool
with the model score *and* the prior-season inputs, so each competing strategy
can be reconstructed from the same frozen pre-lock snapshot:

- ``model``     shipped XGB + EB + defense multiplier
- ``blend_50``  0.5 x prior-season goals/GP + 0.5 x model
- ``emp_blend`` (prior goals + live goals + 1.4) / (prior GP + live GP + 10)
- ``prior_gp``  prior-season goals/GP (model fallback for no prior sample)

Grading uses the frozen ``results.player_goals`` map (goals per player on the
slate), so no post-lock information enters the picks. The report also checks
whether a full 30-day window has accumulated.

Usage:
    python scripts/tims_shadow_report.py
    python scripts/tims_shadow_report.py --days 30 --out data/backtest/tims_shadow_report.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

DEFAULT_DAYS = 30
STRATEGIES = ("model", "blend_50", "emp_blend", "prior_gp")
REQUIRED_FIELDS = ("model_score", "prior_goals_per_game", "prior_gp", "prior_goals")


def _score(strategy: str, c: dict[str, Any]) -> float:
    model = float(c.get("model_score", c.get("final_score", 0.0)) or 0.0)
    prior_rate = c.get("prior_goals_per_game")
    prior_rate = float(prior_rate) if prior_rate is not None else None
    prior_gp = float(c.get("prior_gp", 0.0) or 0.0)
    prior_goals = float(c.get("prior_goals", 0.0) or 0.0)
    live_gp = float(c.get("live_gp", 0) or 0)
    live_goals = float(c.get("live_goals", 0) or 0)
    if strategy == "model":
        return model
    if strategy == "prior_gp":
        return prior_rate if prior_rate is not None else model
    if strategy == "blend_50":
        return 0.5 * (prior_rate if prior_rate is not None else model) + 0.5 * model
    if strategy == "emp_blend":
        return (prior_goals + live_goals + 1.4) / (prior_gp + live_gp + 10.0)
    raise ValueError(strategy)


def _neutral(c: dict[str, Any]) -> float:
    prior_gp = float(c.get("prior_gp", 0.0) or 0.0)
    prior_goals = float(c.get("prior_goals", 0.0) or 0.0)
    live_gp = float(c.get("live_gp", 0) or 0)
    live_goals = float(c.get("live_goals", 0) or 0)
    return (prior_goals + live_goals + 1.4) / (prior_gp + live_gp + 10.0)


def _eligible_day(day: dict[str, Any]) -> bool:
    groups = day.get("groups") or {}
    if not groups:
        return False
    for grp in groups.values():
        for c in grp or []:
            if all(f in c for f in REQUIRED_FIELDS):
                return True
    return False


def summarize(history: dict[str, Any], days: int = DEFAULT_DAYS) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for day in sorted(history.get("days", []), key=lambda d: str(d.get("date", ""))):
        if day.get("settled") and _eligible_day(day):
            eligible.append(day)

    window = eligible[-days:] if days > 0 else eligible
    rows: list[dict[str, Any]] = []
    for day in window:
        groups = day.get("groups") or {}
        player_goals = (day.get("results") or {}).get("player_goals") or {}
        row: dict[str, Any] = {"date": day.get("date"), "hits": {}, "picks": {}}
        for strategy in STRATEGIES:
            hits = 0
            picks: list[str] = []
            ok = True
            for key in sorted(groups, key=lambda k: int(k)):
                bucket = [c for c in (groups.get(key) or []) if not c.get("injury_out")]
                if not bucket:
                    ok = False
                    break
                best = max(bucket, key=lambda c: (_score(strategy, c), _neutral(c), str(c.get("player", ""))))
                picks.append(str(best.get("player", "")))
                hits += 1 if int(player_goals.get(str(best.get("player", "")), 0) or 0) >= 1 else 0
            if ok and len(picks) == 3:
                row["hits"][strategy] = hits
                row["picks"][strategy] = picks
        if len(row["hits"]) == len(STRATEGIES):
            row["shipped_hits"] = (day.get("results") or {}).get("hits_of_3")
            rows.append(row)

    summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        vals = [r["hits"][strategy] for r in rows]
        n = len(vals)
        p1 = sum(1 for v in vals if v == 1) / n if n else 0.0
        p2 = sum(1 for v in vals if v == 2) / n if n else 0.0
        p3 = sum(1 for v in vals if v == 3) / n if n else 0.0
        summary[strategy] = {
            "days": n,
            "hit_1plus_pct": round((1.0 - sum(1 for v in vals if v == 0) / n) * 100.0, 2) if n else 0.0,
            "hit_2plus_pct": round(sum(1 for v in vals if v >= 2) / n * 100.0, 2) if n else 0.0,
            "sweep_3_pct": round(p3 * 100.0, 2) if n else 0.0,
            "mean_hits": round(sum(vals) / n, 3) if n else 0.0,
            "expected_points_per_day": round(10.0 * p1 + 25.0 * p2 + 100.0 * p3, 2) if n else 0.0,
        }

    start = str(window[0]["date"]) if window else None
    end = str(window[-1]["date"]) if window else None
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window_days_requested": days,
        "eligible_days": len(eligible),
        "window": {"start": start, "end": end, "days": len(window)},
        "ready": len(window) >= days and days > 0,
        "remaining_days": max(0, days - len(window)) if days > 0 else 0,
        "summary": summary,
        "days": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Live shadow report for Tims pick strategies.")
    parser.add_argument("--history", default="docs/tims_history.json")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--out", default="data/backtest/tims_shadow_report.json")
    args = parser.parse_args()

    history = json.loads(Path(args.history).read_text())
    report = summarize(history, days=args.days)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(
        f"Shadow window: {report['window']['start']} -> {report['window']['end']} "
        f"({report['window']['days']}/{args.days} days, ready={report['ready']}, "
        f"remaining={report['remaining_days']})"
    )
    header = ("strategy", "1+%", "2+%", "sweep%", "mean", "Epts")
    print("  " + "  ".join(f"{h:>10}" for h in header))
    for strategy, s in report["summary"].items():
        print(
            f"  {strategy:>10}  {s['hit_1plus_pct']:>10.1f}  {s['hit_2plus_pct']:>10.1f}"
            f"  {s['sweep_3_pct']:>10.1f}  {s['mean_hits']:>10.3f}  {s['expected_points_per_day']:>10.2f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
