#!/usr/bin/env python3
"""Fail if admin-only odds/ledger data is present in the public docs payload."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_builder import PUBLIC_DROP_GAME_KEYS

TEAM_BETTING_KEYS = frozenset({
    "best_odds", "best_bookie", "fair_odds", "edge_pct", "kelly_pct",
    "rec_bet", "market_cup_pct", "blended_cup_pct",
})
SUMMARY_BETTING_KEYS = frozenset({
    "bankroll", "devig_source", "kelly_fraction", "market_overround_pct",
    "market_weight", "min_edge_pct", "model_weight", "recommended_bets",
    "recommended_count", "total_rec_stake",
})


def check_docs(root: str | Path = ".") -> list[str]:
    docs = Path(root) / "docs"
    errors: list[str] = []
    if (docs / "calls_ledger.json").exists():
        errors.append("docs/calls_ledger.json must not be published")
    if (Path(root) / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--", "docs/latest_run.html"],
            cwd=str(root), capture_output=True, text=True,
        )
        if tracked.stdout.strip():
            errors.append("docs/latest_run.html must not be tracked (admin log may contain odds)")
    data_path = docs / "data.json"
    if not data_path.exists():
        return [f"missing {data_path}"]
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid docs/data.json: {exc}"]
    if "calls_admin" in data:
        errors.append("docs/data.json contains calls_admin")
    meta = data.get("meta")
    if isinstance(meta, dict) and "odds_usage" in meta:
        errors.append("docs/data.json contains meta.odds_usage")
    for game in data.get("games") or []:
        if not isinstance(game, dict):
            continue
        for key in PUBLIC_DROP_GAME_KEYS:
            if key in game:
                errors.append(f"game {game.get('id')} contains public-banned key {key}")

    cf_path = docs / "cup_futures.json"
    if not cf_path.exists():
        errors.append(f"missing {cf_path}")
    else:
        try:
            cf = json.loads(cf_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid docs/cup_futures.json: {exc}")
            cf = None
        if isinstance(cf, dict):
            cf_meta = cf.get("meta")
            if isinstance(cf_meta, dict):
                if cf_meta.get("odds_enabled"):
                    errors.append("docs/cup_futures.json meta.odds_enabled must be false")
            for team in cf.get("teams") or []:
                if not isinstance(team, dict):
                    continue
                for key in TEAM_BETTING_KEYS:
                    if key in team:
                        errors.append(f"cup futures team contains banned key {key}")
            summary = cf.get("summary")
            if isinstance(summary, dict):
                for key in SUMMARY_BETTING_KEYS:
                    if key in summary:
                        errors.append(f"cup futures summary contains banned key {key}")

    return errors


def main() -> int:
    errors = check_docs(".")
    if errors:
        print("Public docs check FAILED:")
        for err in errors:
            print(" -", err)
        return 1
    print("Public docs check passed: no admin ledger or odds keys in docs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
