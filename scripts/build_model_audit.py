#!/usr/bin/env python3
"""Build docs/model_audit.json from walk-forward Tims backtest artifacts.

The audit answers one question: does the shipped Tims pick path beat simple
baselines (season goals/GP, prior-season goals/GP, one-line empirical Bayes)?
It pools every eligible slate, reports paired bootstrap intervals and exact
McNemar tests, and evaluates a pre-registered ship rule.

Usage:
    python scripts/build_model_audit.py \
        --report data/backtest/tims_backtest_report.json \
        --slates data/backtest/tims_slates.json \
        --out docs/model_audit.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

STRATEGY_META: dict[str, dict[str, str]] = {
    "ai_full": {
        "label": "Shipped model",
        "detail": "XGBoost rate + empirical Bayes + opponent defense multiplier",
        "role": "model",
    },
    "xgb_eb": {
        "label": "XGB + empirical Bayes",
        "detail": "Model rate shrunk toward the prior, no defense multiplier",
        "role": "model",
    },
    "xgb_only": {
        "label": "XGB only",
        "detail": "Raw model rate, no shrinkage",
        "role": "model",
    },
    "emp_blend": {
        "label": "Empirical Bayes (2-line)",
        "detail": "(prior goals + live goals + 1.4) / (prior GP + live GP + 10)",
        "role": "simple",
    },
    "naive_live": {
        "label": "Season goals/GP",
        "detail": "Season goals per game, prior-season rate below 3 GP",
        "role": "simple",
    },
    "naive_prior": {
        "label": "Prior-season goals/GP",
        "detail": "Last season's goals per game only",
        "role": "simple",
    },
    "season_gpg": {
        "label": "Season goals/GP (any GP)",
        "detail": "Pure season goals/game, no games-played floor",
        "role": "simple",
    },
    "season_gpg10": {
        "label": "Season goals/GP (10+ GP)",
        "detail": "Season goals/game once a player reaches 10 games",
        "role": "simple",
    },
    "blend_50": {
        "label": "50/50 prior + model",
        "detail": "0.5 x prior-season goals/GP + 0.5 x shipped model rate",
        "role": "blend",
    },
    "blend_prior_xgb": {
        "label": "50/50 prior + XGB",
        "detail": "0.5 x prior-season goals/GP + 0.5 x raw XGB rate (no EB/defense)",
        "role": "blend",
    },
}

KEY_SECONDARIES = ("emp_blend", "naive_prior", "season_gpg", "naive_live")


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> list[float]:
    n = len(values)
    if n == 0:
        return [0.0, 0.0]
    idx = rng.integers(0, n, size=(draws, n))
    boots = values[idx].mean(axis=1)
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def _mcnemar_exact(b: int, c: int) -> float:
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return round(float(min(1.0, 2.0 * tail)), 4)


def _paired(model_vals: np.ndarray, base_vals: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Paired comparison of two per-slate hit series (model minus baseline)."""
    n = int(len(model_vals))
    if n == 0 or len(base_vals) != n:
        return {"slates": n}
    diff = model_vals - base_vals
    a1, b1 = model_vals >= 1, base_vals >= 1
    a2, b2 = model_vals >= 2, base_vals >= 2
    return {
        "slates": n,
        "mean_hits_diff": round(float(diff.mean()), 4),
        "mean_hits_ci95": _bootstrap_mean_ci(diff, rng),
        "hit1plus_diff_pp": round(float(a1.mean() - b1.mean()) * 100.0, 2),
        "hit1plus_ci95_pp": [round(v * 100.0, 2) for v in _bootstrap_mean_ci(a1.astype(float) - b1.astype(float), rng)],
        "hit1plus_mcnemar": {"model_only": int(np.sum(a1 & ~b1)), "baseline_only": int(np.sum(~a1 & b1))},
        "hit1plus_mcnemar_p": _mcnemar_exact(int(np.sum(a1 & ~b1)), int(np.sum(~a1 & b1))),
        "hit2plus_diff_pp": round(float(a2.mean() - b2.mean()) * 100.0, 2),
        "hit2plus_ci95_pp": [round(v * 100.0, 2) for v in _bootstrap_mean_ci(a2.astype(float) - b2.astype(float), rng)],
        "model_wins": int(np.sum(diff > 0)),
        "baseline_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }


def _summ(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"hit_1plus_pct": 0.0, "hit_2plus_pct": 0.0, "sweep_3_pct": 0.0, "mean_hits": 0.0}
    return {
        "hit_1plus_pct": round(float(np.mean(values >= 1)) * 100.0, 2),
        "hit_2plus_pct": round(float(np.mean(values >= 2)) * 100.0, 2),
        "sweep_3_pct": round(float(np.mean(values == 3)) * 100.0, 2),
        "mean_hits": round(float(np.mean(values)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build docs/model_audit.json from Tims backtest artifacts.")
    parser.add_argument("--report", default="data/backtest/tims_backtest_report.json")
    parser.add_argument("--slates", default="data/backtest/tims_slates.json")
    parser.add_argument("--out", default="data/backtest/model_audit.json")
    parser.add_argument("--baseline-report", default="", help="Optional second report (e.g. blend-grouped) for sensitivity.")
    parser.add_argument("--baseline-slates", default="")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    slates = json.loads(Path(args.slates).read_text())
    results = [r for r in report.get("results", []) if "error" not in r]

    strategies = [s for s in STRATEGY_META if s in (results[0].get("tickets", {}) if results else {})]
    rng = np.random.default_rng(13)

    per_slate: dict[str, np.ndarray] = {}
    top1: dict[str, np.ndarray] = {}
    deviance: dict[str, np.ndarray] = {}
    for s in strategies:
        per_slate[s] = np.asarray([int(row["hits"][s]) for row in slates if s in row.get("hits", {})], dtype=float)
        top1[s] = np.asarray([int(row.get("top1", {}).get(s, 0)) for row in slates if s in row.get("hits", {})], dtype=float)
        deviance[s] = np.asarray([float(row["deviance"][s]) for row in slates if s in row.get("deviance", {})], dtype=float)

    pooled: dict[str, dict[str, Any]] = {}
    utility: dict[str, dict[str, Any]] = {}
    for s in strategies:
        tickets = _summ(per_slate[s])
        tickets["top1_pct"] = round(float(np.mean(top1[s])) * 100.0, 2) if len(top1[s]) else 0.0
        tickets["poisson_deviance"] = round(float(np.mean(deviance[s])), 4) if len(deviance[s]) else 0.0
        pooled[s] = tickets
        vals = per_slate[s]
        u1, u2, u3 = [float(np.mean(vals == k)) for k in (1, 2, 3)]
        p1plus = float(np.mean(vals >= 1))
        utility[s] = {
            "p_exactly_1": round(u1, 4),
            "p_exactly_2": round(u2, 4),
            "p_exactly_3": round(u3, 4),
            "p_1plus": round(p1plus, 4),
            "expected_points_per_day": round(10.0 * u1 + 25.0 * u2 + 100.0 * u3, 4),
            "implied_7day_streak_pct": round((p1plus ** 7) * 100.0, 2),
            "expected_leaderboard_points_per_day": round(float(np.mean(vals)), 4),
        }
    best_by_objective = {
        "streak_p_1plus": max(utility, key=lambda s: utility[s]["p_1plus"]),
        "daily_points": max(utility, key=lambda s: utility[s]["expected_points_per_day"]),
        "leaderboard": max(utility, key=lambda s: utility[s]["expected_leaderboard_points_per_day"]),
    }

    paired = {base: _paired(per_slate["ai_full"], per_slate[base], rng) for base in strategies if base != "ai_full"}
    dev_paired = {}
    for base in strategies:
        if base == "ai_full":
            continue
        diff = deviance["ai_full"] - deviance[base]
        dev_paired[base] = {
            "mean_deviance_diff": round(float(diff.mean()), 4),
            "ci95": _bootstrap_mean_ci(diff, rng),
            "model_better_slates": int(np.sum(diff < 0)),
            "baseline_better_slates": int(np.sum(diff > 0)),
        }

    simple = [s for s in strategies if STRATEGY_META[s]["role"] == "simple"]
    best_simple = max(simple, key=lambda s: pooled[s]["mean_hits"]) if simple else "emp_blend"

    # Pre-registered ship rule, evaluated against the best simple baseline.
    cmp_main = paired[best_simple]
    cmp_dev = dev_paired[best_simple]
    rule = [
        {
            "name": "Ticket volume is not worse",
            "threshold": "mean hits paired 95% CI lower bound ≥ −0.01 and 1+ hit CI lower bound ≥ −1.0 pp",
            "observed": f"mean hits {cmp_main['mean_hits_diff']:+.3f} CI {cmp_main['mean_hits_ci95']}; "
            f"1+ hit {cmp_main['hit1plus_diff_pp']:+.2f} pp CI {cmp_main['hit1plus_ci95_pp']}",
            "pass": bool(cmp_main["mean_hits_ci95"][0] >= -0.01 and cmp_main["hit1plus_ci95_pp"][0] >= -1.0),
        },
        {
            "name": "At least one primary metric wins",
            "threshold": "2+ hit or slate top-1 paired 95% CI lower bound > 0",
            "observed": f"2+ hit {cmp_main['hit2plus_diff_pp']:+.2f} pp CI {cmp_main['hit2plus_ci95_pp']}; "
            f"top-1 {round(float((top1['ai_full'] - top1[best_simple]).mean() * 100.0), 2):+.2f} pp",
            "pass": bool(cmp_main["hit2plus_ci95_pp"][0] > 0),
        },
        {
            "name": "Player-level fit is better",
            "threshold": "Poisson deviance paired 95% CI upper bound < 0",
            "observed": f"deviance {cmp_dev['mean_deviance_diff']:+.4f} CI {cmp_dev['ci95']}",
            "pass": bool(cmp_dev["ci95"][1] < 0),
        },
    ]
    rule_pass = all(r["pass"] for r in rule)

    # Month slices for the shipped model and key baselines.
    month_slices = []
    month_keys: dict[str, list[int]] = {}
    for i, row in enumerate([r for r in slates if "hits" in r]):
        month_keys.setdefault(str(row["date"])[:7], []).append(i)
    for month in sorted(month_keys):
        idx = month_keys[month]
        entry: dict[str, Any] = {"month": month, "slates": len(idx)}
        for s in ("ai_full",) + KEY_SECONDARIES:
            if s in per_slate:
                vals = per_slate[s][idx]
                entry[s] = {
                    "hit_1plus_pct": round(float(np.mean(vals >= 1)) * 100.0, 1),
                    "mean_hits": round(float(np.mean(vals)), 3),
                }
        month_slices.append(entry)

    seasons = []
    for res in results:
        seasons.append(
            {
                "label": res.get("label"),
                "slates": res.get("tims_3tier_slates"),
                "games": res.get("games"),
                "skater_appearances": res.get("skater_appearances"),
                "tickets": res.get("tickets", {}),
                "slate_top_picks": res.get("slate_top_picks", {}),
                "paired_vs_ai_full": res.get("paired_vs_ai_full", {}),
            }
        )

    sensitivity = None
    if args.baseline_report and args.baseline_slates:
        try:
            s_report = json.loads(Path(args.baseline_report).read_text())
            s_slates = json.loads(Path(args.baseline_slates).read_text())
            s_results = [r for r in s_report.get("results", []) if "error" not in r]
            s_pool: dict[str, Any] = {}
            for s in ("ai_full", best_simple):
                vals = np.asarray([int(r["hits"][s]) for r in s_slates if s in r.get("hits", {})], dtype=float)
                s_pool[s] = _summ(vals)
            sensitivity = {
                "group_sort": s_report.get("group_sort", "blend"),
                "slates": len(s_slates),
                "tickets": s_pool,
                "note": "Groups rebuilt by empirical-Bayes rate instead of neutral TOI; shows how much the group rule moves the result.",
            }
        except Exception as exc:  # pragma: no cover - defensive
            sensitivity = {"error": str(exc)}

    total_slates = int(len(per_slate["ai_full"])) if "ai_full" in per_slate else 0
    total_games = int(sum(int(r.get("games") or 0) for r in results))
    total_apps = int(sum(int(r.get("skater_appearances") or 0) for r in results))

    verdict = (
        f"Over {total_slates} slates the shipped model is {'not better' if not rule_pass else 'better'} than the best "
        f"simple baseline ({STRATEGY_META[best_simple]['label']}). "
        f"Mean hits {cmp_main['mean_hits_diff']:+.3f} (95% CI {cmp_main['mean_hits_ci95'][0]:+.3f} to "
        f"{cmp_main['mean_hits_ci95'][1]:+.3f}), 1+ hit {cmp_main['hit1plus_diff_pp']:+.1f} pp, "
        f"2+ hit {cmp_main['hit2plus_diff_pp']:+.1f} pp, player deviance {cmp_dev['mean_deviance_diff']:+.4f}."
    )

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "group_sort": report.get("group_sort", "toi"),
        "totals": {"slates": total_slates, "games": total_games, "skater_appearances": total_apps},
        "strategies": {
            s: {**STRATEGY_META[s], **pooled[s]} for s in strategies
        },
        "best_simple_baseline": best_simple,
        "paired_vs_ai_full": paired,
        "deviance_vs_ai_full": dev_paired,
        "decision_rule": rule,
        "decision_rule_pass": rule_pass,
        "contest_utility": utility,
        "best_by_objective": best_by_objective,
        "verdict": verdict,
        "seasons": seasons,
        "month_slices": month_slices,
        "sensitivity": sensitivity,
        "notes": [
            "Walk-forward only: every slate is scored from pre-puck information; the slate's own results never enter the features.",
            "Groups are built from neutral pre-game TOI unless group_sort says otherwise, so no baseline gets a home-field advantage.",
            "Ticket metrics are one pick per group (three picks per slate) and match the public Tims tracker format.",
            "Paired intervals come from 10,000 slate-level bootstrap resamples; McNemar is exact and two-sided.",
            "Deviance is mean Poisson deviance over every skater-game on the slate, so it scores the whole ranking, not just the picks.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({total_slates} slates, best simple = {best_simple}, rule pass = {rule_pass})")
    print(verdict)
    print("best by objective:", best_by_objective)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
