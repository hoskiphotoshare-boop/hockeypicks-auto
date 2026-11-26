#!/usr/bin/env python3
"""Weekly player-stat recalibration: evaluate recent games and update the small
calibration/interval file without retraining the tree bundle."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import eval_player_stat_models as ep  # noqa: E402

BUNDLE = os.path.join(ROOT, "models", "player_stat_bundles.joblib")
CAL = os.path.join(ROOT, "models", "player_stat_calibration.json")
DRIFT = os.path.join(ROOT, "models", "player_stat_drift.json")
TARGETS = ["goals", "assists", "points", "sog"]


def load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def main() -> None:
    games_path = os.path.join(ROOT, ep.GAMES)
    if not os.path.exists(games_path):
        print("No historical player games cache; nothing to recalibrate.", flush=True)
        return
    import joblib
    bundle = joblib.load(BUNDLE)
    models = bundle.get("models") or {}
    cols = bundle.get("features") or []
    cal_doc = load_json(CAL, {})
    old_cal = cal_doc.get("calibration") or bundle.get("calibration") or {}
    old_k = cal_doc.get("interval_scale") or bundle.get("interval_scale") or {}

    df = pd.read_csv(games_path)
    team = ep.load_team_logs()
    prior = ep.season_rates()
    team_prior = ep.team_prior_rates()
    df = ep.build_features(df, team, prior, team_prior)
    df = df[df["gp_season"] >= 3].copy()
    latest = df["Date"].max()
    recent = df[df["Date"] >= (latest - pd.Timedelta(days=30))].copy()
    if len(recent) < 500:
        recent = df.tail(2000).copy()
    print(f"recent window {recent['Date'].min()} -> {recent['Date'].max()} ({len(recent)} rows)", flush=True)

    new_cal = dict(old_cal)
    new_k = dict(old_k)
    metrics = {}
    for target in TARGETS:
        pred = np.zeros(len(recent), dtype=float)
        q10 = np.zeros(len(recent), dtype=float)
        q90 = np.zeros(len(recent), dtype=float)
        y = recent[target].to_numpy(dtype=float)
        prior_col = "prior_" + ("g" if target == "goals" else "a" if target == "assists" else "p" if target == "points" else "sog") + "_pg"
        for group in ("F", "D"):
            mask = recent["is_defense"] == (1 if group == "D" else 0)
            tr = recent[mask]
            idx = np.where(mask.to_numpy())[0]
            if len(tr) == 0:
                continue
            m = models.get(target, {}).get(group)
            if m is None:
                continue
            X = tr[cols].fillna(0).to_numpy(dtype=float)
            bm = np.log(np.clip(tr[prior_col].fillna(0.05).to_numpy(dtype=float), 0.02, 5.0))
            pred[idx] = np.clip(m["mean"].predict(X, base_margin=bm), 0, None)
            q10[idx] = np.clip(m["q10"].predict(X, base_margin=bm), 0, None)
            q90[idx] = np.clip(m["q90"].predict(X, base_margin=bm), 0, None)
        if not np.any(pred > 0):
            continue
        raw_mean = float(np.mean(pred))
        actual_mean = float(np.mean(y))
        ratio = actual_mean / raw_mean if raw_mean > 0 else 1.0
        prev = float(old_cal.get(target, 1.0) or 1.0)
        blended = float(0.5 * prev + 0.5 * ratio)
        new_cal[target] = round(max(0.8, min(1.2, blended)), 4)
        # interval scale on the recent window
        best_k, best_cov = float(old_k.get(target, 1.0) or 1.0), 0.0
        for k in [round(0.6 + 0.05 * i, 2) for i in range(17)]:
            p = pred * new_cal[target]
            lo = np.maximum(0, p - k * np.maximum(0, pred - q10))
            hi = np.maximum(p, p + k * np.maximum(0, q90 - pred))
            cov = float(np.mean((y >= lo) & (y <= hi)))
            if abs(cov - 0.80) < abs(best_cov - 0.80):
                best_k, best_cov = k, cov
        new_k[target] = round(float(0.5 * float(old_k.get(target, 1.0) or 1.0) + 0.5 * best_k), 3)
        p = pred * new_cal[target]
        lo = np.maximum(0, p - new_k[target] * np.maximum(0, pred - q10))
        hi = np.maximum(p, p + new_k[target] * np.maximum(0, q90 - pred))
        cov = float(np.mean((y >= lo) & (y <= hi)))
        metrics[target] = {
            "rows": int(len(y)),
            "raw_mean": raw_mean,
            "actual_mean": actual_mean,
            "ratio": round(ratio, 4),
            "calibration": new_cal[target],
            "interval_scale": new_k[target],
            "coverage_80": round(cov, 4),
            "mae": round(float(np.mean(np.abs(y - p))), 4),
            "poisson_deviance": round(ep.poisson_deviance(y, p), 4),
        }
        print(target, metrics[target], flush=True)

    now = datetime.now(timezone.utc).isoformat()
    with open(CAL, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "calibration": new_cal,
                "interval_scale": new_k,
                "evaluated_utc": now,
                "source": "weekly_recalibration",
                "window": {
                    "from": str(recent["Date"].min().date()),
                    "to": str(recent["Date"].max().date()),
                    "rows": int(len(recent)),
                },
                "metrics": metrics,
            },
            fh,
            indent=2,
        )
    with open(DRIFT, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "evaluated_utc": now,
                "window": {
                    "from": str(recent["Date"].min().date()),
                    "to": str(recent["Date"].max().date()),
                    "rows": int(len(recent)),
                },
                "metrics": metrics,
            },
            fh,
            indent=2,
        )
    print("wrote", CAL, "and", DRIFT, flush=True)


if __name__ == "__main__":
    main()
