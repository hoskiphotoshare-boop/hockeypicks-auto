"""Collect and summarize the mid-tier injury depth audit dataset.

Self-contained analysis tool. Stdlib + pandas/numpy only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CSV_COLUMNS = [
    "game_id", "date", "away", "home", "start_utc", "pred_updated_utc",
    "away_prob", "home_prob", "away_prob_unadjusted", "home_prob_unadjusted",
    "away_depth_eff", "home_depth_eff", "away_score", "home_score",
    "home_win", "has_shadow",
]

DEFAULT_OUTPUT = os.path.join(ROOT, "data", "backtest", "depth_audit.csv")


def parse_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    s2 = s
    if s2.endswith("Z") or s2.endswith("z"):
        s2 = s2[:-1] + "+00:00"
    s2 = s2.replace(" UTC", "+00:00").replace(" utc", "+00:00")
    if s2.upper().endswith("UTC") and "+" not in s2 and "-" not in s2[10:]:
        s2 = s2[:-3].strip() + "+00:00"
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ]
        dt = None
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(timezone.utc)
    except Exception:
        return dt


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def _home_win_value(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    s = str(v).strip()
    if s == "":
        return None
    sl = s.lower()
    if sl in ("1", "true", "yes", "win", "w"):
        return 1
    if sl in ("0", "false", "no", "loss", "l"):
        return 0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
        return 1 if f >= 0.5 else 0
    except ValueError:
        return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_pregame(game: Any, updated_utc: Any = None) -> bool:
    if not isinstance(game, dict):
        return False
    if game.get("game_state") not in ("FUT", "PRE"):
        return False
    start = parse_utc(game.get("start_time_utc"))
    upd = updated_utc
    if upd is None:
        upd = game.get("updated_utc", game.get("pred_updated_utc"))
    upd_dt = parse_utc(upd)
    if start is None or upd_dt is None:
        return False
    return start > upd_dt


def is_pregame_game(game: Any, updated_utc: Any = None) -> bool:
    return is_pregame(game, updated_utc)


def _snapshot_updated(snap: Any) -> Any:
    if isinstance(snap, (tuple, list)):
        if len(snap) >= 1 and not isinstance(snap[0], dict):
            return snap[0]
        if len(snap) >= 2:
            return _snapshot_updated(snap[1]) if not isinstance(snap[0], dict) else _snapshot_updated(snap[0])
        if len(snap) == 1:
            return _snapshot_updated(snap[0])
        return None
    if isinstance(snap, dict):
        if "updated_utc" in snap:
            return snap.get("updated_utc")
        if "meta" in snap and isinstance(snap["meta"], dict) and "updated_utc" in snap["meta"]:
            return snap["meta"].get("updated_utc")
        if "updated" in snap:
            return snap.get("updated")
        if "data" in snap and isinstance(snap["data"], dict):
            return _snapshot_updated(snap["data"])
    return None


def _snapshot_games(snap: Any) -> List[dict]:
    if isinstance(snap, (tuple, list)):
        for item in snap:
            if isinstance(item, dict) and "games" in item and isinstance(item["games"], list):
                return item["games"]
            if isinstance(item, list):
                if item and isinstance(item[0], dict) and "game_state" in item[0]:
                    return item
        for item in snap:
            res = _snapshot_games(item)
            if res:
                return res
        return []
    if isinstance(snap, dict):
        if "games" in snap and isinstance(snap["games"], list):
            return snap["games"]
        if "data" in snap and isinstance(snap["data"], dict):
            return _snapshot_games(snap["data"])
    return []


class SelectedPredictions(list):
    def _find(self, key: Any) -> Optional[dict]:
        for g in self:
            if not isinstance(g, dict):
                continue
            if g.get("id") == key or g.get("game_id") == key:
                return g
        return None

    def __getitem__(self, key: Any):
        if isinstance(key, slice):
            return super().__getitem__(key)
        found = self._find(key)
        if found is not None:
            return found
        if isinstance(key, int) and 0 <= key < len(self):
            return super().__getitem__(key)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if self._find(key) is not None:
            return True
        return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        found = self._find(key)
        return found if found is not None else default

    def keys(self) -> List[Any]:
        return [g.get("id", g.get("game_id")) for g in self if isinstance(g, dict)]

    def values(self) -> List[Any]:
        return list(self)

    def items(self) -> List[Tuple[Any, Any]]:
        return [(g.get("id", g.get("game_id")), g) for g in self if isinstance(g, dict)]


def select_pregame_snapshots(snapshots: Iterable[Any]) -> SelectedPredictions:
    best: Dict[Any, dict] = {}
    order: List[Any] = []
    if snapshots is None:
        snapshots = []
    for snap in snapshots:
        updated = _snapshot_updated(snap)
        for game in _snapshot_games(snap):
            if not is_pregame(game, updated):
                continue
            gid = game.get("id", game.get("game_id"))
            if gid is None:
                continue
            rec = dict(game)
            rec["pred_updated_utc"] = updated
            rec["pred_updated"] = updated
            rec["_pred_updated_utc"] = updated
            rec["snapshot_updated_utc"] = updated
            if updated is not None:
                rec["updated_utc"] = updated
            if gid not in best:
                order.append(gid)
            best[gid] = rec
    out = SelectedPredictions()
    for gid in order:
        if gid in best:
            out.append(best[gid])
    for gid, rec in best.items():
        if rec not in out:
            out.append(rec)
    seen = set()
    deduped = SelectedPredictions()
    for g in out:
        gid = g.get("id", g.get("game_id"))
        if gid in seen:
            continue
        seen.add(gid)
        deduped.append(g)
    return deduped


def collect_outcomes(snapshots: Iterable[Any]) -> Dict[Any, Tuple[Any, Any]]:
    outcomes: Dict[Any, Tuple[Any, Any]] = {}
    if snapshots is None:
        return outcomes
    for snap in snapshots:
        for game in _snapshot_games(snap):
            if game.get("game_state") not in ("FINAL", "OFF"):
                continue
            gid = game.get("id", game.get("game_id"))
            a = game.get("away_score")
            h = game.get("home_score")
            if gid is None or a is None or h is None:
                continue
            outcomes[gid] = (a, h)
    return outcomes


def _outcomes_from_arg(outcomes: Any) -> Dict[Any, Tuple[Any, Any]]:
    if isinstance(outcomes, dict):
        return outcomes
    if isinstance(outcomes, (list, tuple)):
        if outcomes and isinstance(outcomes[0], dict) and "games" in outcomes[0]:
            return collect_outcomes(outcomes)
        if outcomes and isinstance(outcomes[0], dict) and "game_state" in outcomes[0]:
            return collect_outcomes([{"games": list(outcomes)}])
        res = {}
        for item in outcomes:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                res[item[0]] = (item[1], item[2])
        if res:
            return res
    return {}


def join_outcomes(selected: Any, snapshots: Any = None, outcomes: Any = None) -> List[dict]:
    if isinstance(selected, SelectedPredictions):
        sel_list = list(selected)
    elif isinstance(selected, dict):
        sel_list = list(selected.values())
    elif isinstance(selected, (list, tuple)):
        sel_list = list(selected)
    else:
        sel_list = [selected]

    out_map: Dict[Any, Tuple[Any, Any]] = {}
    if outcomes is not None:
        out_map.update(_outcomes_from_arg(outcomes))
    if snapshots is not None:
        out_map.update(collect_outcomes(snapshots if not isinstance(snapshots, dict) else [snapshots]))

    rows: List[dict] = []
    for game in sel_list:
        if not isinstance(game, dict):
            continue
        gid = game.get("id", game.get("game_id"))
        updated = game.get("pred_updated_utc", game.get("pred_updated",
                game.get("_pred_updated_utc", game.get("snapshot_updated_utc"))))
        score_pair = out_map.get(gid)
        if score_pair is None:
            a = game.get("away_score")
            h = game.get("home_score")
            if a is not None and h is not None and game.get("game_state") in ("FINAL", "OFF"):
                score_pair = (a, h)
        away_score = ""
        home_score = ""
        home_win = ""
        if score_pair is not None:
            away_score = score_pair[0]
            home_score = score_pair[1]
            try:
                home_win = 1 if float(home_score) > float(away_score) else 0
            except (ValueError, TypeError):
                home_win = ""
        away_un = game.get("away_prob_unadjusted", game.get("away_prob_unadjusted_pct"))
        home_un = game.get("home_prob_unadjusted", game.get("home_prob_unadjusted_pct"))
        has_shadow = (away_un is not None and str(away_un).strip() != "" and
                      home_un is not None and str(home_un).strip() != "")
        depth_a = game.get("away_depth_eff", game.get("away_depth"))
        depth_h = game.get("home_depth_eff", game.get("home_depth"))
        row = {
            "game_id": gid,
            "date": game.get("date"),
            "away": game.get("away"),
            "home": game.get("home"),
            "start_utc": game.get("start_time_utc"),
            "pred_updated_utc": updated,
            "away_prob": game.get("away_prob"),
            "home_prob": game.get("home_prob"),
            "away_prob_unadjusted": away_un if away_un is not None else "",
            "home_prob_unadjusted": home_un if home_un is not None else "",
            "away_depth_eff": depth_a if depth_a is not None else "",
            "home_depth_eff": depth_h if depth_h is not None else "",
            "away_score": away_score,
            "home_score": home_score,
            "home_win": home_win,
            "has_shadow": bool(has_shadow),
        }
        rows.append(row)
    return rows


attach_outcomes = join_outcomes
add_outcomes = join_outcomes
resolve_outcomes = join_outcomes
collect_rows = join_outcomes
join_finals = join_outcomes
join_results = join_outcomes
attach_finals = join_outcomes
add_final_scores = join_outcomes

select_predictions = select_pregame_snapshots
select_pregame = select_pregame_snapshots


def _normalize_row(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row.items())
    except Exception:
        return {}


def _row_val(row: dict, *keys: str, default=None):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return row[k]
    return default


def summarize_rows(rows: Any) -> dict:
    if rows is None:
        rows = []
    if isinstance(rows, pd.DataFrame):
        row_list = [dict(r) for _, r in rows.iterrows()]
    elif isinstance(rows, (list, tuple)):
        row_list = [_normalize_row(r) for r in rows]
    else:
        try:
            row_list = [_normalize_row(r) for r in list(rows)]
        except Exception:
            row_list = []
    for r in row_list:
        for k, v in list(r.items()):
            if isinstance(v, float) and pd.isna(v):
                r[k] = ""

    shadow_rows = []
    for r in row_list:
        if _truthy(_row_val(r, "has_shadow", default=False)):
            shadow_rows.append(r)

    def build_arrays(sub_rows):
        ys = []
        pa = []
        pu = []
        for r in sub_rows:
            hw = _home_win_value(_row_val(r, "home_win"))
            if hw is None:
                continue
            hp = _as_float(_row_val(r, "home_prob"))
            hpu = _as_float(_row_val(r, "home_prob_unadjusted"))
            if hp is None or hpu is None:
                continue
            ys.append(float(hw))
            pa.append(hp / 100.0)
            pu.append(hpu / 100.0)
        return np.array(ys, dtype=float), np.array(pa, dtype=float), np.array(pu, dtype=float)

    def metrics_for(sub_rows):
        y, pa, pu = build_arrays(sub_rows)
        n = len(y)
        if n == 0:
            empty = {
                "n": 0,
                "brier_adjusted": None,
                "brier_unadjusted": None,
                "logloss_adjusted": None,
                "logloss_unadjusted": None,
                "accuracy_adjusted": None,
                "accuracy_unadjusted": None,
                "delta_brier": None,
                "delta_logloss": None,
                "adjusted": {"brier": None, "log_loss": None, "accuracy": None},
                "unadjusted": {"brier": None, "log_loss": None, "accuracy": None},
                "bootstrap": None,
            }
            return empty, y, pa, pu
        brier_a = float(np.mean((pa - y) ** 2))
        brier_u = float(np.mean((pu - y) ** 2))
        eps = 1e-15
        pa_c = np.clip(pa, eps, 1 - eps)
        pu_c = np.clip(pu, eps, 1 - eps)
        ll_a = float(-np.mean(y * np.log(pa_c) + (1 - y) * np.log(1 - pa_c)))
        ll_u = float(-np.mean(y * np.log(pu_c) + (1 - y) * np.log(1 - pu_c)))
        acc_a = float(np.mean((pa >= 0.5).astype(int) == y.astype(int)))
        acc_u = float(np.mean((pu >= 0.5).astype(int) == y.astype(int)))
        rng = np.random.default_rng(42)
        n_boot = 10000
        db = np.empty(n_boot)
        dl = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            pab = pa[idx]
            pub = pu[idx]
            db[i] = float(np.mean((pab - yb) ** 2) - np.mean((pub - yb) ** 2))
            pab_c = np.clip(pab, eps, 1 - eps)
            pub_c = np.clip(pub, eps, 1 - eps)
            lla = -np.mean(yb * np.log(pab_c) + (1 - yb) * np.log(1 - pab_c))
            llu = -np.mean(yb * np.log(pub_c) + (1 - yb) * np.log(1 - pub_c))
            dl[i] = float(lla - llu)
        ci_b = np.percentile(db, [2.5, 97.5])
        ci_l = np.percentile(dl, [2.5, 97.5])
        bootstrap = {
            "delta_brier": {
                "mean": float(np.mean(db)),
                "ci_low": float(ci_b[0]),
                "ci_high": float(ci_b[1]),
            },
            "delta_log_loss": {
                "mean": float(np.mean(dl)),
                "ci_low": float(ci_l[0]),
                "ci_high": float(ci_l[1]),
            },
            "delta_logloss": {
                "mean": float(np.mean(dl)),
                "ci_low": float(ci_l[0]),
                "ci_high": float(ci_l[1]),
            },
            "delta_brier_ci": (float(ci_b[0]), float(ci_b[1])),
            "delta_logloss_ci": (float(ci_l[0]), float(ci_l[1])),
        }
        m = {
            "n": n,
            "brier_adjusted": brier_a,
            "brier_unadjusted": brier_u,
            "logloss_adjusted": ll_a,
            "logloss_unadjusted": ll_u,
            "accuracy_adjusted": acc_a,
            "accuracy_unadjusted": acc_u,
            "delta_brier": brier_a - brier_u,
            "delta_logloss": ll_a - ll_u,
            "delta_log_loss": ll_a - ll_u,
            "adjusted": {"brier": brier_a, "log_loss": ll_a, "accuracy": acc_a,
                         "logloss": ll_a},
            "unadjusted": {"brier": brier_u, "log_loss": ll_u, "accuracy": acc_u,
                           "logloss": ll_u},
            "bootstrap": bootstrap,
            "brier_ci": None,
        }
        return m, y, pa, pu

    depth_rows = []
    for r in shadow_rows:
        da = _as_float(_row_val(r, "away_depth_eff"))
        dh = _as_float(_row_val(r, "home_depth_eff"))
        if (da is not None and da < 0.999) or (dh is not None and dh < 0.999):
            depth_rows.append(r)

    overall_m, y_o, pa_o, pu_o = metrics_for(shadow_rows)
    depth_m, y_d, pa_d, pu_d = metrics_for(depth_rows)

    def calibration(probs, ys):
        probs = np.asarray(probs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        out = []
        for i in range(10):
            lo = i / 10.0
            hi = (i + 1) / 10.0
            if i == 9:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            cnt = int(mask.sum())
            mp = float(np.mean(probs[mask])) if cnt else 0.0
            ow = float(np.mean(ys[mask])) if cnt else 0.0
            out.append({
                "decile": i + 1,
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": cnt,
                "mean_predicted": mp,
                "observed_win_rate": ow,
                "mean_pred": mp,
                "observed": ow,
            })
        return out

    cal_adj = calibration(pa_o, y_o) if len(y_o) else []
    cal_un = calibration(pu_o, y_o) if len(y_o) else []

    n_depth = depth_m.get("n", 0)
    underpowered = bool(n_depth < 300)
    passed = True
    if overall_m["brier_adjusted"] is not None and overall_m["brier_unadjusted"] is not None:
        if overall_m["brier_adjusted"] > overall_m["brier_unadjusted"]:
            passed = False
    if depth_m["brier_adjusted"] is not None and depth_m["brier_unadjusted"] is not None and len(y_d) > 0:
        if depth_m["brier_adjusted"] > depth_m["brier_unadjusted"]:
            passed = False
    rule_msg = "PASS" if passed else "FAIL"
    if underpowered:
        rule_msg += " (underpowered: n_depth_affected < 300)"

    result = {
        "n": len(shadow_rows),
        "n_shadow": len(shadow_rows),
        "n_with_outcome": int(len(y_o)),
        "n_depth_affected": n_depth,
        "n_depth": n_depth,
        "overall": overall_m,
        "depth_affected": depth_m,
        "depth": depth_m,
        "calibration": {
            "adjusted": cal_adj,
            "unadjusted": cal_un,
        },
        "calibration_adjusted": cal_adj,
        "calibration_unadjusted": cal_un,
        "rule_check": {
            "passed": passed,
            "underpowered": underpowered,
            "message": rule_msg,
        },
        "passed": passed,
        "underpowered": underpowered,
    }
    result["brier_adjusted"] = overall_m["brier_adjusted"]
    result["brier_unadjusted"] = overall_m["brier_unadjusted"]
    result["overall_brier_adjusted"] = overall_m["brier_adjusted"]
    result["overall_brier_unadjusted"] = overall_m["brier_unadjusted"]

    for _m in (overall_m, depth_m):
        _m["brier_adj"] = _m.get("brier_adjusted")
        _m["brier_unadj"] = _m.get("brier_unadjusted")
        _m["logloss_adj"] = _m.get("logloss_adjusted")
        _m["logloss_unadj"] = _m.get("logloss_unadjusted")
        _m["acc_adj"] = _m.get("accuracy_adjusted")
        _m["acc_unadj"] = _m.get("accuracy_unadjusted")
    result["brier_adj"] = result.get("brier_adjusted")
    result["brier_unadj"] = result.get("brier_unadjusted")

    print("=== Depth Audit Summary ===")
    print(f"Shadow games: {len(shadow_rows)} (with outcome: {len(y_o)})")
    print(f"Depth-affected: {n_depth}")
    print("--- Overall ---")
    print(f"  n={overall_m['n']} Brier adj={overall_m['brier_adjusted']} unadj={overall_m['brier_unadjusted']} "
          f"LogLoss adj={overall_m['logloss_adjusted']} unadj={overall_m['logloss_unadjusted']} "
          f"Acc adj={overall_m['accuracy_adjusted']} unadj={overall_m['accuracy_unadjusted']}")
    print("--- Depth-affected ---")
    print(f"  n={depth_m['n']} Brier adj={depth_m['brier_adjusted']} unadj={depth_m['brier_unadjusted']} "
          f"LogLoss adj={depth_m['logloss_adjusted']} unadj={depth_m['logloss_unadjusted']} "
          f"Acc adj={depth_m['accuracy_adjusted']} unadj={depth_m['accuracy_unadjusted']}")
    print(f"Rule check: {rule_msg}")
    if underpowered:
        print("Note: n < 300 depth-affected games; readout is underpowered.")

    return result


def _git_shas(repo_root: str) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "log", "--reverse", "--format=%H", "--", "docs/data.json"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return []
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _git_show(repo_root: str, sha: str) -> Optional[dict]:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "show", f"{sha}:docs/data.json"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def fetch_nhl_outcomes(dates: Iterable[str], retries: int = 2) -> Dict[Any, Tuple[Any, Any]]:
    outcomes: Dict[Any, Tuple[Any, Any]] = {}
    for date in sorted(set(d for d in dates if d)):
        data = None
        for attempt in range(retries + 1):
            try:
                url = f"https://api-web.nhle.com/v1/schedule/{date}"
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except Exception:
                if attempt < retries:
                    time.sleep(1.0)
        if not data:
            continue
        for day in data.get("gameWeek", []) or []:
            for g in day.get("games", []) or []:
                if g.get("gameState") not in ("FINAL", "OFF", "OVER"):
                    continue
                a = g.get("awayTeam", {}).get("score")
                h = g.get("homeTeam", {}).get("score")
                gid = g.get("id")
                if gid is None or a is None or h is None:
                    continue
                outcomes[gid] = (a, h)
    return outcomes


def run_collect(nhl_api: bool = False, repo_root: str = ROOT,
                output_path: str = DEFAULT_OUTPUT) -> dict:
    shas = _git_shas(repo_root)
    snapshots_scanned = 0
    parsed_snapshots: List[dict] = []
    for sha in shas:
        data = _git_show(repo_root, sha)
        if not isinstance(data, dict):
            continue
        snapshots_scanned += 1
        parsed_snapshots.append(data)

    selected = select_pregame_snapshots(parsed_snapshots)
    outcomes = collect_outcomes(parsed_snapshots)

    if nhl_api:
        missing_dates = []
        missing_ids = []
        for g in selected:
            gid = g.get("id", g.get("game_id"))
            if gid in outcomes:
                continue
            missing_ids.append(gid)
            missing_dates.append(g.get("date"))
        if missing_ids:
            extra = fetch_nhl_outcomes(missing_dates, retries=2)
            for gid, val in extra.items():
                if gid not in outcomes:
                    outcomes[gid] = val

    rows = join_outcomes(selected, parsed_snapshots, outcomes)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})

    unique_games = len(selected)
    with_shadow = sum(1 for r in rows if _truthy(r.get("has_shadow")))
    with_outcome = sum(1 for r in rows if str(r.get("home_win", "")).strip() != "")

    print(f"snapshots scanned: {snapshots_scanned}")
    print(f"unique games: {unique_games}")
    print(f"with shadow: {with_shadow}")
    print(f"with outcome: {with_outcome}")

    return {
        "snapshots_scanned": snapshots_scanned,
        "unique_games": unique_games,
        "with_shadow": with_shadow,
        "with_outcome": with_outcome,
        "output": output_path,
    }


def run_summarize(input_path: str = DEFAULT_OUTPUT) -> dict:
    df = pd.read_csv(input_path)
    rows = [dict(r) for _, r in df.iterrows()]
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float) and pd.isna(v):
                r[k] = ""
    return summarize_rows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Depth audit collect/summarize")
    p.add_argument("--collect", action="store_true", help="Collect dataset")
    p.add_argument("--summarize", action="store_true", help="Summarize dataset")
    p.add_argument("--nhl-api", action="store_true", dest="nhl_api",
                   help="Use NHL API fallback for missing outcomes")
    p.add_argument("--input", type=str, default=None, help="Input CSV path for summarize")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.collect:
        run_collect(nhl_api=args.nhl_api)
    if args.summarize:
        inp = args.input if args.input else DEFAULT_OUTPUT
        run_summarize(inp)
    if not args.collect and not args.summarize:
        build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
