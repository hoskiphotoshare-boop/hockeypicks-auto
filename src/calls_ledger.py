"""Model call ledger: snapshot, close, grade, and summarize public/admin calls."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.infer_gate import parse_utc_timestamp


UTC_TZ = timezone.utc


def _now_iso(now_utc: Any) -> str:
    if hasattr(now_utc, "isoformat"):
        return now_utc.isoformat()
    return str(now_utc)


def _normalize_now(now_utc: Any) -> datetime:
    if isinstance(now_utc, str):
        dt = parse_utc_timestamp(now_utc)
        if dt is None:
            try:
                dt = datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(UTC_TZ)
        now_utc = dt
    if not isinstance(now_utc, datetime):
        now_utc = datetime.now(UTC_TZ)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC_TZ)
    else:
        now_utc = now_utc.astimezone(UTC_TZ)
    return now_utc


def _atomic_write(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def load_calls(path: str) -> list[dict[str, Any]]:
    """Load a calls list, tolerating missing or corrupt files."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_calls(path: str, calls: list[dict[str, Any]]) -> None:
    """Atomically persist a calls list."""
    _atomic_write(path, calls if isinstance(calls, list) else [])


def _to_prob(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN guard
        return None
    if abs(f) > 1.0:
        f = f / 100.0
    return f


def _parse_start(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_TZ)
        return dt.astimezone(UTC_TZ)
    return parse_utc_timestamp(value if isinstance(value, str) else (str(value) if value is not None else None))


def _call_id(call: dict[str, Any]) -> Any:
    return call.get("id", call.get("call_id"))


def _is_admin_call(call: dict[str, Any]) -> bool:
    return "side" in call and "price" in call


def _is_graded(call: dict[str, Any]) -> bool:
    return call.get("result") in ("win", "loss") or call.get("graded_at") is not None


def snapshot_calls(
    games: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    now_utc: Any,
    *,
    admin: bool = False,
    kind: str | None = None,
    **kwargs: Any,
) -> int:
    """Snapshot new open calls. Returns number created."""
    now = _normalize_now(now_utc)
    now_iso = now.isoformat()
    existing_ids = {_call_id(c) for c in (existing or []) if _call_id(c) is not None}
    created = 0

    for game in games or []:
        gid = game.get("id")
        if gid is None or gid in existing_ids:
            continue

        start_dt = _parse_start(game.get("start_time_utc"))
        # Never snapshot a started or unparseable-past game; require future kickoff.
        if start_dt is None or start_dt <= now:
            continue

        is_admin = bool(admin or kind == 'admin' or kwargs.get('is_admin') or kwargs.get('admin_mode') or (existing and _is_admin_call(existing[0])))

        if is_admin:
            # choose larger of away_edge/home_edge only if edge_pct >= 2 and odds present
            ae = game.get("away_edge")
            he = game.get("home_edge")
            candidates = []
            if isinstance(ae, dict) and ae.get("edge_pct") is not None:
                candidates.append(("away", ae, game.get("away"), game.get("away_prob")))
            if isinstance(he, dict) and he.get("edge_pct") is not None:
                candidates.append(("home", he, game.get("home"), game.get("home_prob")))
            chosen = None
            for side, edge, team, model_prob in candidates:
                if edge.get("odds") is None:
                    continue
                try:
                    _edge_pct = float(edge.get("edge_pct", 0) or 0)
                    _price = float(edge.get("odds"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(_edge_pct) or not math.isfinite(_price):
                    continue
                if _edge_pct < 2:
                    continue
                if chosen is None or _edge_pct > float(chosen[1].get("edge_pct", 0) or 0):
                    chosen = (side, edge, team, model_prob)
            if chosen is None:
                continue
            side, edge, team, model_prob = chosen
            call = {
                "id": gid,
                "call_id": gid,
                "date": game.get("date"),
                "away": game.get("away"),
                "home": game.get("home"),
                "market": "ml",
                "side": side,
                "team": team,
                "price": float(edge.get("odds")),
                "bookmaker": game.get("bookie"),
                "edge": (float(edge.get("edge_pct", 0) or 0) / 100.0),
                "model_prob": _to_prob(model_prob),
                "kickoff": game.get("start_time_utc"),
                "closing_price": None,
                "closing_at": None,
                "snapshot_at": now_iso,
                "status": "open",
                "result": None,
                "profit": None,
                "final_score": None,
                "graded_at": None,
            }
            existing.append(call)
            existing_ids.add(gid)
            created += 1
        else:
            away_prob = _to_prob(game.get("away_prob"))
            home_prob = _to_prob(game.get("home_prob"))
            if away_prob is None or home_prob is None:
                continue
            if away_prob >= home_prob:
                pick = game.get("away")
                pick_prob = away_prob
            else:
                pick = game.get("home")
                pick_prob = home_prob
            call = {
                "id": gid,
                "call_id": gid,
                "date": game.get("date"),
                "away": game.get("away"),
                "home": game.get("home"),
                "pick": pick,
                "pick_prob": pick_prob,
                "away_prob": away_prob,
                "home_prob": home_prob,
                "kickoff": game.get("start_time_utc"),
                "snapshot_at": now_iso,
                "status": "open",
                "closing_prob": None,
                "result": None,
                "final_score": None,
                "graded_at": None,
            }
            existing.append(call)
            existing_ids.add(gid)
            created += 1

    return created


def update_closing(
    games: list[dict[str, Any]],
    existing_public: list[dict[str, Any]],
    existing_admin: list[dict[str, Any]],
    now_utc: Any,
) -> int:
    """Refresh closing probs/prices for open, pre-start calls."""
    now = _normalize_now(now_utc)
    now_iso = now.isoformat()
    games_by_id = {g.get("id"): g for g in (games or []) if g.get("id") is not None}

    updated = 0

    # public
    for call in existing_public or []:
        if _is_graded(call):
            continue
        if call.get("status") not in (None, "open"):
            continue
        gid = _call_id(call)
        game = games_by_id.get(gid)
        if not game:
            continue
        start_dt = _parse_start(game.get("start_time_utc")) or _parse_start(call.get("kickoff"))
        if start_dt is None or start_dt <= now:
            continue
        state = str(game.get("game_state", "")).upper()
        if state not in ("FUT", "PRE"):
            continue

        pick = call.get("pick")
        away = game.get("away")
        home = game.get("home")
        if pick == away:
            closing = _to_prob(game.get("away_prob"))
        elif pick == home:
            closing = _to_prob(game.get("home_prob"))
        else:
            closing = None
        if closing is None:
            continue
        call["closing_prob"] = closing
        call["closing_at"] = now_iso
        updated += 1

    # admin
    for call in existing_admin or []:
        if _is_graded(call):
            continue
        if call.get("status") not in (None, "open"):
            continue
        gid = _call_id(call)
        game = games_by_id.get(gid)
        if not game:
            continue
        start_dt = _parse_start(game.get("start_time_utc")) or _parse_start(call.get("kickoff"))
        if start_dt is None or start_dt <= now:
            continue
        state = str(game.get("game_state", "")).upper()
        if state not in ("FUT", "PRE"):
            continue

        side = call.get("side")
        if side == "away":
            price = game.get("away_odds")
        elif side == "home":
            price = game.get("home_odds")
        else:
            price = None
        if price is None:
            # fall back to edge odds if available
            edge = game.get("away_edge") if side == "away" else game.get("home_edge")
            if isinstance(edge, dict):
                price = edge.get("odds")
        if price is None:
            continue
        try:
            call["closing_price"] = round(float(price), 3) if not isinstance(price, int) else price
        except (TypeError, ValueError):
            call["closing_price"] = price
        call["closing_at"] = now_iso
        updated += 1

    return updated


def _grade_one_public(call: dict[str, Any], away_score: Any, home_score: Any, now_iso: str) -> bool:
    try:
        a = int(float(away_score))
        h = int(float(home_score))
    except (TypeError, ValueError):
        return False
    pick = call.get("pick")
    away = call.get("away")
    home = call.get("home")
    if pick == away:
        win = a > h
    elif pick == home:
        win = h > a
    else:
        # fallback: pick side unknown; infer from pick_prob? default home
        win = h > a
    call["result"] = "win" if win else "loss"
    call["final_score"] = f"{a}-{h}"
    call["graded_at"] = now_iso
    call["status"] = "graded"
    return True


def _grade_one_admin(call: dict[str, Any], away_score: Any, home_score: Any, now_iso: str) -> bool:
    try:
        a = int(float(away_score))
        h = int(float(home_score))
    except (TypeError, ValueError):
        return False
    side = call.get("side")
    if side == "away":
        win = a > h
    elif side == "home":
        win = h > a
    else:
        win = False
    call["result"] = "win" if win else "loss"
    call["final_score"] = f"{a}-{h}"
    call["graded_at"] = now_iso
    call["status"] = "graded"
    price = call.get("price")
    try:
        p = float(price) if price is not None else 0.0
    except (TypeError, ValueError):
        p = 0.0
    if win:
        call["profit"] = round(p - 1.0, 4) if p else 0.0
    else:
        call["profit"] = -1.0
    return True


def grade_calls(
    games: list[dict[str, Any]],
    public_calls: list[dict[str, Any]],
    admin_calls: list[dict[str, Any]],
    now_utc: Any,
    fetch_scores: Optional[Callable[[str], dict[str, Any]]] = None,
) -> int:
    """Grade final games. Returns number of calls graded."""
    now = _normalize_now(now_utc)
    now_iso = now.isoformat()
    today_str = now.date().isoformat()

    games_by_id = {g.get("id"): g for g in (games or []) if g.get("id") is not None}

    # fetch_scores cache by date
    fetch_cache: dict[str, dict[str, Any]] = {}

    def _get_scores_for_missing(call: dict[str, Any]) -> tuple[Any, Any, str] | None:
        # returns (away_score, home_score, state) if available
        date_str = str(call.get("date") or "")
        if not date_str or date_str >= today_str:
            return None
        if fetch_scores is None:
            return None
        if date_str not in fetch_cache:
            try:
                fetch_cache[date_str] = fetch_scores(date_str) or {}
            except Exception:
                fetch_cache[date_str] = {}
        info = (fetch_cache[date_str] or {}).get(_call_id(call))
        if not isinstance(info, dict):
            return None
        state = str(info.get("state", "")).upper()
        a = info.get("away_score")
        h = info.get("home_score")
        if a is None or h is None:
            return None
        if state not in ("FINAL", "OFF"):
            return None
        return a, h, state

    graded_count = 0

    # public
    for call in public_calls or []:
        if _is_graded(call):
            continue
        gid = _call_id(call)
        game = games_by_id.get(gid)
        away_score = None
        home_score = None
        state = None
        if game is not None:
            state = str(game.get("game_state", "")).upper()
            away_score = game.get("away_score")
            home_score = game.get("home_score")
        else:
            res = _get_scores_for_missing(call)
            if res is not None:
                away_score, home_score, state = res
        if state not in ("FINAL", "OFF"):
            continue
        if away_score is None or home_score is None:
            continue
        if _grade_one_public(call, away_score, home_score, now_iso):
            graded_count += 1

    # admin
    for call in admin_calls or []:
        if _is_graded(call):
            continue
        gid = _call_id(call)
        game = games_by_id.get(gid)
        away_score = None
        home_score = None
        state = None
        if game is not None:
            state = str(game.get("game_state", "")).upper()
            away_score = game.get("away_score")
            home_score = game.get("home_score")
        else:
            res = _get_scores_for_missing(call)
            if res is not None:
                away_score, home_score, state = res
        if state not in ("FINAL", "OFF"):
            continue
        if away_score is None or home_score is None:
            continue
        if _grade_one_admin(call, away_score, home_score, now_iso):
            graded_count += 1

    return graded_count


def _is_graded_call(call: dict[str, Any]) -> bool:
    return call.get("result") in ("win", "loss")


def summarize(public_calls: list[dict[str, Any]]) -> dict[str, Any]:
    calls = list(public_calls or [])
    graded = sorted(
        [c for c in calls if _is_graded_call(c)],
        key=lambda c: str(c.get("graded_at") or c.get("kickoff") or ""),
    )
    open_calls = [c for c in calls if not _is_graded_call(c)]
    wins = sum(1 for c in graded if c.get("result") == "win")
    losses = sum(1 for c in graded if c.get("result") == "loss")
    total_graded = len(graded)
    win_rate = (wins / total_graded) if total_graded else 0.0

    brier_vals = []
    for c in graded:
        pp = c.get("pick_prob")
        try:
            p = float(pp)
        except (TypeError, ValueError):
            continue
        outcome = 1.0 if c.get("result") == "win" else 0.0
        brier_vals.append((p - outcome) ** 2)
    brier = (sum(brier_vals) / len(brier_vals)) if brier_vals else 0.0

    # current streak from list order (most recent graded at end)
    streak_type = None
    streak_n = 0
    for c in graded:
        r = c.get("result")
        t = "W" if r == "win" else "L"
        if streak_type == t:
            streak_n += 1
        else:
            streak_type = t
            streak_n = 1
    if streak_type is None:
        current_streak = {"type": "W", "n": 0}
    else:
        current_streak = {"type": streak_type, "n": streak_n}

    buckets = {"50-55": [], "55-60": [], "60-65": [], "65+": []}
    for c in graded:
        try:
            p = float(c.get("pick_prob")) * 100.0
        except (TypeError, ValueError):
            continue
        if p < 55:
            key = "50-55"
        elif p < 60:
            key = "55-60"
        elif p < 65:
            key = "60-65"
        else:
            key = "65+"
        buckets[key].append(c)

    by_confidence = {}
    for key, items in buckets.items():
        w = sum(1 for c in items if c.get("result") == "win")
        g = len(items)
        by_confidence[key] = {
            "calls": g,
            "n": g,
            "wins": w,
            "losses": g - w,
            "win_rate": (w / g) if g else 0.0,
        }

    ordered = sorted(calls, key=lambda c: (str(c.get("kickoff") or c.get("date") or ""), str(c.get("call_id") or "")))
    recent = [dict(c) for c in ordered[-12:]]

    return {
        "calls": len(calls),
        "graded": total_graded,
        "open": len(open_calls),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "brier": brier,
        "current_streak": current_streak,
        "by_confidence": by_confidence,
        "recent": recent,
    }


def summarize_admin(admin_calls: list[dict[str, Any]]) -> dict[str, Any]:
    calls = list(admin_calls or [])
    graded = sorted(
        [c for c in calls if _is_graded_call(c)],
        key=lambda c: str(c.get("graded_at") or c.get("kickoff") or ""),
    )
    open_calls = [c for c in calls if not _is_graded_call(c)]
    wins = sum(1 for c in graded if c.get("result") == "win")
    losses = sum(1 for c in graded if c.get("result") == "loss")
    total_graded = len(graded)
    win_rate = (wins / total_graded) if total_graded else 0.0
    profit = 0.0
    for c in graded:
        try:
            profit += float(c.get("profit") or 0.0)
        except (TypeError, ValueError):
            continue
    roi = (profit / total_graded) if total_graded else 0.0
    ordered = sorted(calls, key=lambda c: (str(c.get("kickoff") or c.get("date") or ""), str(c.get("call_id") or "")))
    recent = [dict(c) for c in ordered[-12:]]
    return {
        "calls": len(calls),
        "graded": total_graded,
        "open": len(open_calls),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "roi": roi,
        "profit": profit,
        "recent": recent,
    }


def update_calls_ledger(
    docs_dir: str,
    games: list[dict[str, Any]],
    now_utc: Any,
    fetch_scores: Optional[Callable[[str], dict[str, Any]]] = None,
) -> dict[str, Any]:
    now = _normalize_now(now_utc)
    history_path = os.path.join(docs_dir, "calls_history.json")
    ledger_path = os.path.join(docs_dir, "calls_ledger.json")
    scoreboard_path = os.path.join(docs_dir, "calls_scoreboard.json")

    public_calls = load_calls(history_path)
    admin_calls = load_calls(ledger_path)

    snapshot_calls(games, public_calls, now, admin=False)
    snapshot_calls(games, admin_calls, now, admin=True)

    update_closing(games, public_calls, admin_calls, now)
    grade_calls(games, public_calls, admin_calls, now, fetch_scores=fetch_scores)

    save_calls(history_path, public_calls)
    save_calls(ledger_path, admin_calls)

    summary = summarize(public_calls)
    admin_summary = summarize_admin(admin_calls)
    # Public file: no admin odds/ROI. The admin summary goes to
    # calls_ledger.json and data.json.calls_admin (stripped by the worker).
    _atomic_write(scoreboard_path, summary)

    return {"summary": summary, "admin_summary": admin_summary, "recent": summary.get("recent", [])}
