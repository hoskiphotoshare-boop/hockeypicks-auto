"""Pure-Python grouping helpers for scored Tim Hortons Challenge candidates."""

from __future__ import annotations

from typing import Any


def group_tims_candidates(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Group scored Tims candidates by group number without pandas dtype coercion.

    pandas converts mixed ``None``/string ``injury_status`` values to a string
    dtype and emits ``NaN`` when converting records back, which produces
    invalid JSON for the dashboard. Pure-Python sorting keeps ``None`` intact.
    """
    ordered = sorted(
        items,
        key=lambda item: (
            int(item.get("group", 1)),
            bool(item.get("injury_out")),
            -float(item.get("final_score", 0.0)),
        ),
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in ordered:
        groups.setdefault(str(int(item.get("group", 1))), []).append(item)
    return groups
