"""Publish admin-only artifacts to the Cloudflare Worker via a private endpoint."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "https://hockeypicks-ai.pages.dev/api/admin/publish"


def publish_admin_artifacts(
    *,
    data: Any = None,
    cup_futures: Any = None,
    latest_run_html: Any = None,
    url: str | None = None,
    token: str | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """POST provided sections to the admin publish endpoint. Never raises."""
    resolved_token = token if token is not None else os.environ.get("HP_ADMIN_PUBLISH_TOKEN")
    if not resolved_token:
        return {"published": False, "reason": "no token"}

    resolved_url = url if url is not None else os.environ.get("HP_ADMIN_PUBLISH_URL", DEFAULT_URL)

    body: dict[str, Any] = {}
    if data is not None:
        body["data"] = data
    if cup_futures is not None:
        body["cup_futures"] = cup_futures
    if latest_run_html is not None:
        body["latest_run_html"] = latest_run_html

    try:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            resolved_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {resolved_token}",
                "Content-Type": "application/json",
                "User-Agent": "hockeypicks-admin-publisher",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else {}
            except Exception:
                parsed = {}
            result = {"published": True}
            if isinstance(parsed, dict):
                result.update(parsed)
            return result
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        print(f"Warning: admin publish failed: HTTP {exc.code} {raw[:200]}")
        return {"published": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:
        print(f"Warning: admin publish failed: {exc}")
        return {"published": False, "error": str(exc)}
