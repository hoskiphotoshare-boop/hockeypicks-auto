"""Tests for the public docs data-exposure guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_public_docs import check_docs


class PublicDocsGuardTests(unittest.TestCase):
    def test_repo_docs_are_clean(self) -> None:
        self.assertEqual(check_docs("."), [])

    def test_detects_calls_admin_and_odds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(
                json.dumps({"calls_admin": {}, "games": [{"id": 1, "away_edge": {"edge_pct": 2}}]})
            )
            (docs / "calls_ledger.json").write_text("[]")
            errors = check_docs(tmp)
            self.assertTrue(any("calls_ledger" in e for e in errors))
            self.assertTrue(any("calls_admin" in e for e in errors))
            self.assertTrue(any("away_edge" in e for e in errors))


if __name__ == "__main__":
    unittest.main()


class PublicDocsGuardExtendedTests(unittest.TestCase):
    def test_detects_meta_odds_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"meta": {"odds_usage": {"x": 1}}, "games": []}))
            (docs / "cup_futures.json").write_text(json.dumps({"meta": {"odds_enabled": False}, "teams": [], "summary": {}}))
            errors = check_docs(tmp)
            self.assertTrue(any("odds_usage" in e for e in errors))

    def test_detects_new_game_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"games": [{"id": 1, "bookie": "X", "bookie_key": "y"}]}))
            (docs / "cup_futures.json").write_text(json.dumps({"meta": {"odds_enabled": False}, "teams": [], "summary": {}}))
            errors = check_docs(tmp)
            self.assertTrue(any("bookie" in e for e in errors))

    def test_cup_futures_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"games": []}))
            errors = check_docs(tmp)
            self.assertTrue(any("cup_futures" in e for e in errors))

    def test_cup_futures_banned_team_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"games": []}))
            (docs / "cup_futures.json").write_text(json.dumps({
                "meta": {"odds_enabled": False},
                "teams": [{"team": "TOR", "best_odds": 5.0}],
                "summary": {},
            }))
            errors = check_docs(tmp)
            self.assertTrue(any("best_odds" in e for e in errors))

    def test_cup_futures_odds_enabled_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"games": []}))
            (docs / "cup_futures.json").write_text(json.dumps({
                "meta": {"odds_enabled": True},
                "teams": [],
                "summary": {},
            }))
            errors = check_docs(tmp)
            self.assertTrue(any("odds_enabled" in e for e in errors))

    def test_cup_futures_banned_summary_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "data.json").write_text(json.dumps({"games": []}))
            (docs / "cup_futures.json").write_text(json.dumps({
                "meta": {"odds_enabled": False},
                "teams": [],
                "summary": {"bankroll": 1000},
            }))
            errors = check_docs(tmp)
            self.assertTrue(any("bankroll" in e for e in errors))
