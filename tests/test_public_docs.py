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
