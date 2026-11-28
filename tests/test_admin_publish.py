"""Tests for src.admin_publish.publish_admin_artifacts."""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

from src.admin_publish import publish_admin_artifacts, DEFAULT_URL


class AdminPublishTests(unittest.TestCase):
    def test_missing_token_noop(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HP_ADMIN_PUBLISH_TOKEN", None)
            with patch("src.admin_publish.urllib.request.urlopen") as mock_urlopen:
                res = publish_admin_artifacts(data={"games": []})
                self.assertEqual(res, {"published": False, "reason": "no token"})
                mock_urlopen.assert_not_called()

    def test_success_posts_bearer_and_sections(self):
        os.environ["HP_ADMIN_PUBLISH_TOKEN"] = "secret-token"
        try:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"ok": true, "sections": ["data"]}'
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = False
            with patch("src.admin_publish.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    res = publish_admin_artifacts(data={"games": [1]}, latest_run_html="<html>")
                self.assertTrue(res.get("published"))
                self.assertEqual(res.get("ok"), True)
                req = mock_urlopen.call_args[0][0]
                self.assertEqual(req.get_header("Authorization"), "Bearer secret-token")
                self.assertEqual(req.get_header("Content-type"), "application/json")
                self.assertEqual(req.get_header("User-agent"), "hockeypicks-admin-publisher")
                body = json.loads(req.data.decode("utf-8"))
                self.assertIn("data", body)
                self.assertIn("latest_run_html", body)
                self.assertNotIn("cup_futures", body)
                self.assertNotIn("secret-token", buf.getvalue())
        finally:
            os.environ.pop("HP_ADMIN_PUBLISH_TOKEN", None)

    def test_http_error_swallowed(self):
        with patch.dict(os.environ, {"HP_ADMIN_PUBLISH_TOKEN": "t"}):
            err = HTTPError("http://x", 500, "boom", {}, io.BytesIO(b"server error body"))
            with patch("src.admin_publish.urllib.request.urlopen", side_effect=err):
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    res = publish_admin_artifacts(cup_futures={"teams": []})
                self.assertFalse(res.get("published"))
                self.assertIn("error", res)
                out = buf.getvalue()
                self.assertIn("500", out)
                self.assertNotIn("t", out)

    def test_url_override_env(self):
        with patch.dict(os.environ, {"HP_ADMIN_PUBLISH_TOKEN": "t", "HP_ADMIN_PUBLISH_URL": "https://example.com/pub"}):
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{}'
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = False
            with patch("src.admin_publish.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                publish_admin_artifacts(data={"games": []})
                req = mock_urlopen.call_args[0][0]
                self.assertEqual(req.full_url, "https://example.com/pub")

    def test_default_url_constant(self):
        self.assertEqual(DEFAULT_URL, "https://hockeypicks-ai.pages.dev/api/admin/publish")


if __name__ == "__main__":
    unittest.main()
