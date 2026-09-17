"""Tests for the FastAPI backend that serves the web UI."""

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from interfaces.api import server


class TestApi(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        server.sessions = server.SessionPersistence(session_dir=self._tmp.name)
        server._agents.clear()
        # Force offline mode so no network calls are made.
        self._env = patch.dict(os.environ, {}, clear=True)
        self._env.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_health(self):
        data = self.client.get("/api/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["offline"])
        self.assertIsInstance(data["provider"], str)
        self.assertGreater(data["tools"], 0)
        self.assertEqual(data["model"], "gpt-4o-mini")
        self.assertEqual(data["chain"], [])
        self.assertTrue(data["config_errors"])

    def test_no_token_configured_means_open_access(self):
        """With no OMNIASSIST_API_TOKEN set, local use needs no credentials."""
        self.assertEqual(self.client.get("/api/sessions").status_code, 200)

    def test_tools_endpoint(self):
        tools = self.client.get("/api/tools").json()["tools"]
        names = {t["name"] for t in tools}
        self.assertIn("run_shell", names)
        self.assertIn("basic_calculate", names)
        self.assertNotIn("DDGS", names)

    def test_run_endpoint_returns_answer_and_events(self):
        res = self.client.post("/api/run", json={"prompt": "hello"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["session_id"])
        self.assertIn("Offline mode", data["response"])
        self.assertEqual(data["events"][0]["type"], "plan")

    def test_session_is_persisted_and_listed(self):
        sid = self.client.post("/api/run", json={"prompt": "remember this"}).json()["session_id"]

        listed = self.client.get("/api/sessions").json()["sessions"]
        self.assertEqual([s["session_id"] for s in listed], [sid])

        stored = self.client.get(f"/api/sessions/{sid}").json()
        self.assertEqual(stored["messages"][0]["content"], "remember this")

    def test_unknown_session_returns_404(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_delete_session(self):
        sid = self.client.post("/api/run", json={"prompt": "temp"}).json()["session_id"]
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 404)

    def test_traversal_in_session_id_is_rejected(self):
        # Starlette rejects the encoded slash at routing time, so the request
        # never reaches the handler or the filesystem.
        self.assertIn(
            self.client.get("/api/sessions/..%2f..%2fetc%2fpasswd").status_code,
            (400, 404),
        )

    def test_persistence_rejects_unsafe_session_ids(self):
        for bad in ("../../etc/passwd", "a/b", "", "x" * 70):
            self.assertIn("error", server.sessions.load_session(bad))
            self.assertFalse(server.sessions.save_session(bad, {}).startswith("Session saved"))

    def test_index_serves_the_ui(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("OmniAssist", res.text)

    def test_static_assets_are_served(self):
        for asset in ("/static/app.js", "/static/styles.css"):
            self.assertEqual(self.client.get(asset).status_code, 200, asset)

    def test_websocket_streams_events(self):
        with self.client.websocket_connect("/ws/run") as ws:
            ws.send_json({"prompt": "hello"})
            received = []
            while True:
                frame = ws.receive_json()
                received.append(frame["type"])
                if frame["type"] == "done":
                    break
            self.assertEqual(received[0], "session")
            self.assertIn("plan", received)
            self.assertIn("final", received)


class TestAuth(unittest.TestCase):
    """The API exposes shell-executing tools, so token auth must actually gate it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        server.sessions = server.SessionPersistence(session_dir=self._tmp.name)
        server._agents.clear()
        server.API_TOKEN = "secret-token"
        self._env = patch.dict(os.environ, {}, clear=True)
        self._env.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        server.API_TOKEN = None
        self._env.stop()
        self._tmp.cleanup()

    def test_run_requires_token(self):
        self.assertEqual(self.client.post("/api/run", json={"prompt": "hi"}).status_code, 401)

    def test_run_rejects_wrong_token(self):
        res = self.client.post(
            "/api/run", json={"prompt": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        self.assertEqual(res.status_code, 401)

    def test_run_accepts_correct_token(self):
        res = self.client.post(
            "/api/run",
            json={"prompt": "hi"},
            headers={"Authorization": "Bearer secret-token"},
        )
        self.assertEqual(res.status_code, 200)

    def test_sessions_require_token(self):
        self.assertEqual(self.client.get("/api/sessions").status_code, 401)

    def test_delete_requires_token(self):
        self.assertEqual(self.client.delete("/api/sessions/abc").status_code, 401)

    def test_websocket_requires_token(self):
        from starlette.websockets import WebSocketDisconnect

        with self.assertRaises(WebSocketDisconnect), self.client.websocket_connect("/ws/run") as ws:
            ws.receive_json()

    def test_health_stays_public(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
