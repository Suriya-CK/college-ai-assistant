import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app.server as server
from app.config import Config

SAVED = ["BASE_DIR", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
         "OPENROUTER_MODEL", "DEFAULT_LLM_PROVIDER"]


class TestSettings(unittest.TestCase):
    """Settings must save keys to .env, use them immediately, and report whether they actually work."""

    def setUp(self):
        self.original = {name: getattr(Config, name) for name in SAVED}
        self.tmp = tempfile.TemporaryDirectory()
        Config.BASE_DIR = Path(self.tmp.name)                    # never touch the real .env
        (Config.BASE_DIR / ".env").write_text("GROQ_API_KEY=old-key\nDEFAULT_LLM_PROVIDER=groq\nPORT=8000\n")
        self.original_complete = server.rag_engine.complete
        server.rag_engine.complete = lambda system, prompt: "OK"   # no real API calls
        self.client = TestClient(server.app)

    def tearDown(self):
        server.rag_engine.complete = self.original_complete
        for name, value in self.original.items():
            setattr(Config, name, value)
        self.tmp.cleanup()

    def env(self):
        return (Config.BASE_DIR / ".env").read_text()

    def test_new_groq_key_is_saved_and_used_immediately(self):
        r = self.client.post("/api/settings", json={"groq_api_key": '  "gsk_new_key_1234"\n', "default_provider": "groq"}).json()
        self.assertEqual(Config.GROQ_API_KEY, "gsk_new_key_1234")          # quotes/whitespace stripped, live now
        self.assertIn("GROQ_API_KEY=gsk_new_key_1234", self.env())
        self.assertNotIn("old-key", self.env())
        self.assertIn("PORT=8000", self.env())                              # other settings untouched
        self.assertTrue(r["check"]["ok"])
        self.assertEqual(r["active_provider"], "groq")

    def test_empty_key_box_keeps_the_saved_key(self):
        Config.GROQ_API_KEY = "old-key"
        self.client.post("/api/settings", json={"groq_api_key": "", "default_provider": "groq"})
        self.assertEqual(Config.GROQ_API_KEY, "old-key")

    def test_openrouter_key_and_model(self):
        r = self.client.post("/api/settings", json={
            "openrouter_api_key": "sk-or-v1-abcdef123456", "openrouter_model": "google/gemma-4-31b-it:free",
            "default_provider": "openrouter"}).json()
        self.assertEqual(r["active_provider"], "openrouter")
        self.assertIn("OPENROUTER_API_KEY=sk-or-v1-abcdef123456", self.env())
        self.assertIn("OPENROUTER_MODEL=google/gemma-4-31b-it:free", self.env())
        self.assertIn("DEFAULT_LLM_PROVIDER=openrouter", self.env())
        settings = self.client.get("/api/settings").json()
        self.assertTrue(settings["has_openrouter_key"])
        self.assertEqual(settings["openrouter_masked_key"], "sk-or-...3456")

    def test_rejected_key_is_reported(self):
        def rejected(system, prompt):
            raise RuntimeError('Groq API returned status 401: {"error":{"message":"Invalid API Key"}}')
        server.rag_engine.complete = rejected
        r = self.client.post("/api/settings", json={"groq_api_key": "gsk_wrong", "default_provider": "groq"}).json()
        self.assertFalse(r["check"]["ok"])
        self.assertIn("rejected", r["check"]["message"])

    def test_rate_limited_key_is_reported(self):
        def limited(system, prompt):
            raise RuntimeError("Groq API returned status 429: Rate limit reached on tokens per day (TPD)")
        server.rag_engine.complete = limited
        r = self.client.post("/api/settings", json={"groq_api_key": "gsk_limited_key", "default_provider": "groq"}).json()
        self.assertFalse(r["check"]["ok"])
        self.assertIn("rate-limited", r["check"]["message"])

    def test_provider_without_key_is_reported(self):
        Config.OPENROUTER_API_KEY = ""
        r = self.client.post("/api/settings", json={"default_provider": "openrouter"}).json()
        self.assertEqual(r["active_provider"], "fallback")
        self.assertIn("No API key", r["check"]["message"])


if __name__ == "__main__":
    unittest.main()


class TestOpenRouterFallback(unittest.TestCase):
    """A busy free model must not end in the no-AI answer: the next free model is tried."""

    def setUp(self):
        import app.rag_engine as rag_module
        self.rag_module = rag_module
        self.original_post = rag_module.requests.post
        self.original = (Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL, Config.OPENROUTER_FALLBACK_MODELS)
        Config.OPENROUTER_API_KEY = "sk-or-test"
        Config.OPENROUTER_MODEL = "busy/model:free"
        Config.OPENROUTER_FALLBACK_MODELS = ["empty/model:free", "working/model:free"]

    def tearDown(self):
        self.rag_module.requests.post = self.original_post
        Config.OPENROUTER_API_KEY, Config.OPENROUTER_MODEL, Config.OPENROUTER_FALLBACK_MODELS = self.original

    def fake_post(self, replies, tried):
        class Resp:
            def __init__(self, status, body):
                self.status_code, self._body, self.text = status, body, str(body)
            def json(self):
                return self._body
        def post(url, headers=None, json=None, timeout=None):
            tried.append(json["model"])
            return Resp(*replies[json["model"]])
        return post

    def test_busy_model_falls_through_to_a_working_one(self):
        tried = []
        self.rag_module.requests.post = self.fake_post({
            "busy/model:free": (429, {"error": {"message": "temporarily rate-limited upstream"}}),
            "empty/model:free": (200, {"choices": [{"message": {"content": ""}}]}),
            "working/model:free": (200, {"choices": [{"message": {"content": "வணக்கம்"}}]}),
        }, tried)
        reply = self.rag_module.RAGEngine()._call_openrouter("sys", [], "hi")
        self.assertEqual(reply, "வணக்கம்")
        self.assertEqual(tried, ["busy/model:free", "empty/model:free", "working/model:free"])

    def test_bad_key_stops_immediately(self):
        tried = []
        self.rag_module.requests.post = self.fake_post({m: (401, {"error": {"message": "User not found."}})
                                                        for m in ["busy/model:free", "empty/model:free", "working/model:free"]}, tried)
        with self.assertRaises(RuntimeError):
            self.rag_module.RAGEngine()._call_openrouter("sys", [], "hi")
        self.assertEqual(tried, ["busy/model:free"])
