# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""Provider choice and word-by-word streaming for the OpenAI-compatible providers (no network)."""
import json
import unittest
from unittest import mock
from app.config import Config
from app.rag_engine import RAGEngine


class FakeStream:
    status_code = 200
    def __init__(self, pieces):
        self.lines = [f"data: {json.dumps({'choices': [{'delta': d}]})}" for d in pieces] + ["data: [DONE]"]
        self.encoding = None
    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestProviders(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(Config, k) for k in ("DEFAULT_LLM_PROVIDER", "CEREBRAS_API_KEY", "GROQ_API_KEY")}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(Config, k, v)

    def test_cerebras_is_chosen_when_selected_with_a_key(self):
        rag = RAGEngine.__new__(RAGEngine)
        Config.DEFAULT_LLM_PROVIDER, Config.CEREBRAS_API_KEY = "cerebras", "csk-test"
        self.assertEqual(rag._get_active_provider(), "cerebras")
        Config.CEREBRAS_API_KEY = ""
        self.assertEqual(rag._get_active_provider(), "fallback")

    def test_stream_yields_only_answer_text(self):
        rag = RAGEngine.__new__(RAGEngine)
        Config.CEREBRAS_API_KEY = "csk-test"
        fake = FakeStream([{"reasoning": "thinking..."}, {"content": "B.E. "}, {"content": "Cyber Security"}])
        with mock.patch("app.rag_engine.requests.post", return_value=fake) as post:
            pieces = list(rag._stream_chat("cerebras", "sys", [], "question"))
        self.assertEqual(pieces, ["B.E. ", "Cyber Security"])
        url = post.call_args[0][0]
        payload = post.call_args[1]["json"]
        self.assertEqual(url, "https://api.cerebras.ai/v1/chat/completions")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], Config.CEREBRAS_MODEL)
        self.assertEqual(payload["max_completion_tokens"], 2500)
        self.assertEqual(fake.encoding, "utf-8")                    # Tamil / ₹ survive

    def test_extraction_uses_extract_model_without_chat_limits(self):
        rag = RAGEngine.__new__(RAGEngine)
        Config.DEFAULT_LLM_PROVIDER, Config.CEREBRAS_API_KEY = "cerebras", "csk-test"
        reply = mock.Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "{\"facts\": []}"}}]})
        with mock.patch("app.rag_engine.requests.post", return_value=reply) as post:
            rag.complete("sys", "page text")
        payload = post.call_args[1]["json"]
        self.assertEqual(payload["model"], Config.CEREBRAS_EXTRACT_MODEL)
        self.assertNotIn("max_completion_tokens", payload)


    def test_per_minute_limit_without_retry_hint_waits_and_retries(self):
        rag = RAGEngine.__new__(RAGEngine)
        calls = []
        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('Cerebras API returned status 429: {"message":"Requests per minute limit exceeded"}')
            return "answer"
        rag._call_cerebras = flaky
        with mock.patch("app.rag_engine.time.sleep") as slept:
            self.assertEqual(rag._call_with_retry("cerebras", "sys", [], "q"), "answer")
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(slept.call_args[0][0], 10)

    def test_daily_quota_is_not_retried(self):
        rag = RAGEngine.__new__(RAGEngine)
        rag._call_cerebras = mock.Mock(side_effect=RuntimeError("Cerebras API returned status 429: daily token quota exceeded"))
        with mock.patch("app.rag_engine.time.sleep"), self.assertRaises(RuntimeError):
            rag._call_with_retry("cerebras", "sys", [], "q")


if __name__ == "__main__":
    unittest.main()
