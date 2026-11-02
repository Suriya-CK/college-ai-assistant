import shutil
import unittest
from pathlib import Path

from app.storage import Database
from app.rag_engine import RAGEngine
from app.voice_service import resolve_spoken_language


class TestSpokenLanguage(unittest.TestCase):
    def test_whisper_language_is_used(self):
        self.assertEqual(resolve_spoken_language("Tamil", "இங்கு என்னென்ன படிப்புகள் உள்ளன?"), "ta")
        self.assertEqual(resolve_spoken_language("English", "Which companies hire here?"), "en")
        self.assertEqual(resolve_spoken_language("hindi", "यहाँ कौन से कोर्स हैं?"), "hi")

    def test_tamil_spoken_but_written_in_latin_letters(self):
        self.assertEqual(resolve_spoken_language("Tamil", "inga enna courses irukku"), "ta")

    def test_script_wins_when_whisper_is_unsure_or_unsupported(self):
        self.assertEqual(resolve_spoken_language("Urdu", "இங்கு என்ன படிப்புகள்?"), "ta")
        self.assertEqual(resolve_spoken_language(None, "ఇక్కడ ఏ కోర్సులు ఉన్నాయి?"), "te")
        self.assertEqual(resolve_spoken_language("Urdu", "hello"), "en")

    def test_marathi_vs_hindi_trusts_whisper(self):
        self.assertEqual(resolve_spoken_language("Marathi", "इथे कोणते अभ्यासक्रम आहेत?"), "mr")


class TestAutoLanguageAnswer(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_lang_db")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.test_dir / "test.db")
        self.cid = self.db.create_or_get_college("https://example.edu", "example.edu", "Example College")["id"]
        self.db.save_facts(self.cid, [{"category": "ug_program", "name": "B.E. Civil Engineering"}], "https://example.edu/c")
        self.rag = RAGEngine(self.db)
        self.rag._get_active_provider = lambda: "fallback"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_auto_detects_typed_tamil(self):
        ans = self.rag.answer_question(self.cid, "இங்கு என்னென்ன படிப்புகள் உள்ளன?", language="auto")
        self.assertEqual(ans["language"], "ta")
        self.assertTrue(ans["speech_locale"].startswith("ta"))

    def test_rate_limited_answer_is_retried(self):
        calls = []
        def flaky_groq(system, history, prompt):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("Groq API returned status 429: Rate limit reached on tokens per minute (TPM). "
                                   "Please try again in 0.1s.")
            return "நிறுவனங்கள்: Infosys\n[SPOKEN_SUMMARY]\nInfosys.\n[/SPOKEN_SUMMARY]"
        self.rag._get_active_provider = lambda: "groq"
        self.rag._call_groq = flaky_groq
        ans = self.rag.answer_question(self.cid, "இங்கு என்னென்ன படிப்புகள் உள்ளன?", language="ta")
        self.assertEqual(len(calls), 2)
        self.assertEqual(ans["provider"], "groq")

    def test_fallback_list_uses_question_language(self):
        import app.rag_engine as rag_module
        original = rag_module.translate_to_language
        rag_module.translate_to_language = lambda text, lang: f"[{lang}] {text}"
        try:
            ans = self.rag.answer_question(self.cid, "இங்கு என்னென்ன படிப்புகள் உள்ளன?", language="ta")
        finally:
            rag_module.translate_to_language = original
        self.assertIn("### [ta] Undergraduate programmes", ans["content"])
        self.assertIn("B.E. Civil Engineering", ans["content"])          # names stay as written
        self.assertIn("[ta] I found", ans["spoken_text"])

    def test_auto_detects_typed_english(self):
        ans = self.rag.answer_question(self.cid, "List all UG courses", language="auto")
        self.assertEqual(ans["language"], "en")


if __name__ == "__main__":
    unittest.main()
