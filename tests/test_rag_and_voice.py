import unittest
import shutil
from pathlib import Path

from app.storage import Database
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine
from app.voice_service import VoiceService

class TestRAGAndVoice(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_rag_dir")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.test_dir / "test.db")
        self.vector_engine = VectorSearchEngine(self.db)
        self.rag = RAGEngine(self.db, self.vector_engine)
        self.voice = VoiceService(self.test_dir / "audio")

        # Seed sample college knowledge
        self.col = self.db.create_or_get_college("https://caltech.edu", "caltech.edu", "Caltech")
        p_id = self.db.save_page(
            self.col["id"],
            "https://caltech.edu/admissions",
            "Admissions at Caltech",
            "Undergraduate admissions requires strong mathematics and physics background. Applications are due January 3."
        )
        self.db.save_chunk(
            self.col["id"],
            p_id,
            "https://caltech.edu/admissions",
            "Admissions at Caltech",
            "Requirements",
            0,
            "Undergraduate admissions requires strong mathematics and physics background. Applications are due January 3."
        )
        self.vector_engine.index_pending_chunks(self.col["id"])

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_hybrid_search(self):
        results = self.vector_engine.hybrid_search(self.col["id"], "mathematics admissions deadline", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("January 3", results[0]["text"])

    def test_rag_answer(self):
        answer = self.rag.answer_question(self.col["id"], "When are applications due?")
        self.assertIn("content", answer)
        self.assertIn("spoken_text", answer)
        self.assertTrue(len(answer["citations"]) > 0)
        self.assertEqual(answer["citations"][0]["url"], "https://caltech.edu/admissions")

    def test_voice_text_cleaning(self):
        markdown_text = "### Welcome!\nVisit [Caltech Admissions](https://caltech.edu/admissions) for **more** details. [Source: Page]"
        cleaned = self.voice.clean_text_for_speech(markdown_text)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("[Source: Page]", cleaned)
        self.assertIn("Visit Caltech Admissions for more details", cleaned)

if __name__ == "__main__":
    unittest.main()
