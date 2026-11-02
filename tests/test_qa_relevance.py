import unittest
import tempfile
import shutil
from pathlib import Path
from app.storage import Database
from app.vector_store import VectorSearchEngine
from app.rag_engine import RAGEngine

class TestQARelevance(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(Path(self.tmpdir) / "test.db")
        self.vs = VectorSearchEngine(self.db)
        self.rag = RAGEngine(self.db, self.vs)

        self.col = self.db.create_or_get_college("https://stanford.edu", "stanford.edu", "Stanford University")
        self.cid = self.col["id"]

        # Page 1: Tuition
        p1 = self.db.save_page(
            self.cid,
            "https://stanford.edu/tuition",
            "Tuition & Costs",
            "Undergraduate tuition for 2026 is $62,000 per academic year. Housing and dining cost is $18,500."
        )
        self.db.save_chunk(
            self.cid, p1,
            "https://stanford.edu/tuition",
            "Tuition & Costs",
            "Tuition Rates", 0,
            "Undergraduate tuition for 2026 is $62,000 per academic year. Housing and dining cost is $18,500."
        )

        # Page 2: Campus
        p2 = self.db.save_page(
            self.cid,
            "https://stanford.edu/campus",
            "Campus Life",
            "Founded in 1891, the campus covers 8,000 acres with palm trees and bike paths."
        )
        self.db.save_chunk(
            self.cid, p2,
            "https://stanford.edu/campus",
            "Campus Life",
            "Campus Overview", 0,
            "Founded in 1891, the campus covers 8,000 acres with palm trees and bike paths."
        )

        # Page 3: Admissions
        p3 = self.db.save_page(
            self.cid,
            "https://stanford.edu/admissions",
            "Admissions Deadlines",
            "Undergraduate regular decision applications are due on January 5. Standardized test scores are optional."
        )
        self.db.save_chunk(
            self.cid, p3,
            "https://stanford.edu/admissions",
            "Admissions Deadlines",
            "Application Deadlines", 0,
            "Undergraduate regular decision applications are due on January 5. Standardized test scores are optional."
        )

        self.vs.index_pending_chunks(self.cid)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_tuition_relevance(self):
        ans = self.rag.answer_question(self.cid, "What is the undergraduate tuition fee and housing cost?")
        self.assertIn("$62,000", ans["content"])
        self.assertIn("$18,500", ans["content"])
        self.assertNotIn("palm trees", ans["content"])

    def test_deadline_relevance(self):
        ans = self.rag.answer_question(self.cid, "When is the application deadline for regular decision?")
        self.assertIn("January 5", ans["content"])
        self.assertNotIn("$62,000", ans["content"])

if __name__ == "__main__":
    unittest.main()
