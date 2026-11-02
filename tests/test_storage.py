import unittest
import shutil
from pathlib import Path
import numpy as np

from app.storage import Database

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_db")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.test_dir / "test.db")

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_college_and_pages(self):
        col = self.db.create_or_get_college("https://example.edu", "example.edu", "Example University")
        self.assertIsNotNone(col["id"])
        self.assertEqual(col["name"], "Example University")

        page_id = self.db.save_page(
            college_id=col["id"],
            url="https://example.edu/admissions",
            title="Admissions Overview",
            clean_text="Admission requirements for undergrads include SAT or ACT scores."
        )
        self.assertIsInstance(page_id, int)

        chunk_id = self.db.save_chunk(
            college_id=col["id"],
            page_id=page_id,
            url="https://example.edu/admissions",
            title="Admissions Overview",
            heading="Requirements",
            chunk_index=0,
            text="Admission requirements for undergrads include SAT or ACT scores.",
            embedding=np.zeros(128, dtype=np.float32)
        )
        self.assertIsInstance(chunk_id, int)

        # Test BM25 Full Text Search
        results = self.db.search_bm25(col["id"], "admissions requirements")
        self.assertTrue(len(results) > 0)
        self.assertIn("SAT", results[0]["text"])

    def test_same_url_in_two_colleges(self):
        a = self.db.create_or_get_college("https://www.annauniv.edu", "www.annauniv.edu", "Anna University")
        b = self.db.create_or_get_college("https://ceg.annauniv.edu", "ceg.annauniv.edu", "CEG")
        pa = self.db.save_page(a["id"], "https://ceg.annauniv.edu/", "Home", "CEG home page text")
        pb = self.db.save_page(b["id"], "https://ceg.annauniv.edu/", "Home", "CEG home page text")
        self.assertNotEqual(pa, pb)
        self.db.update_college_counts(b["id"])
        self.assertEqual(self.db.get_college(b["id"])["page_count"], 1)

if __name__ == "__main__":
    unittest.main()
