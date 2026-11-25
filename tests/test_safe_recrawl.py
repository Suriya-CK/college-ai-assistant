# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""A re-crawl writes into a hidden staging copy; the old data is replaced only when the crawl worked."""
import shutil
import tempfile
import unittest
from pathlib import Path
from app.storage import Database


class TestSafeRecrawl(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = Database(Path(self.dir) / "t.db")
        self.college = self.db.create_or_get_college("https://x.edu/", "x.edu", "X College")
        self.cid = self.college["id"]
        self.add(self.cid, "https://x.edu/old", "Old fees page: tuition is 50,000 rupees", "Old fee")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def add(self, cid, url, text, fact):
        page = self.db.save_page(cid, url, "Page", text)
        self.db.save_chunks_batch(cid, page, url, "Page", [{"heading": "Fees", "text": text}])
        self.db.save_facts(cid, [{"category": "fee", "name": fact, "details": text[-14:]}], url)

    def test_staging_is_hidden_and_swap_replaces_everything(self):
        staging = self.db.create_staging_college(self.college)
        self.assertEqual([c["id"] for c in self.db.list_colleges()], [self.cid])     # never shown in the app
        self.add(staging, "https://x.edu/new", "New fees page: tuition is 65,000 rupees", "New fee")
        # Until the swap, questions still see the old data
        self.assertEqual([p["url"] for p in self.db.get_pages_with_text(self.cid)], ["https://x.edu/old"])

        self.db.replace_college_content(self.cid, staging)
        self.assertEqual([p["url"] for p in self.db.get_pages_with_text(self.cid)], ["https://x.edu/new"])
        self.assertEqual([f["name"] for f in self.db.get_facts(self.cid)], ["New fee"])
        self.assertIsNone(self.db.get_college(staging))
        hits = self.db.search_bm25(self.cid, "tuition", limit=5)
        self.assertTrue(hits and all("65,000" in h["text"] for h in hits))          # search index moved too
        self.assertEqual(self.db.get_college(self.cid)["page_count"], 1)

    def test_discarded_staging_leaves_old_data(self):
        staging = self.db.create_staging_college(self.college)
        self.add(staging, "https://x.edu/half", "Half crawl page with tuition", "Half")
        self.db.delete_college(staging)
        self.assertEqual([p["url"] for p in self.db.get_pages_with_text(self.cid)], ["https://x.edu/old"])
        self.assertEqual([f["name"] for f in self.db.get_facts(self.cid)], ["Old fee"])
        self.assertTrue(all("50,000" in h["text"] for h in self.db.search_bm25(self.cid, "tuition", limit=5)))

    def test_leftover_staging_from_an_interrupted_crawl_is_cleaned(self):
        first = self.db.create_staging_college(self.college)
        self.db.create_staging_college(self.college)
        self.assertIsNone(self.db.get_college(first))


if __name__ == "__main__":
    unittest.main()
