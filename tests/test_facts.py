import json
import shutil
import unittest
from pathlib import Path

from app.storage import Database
from app.extractor import FactExtractor, retry_after_seconds
from app.rag_engine import RAGEngine, fact_categories_for, merge_similar_facts


class TestFacts(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_facts_db")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.test_dir / "test.db")
        self.college = self.db.create_or_get_college("https://example.edu", "example.edu", "Example College")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_facts_tolerates_fences_and_filters_bad_items(self):
        raw = '```json\n{"facts": [{"category": "recruiter", "name": "Infosys"}, ' \
              '{"category": "bogus", "name": "x y"}, {"category": "ug_program", "name": "B.E. CSE", "details": "4 years"}]}\n```'
        facts = FactExtractor.parse_facts(raw)
        self.assertEqual([f["name"] for f in facts], ["Infosys", "B.E. CSE"])
        self.assertEqual(FactExtractor.parse_facts("no json here"), [])

    def test_save_facts_dedupes(self):
        cid = self.college["id"]
        added = self.db.save_facts(cid, [{"category": "recruiter", "name": "TCS"}], "https://example.edu/p")
        again = self.db.save_facts(cid, [{"category": "recruiter", "name": "T.C.S", "details": "2024"}], "https://example.edu/p2")
        self.assertEqual((added, again), (1, 0))
        facts = self.db.get_facts(cid, ["recruiter"])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["details"], "2024")   # details filled in on the duplicate

    def test_extract_college_with_fake_llm(self):
        cid = self.college["id"]
        self.db.save_page(cid, "https://example.edu/placements", "Placements",
                          "Placement 2024: companies that recruited our students include Infosys and Zoho. Highest package 12 LPA.")
        self.db.save_page(cid, "https://example.edu/gallery", "Gallery", "Photos of the annual day.")

        seen_urls = []
        def fake_llm(system, prompt):
            seen_urls.append(prompt.split("URL: ")[1].split("\n")[0])
            return json.dumps({"facts": [
                {"category": "recruiter", "name": "Infosys"}, {"category": "recruiter", "name": "Zoho"},
                {"category": "placement_stat", "name": "Highest package", "details": "12 LPA (2024)"},
            ]})

        from app.config import Config
        Config.EXTRACT_MIN_INTERVAL = 0
        added = FactExtractor(self.db, fake_llm).extract_college(cid)
        self.assertEqual(added, 3)
        self.assertEqual(seen_urls, ["https://example.edu/placements"])   # irrelevant page never sent to the LLM

    def test_retry_after_parsing(self):
        self.assertAlmostEqual(retry_after_seconds("Please try again in 8.82s. Need more tokens?"), 8.82)
        self.assertAlmostEqual(retry_after_seconds("Please try again in 1m23.5s."), 83.5)
        self.assertEqual(retry_after_seconds("Please try again in 2h5m"), 7500)
        self.assertEqual(retry_after_seconds('"retryDelay": "30s"'), 30)
        self.assertIsNone(retry_after_seconds("Internal server error"))

    def test_daily_quota_stops_extraction_quickly(self):
        cid = self.college["id"]
        for i in range(5):
            self.db.save_page(cid, f"https://example.edu/p{i}", f"P{i}",
                              f"Placement {i}: companies recruited students, package {i} LPA, B.Tech admission.")
        calls = []
        def exhausted(system, prompt):
            calls.append(1)
            raise RuntimeError("Groq API returned status 429: Rate limit reached on tokens per day (TPD): "
                               "Limit 200000, Used 199000. Please try again in 14m2s.")
        from app.config import Config
        Config.EXTRACT_MIN_INTERVAL = 0
        ex = FactExtractor(self.db, exhausted)
        self.assertEqual(ex.extract_college(cid), 0)
        self.assertEqual(len(calls), 1)                 # no retry loop on a daily quota
        self.assertIn("quota used up", ex.stopped_reason)

    def test_relevant_text_keeps_recruiter_list(self):
        page = ("# Welcome\nCampus photos and news.\n\n# Our Recruiters\nInfosys\nZoho\nTCS\n\n"
                "# Gallery\nAnnual day pictures.")
        kept = FactExtractor.relevant_text(page)
        self.assertIn("Zoho", kept)
        self.assertNotIn("Annual day", kept)

    def test_merge_similar_facts(self):
        facts = [
            {"category": "ug_program", "name": "B.E.", "details": ""},
            {"category": "ug_program", "name": "B.E. / B.Tech.", "details": ""},
            {"category": "ug_program", "name": "B.E. (Full Time)", "details": "R-2025"},
            {"category": "ug_program", "name": "B.E. Civil Engineering", "details": "Intake 60"},
            {"category": "ug_program", "name": "B.E. Civil Engineering (Part Time)", "details": "R-2023"},
            {"category": "ug_program", "name": "B.Arch", "details": "5 years"},
            {"category": "pg_program", "name": "MBA", "details": ""},
        ]
        names = [f["name"] for f in merge_similar_facts(facts)]
        self.assertEqual(names, ["B.E. Civil Engineering", "B.Arch", "MBA"])
        civil = merge_similar_facts(facts)[0]
        self.assertIn("Intake 60", civil["details"])
        self.assertIn("R-2023", civil["details"])

    def test_programme_pages_before_syllabus_pdfs(self):
        cid = self.college["id"]
        self.db.save_page(cid, "https://example.edu/files/me-cse-syllabus.pdf", "Syllabus (PDF)",
                          "Course code CS101 programme course credits course outcome. " * 400)
        self.db.save_page(cid, "https://example.edu/admission/programmes-offered", "Programmes Offered",
                          "B.E. Civil Engineering, B.E. Mechanical Engineering, B.Tech Information Technology programmes.")
        self.db.save_page(cid, "https://example.edu/cdc/placement-statistics", "Placement Statistics",
                          "Placement 2025: 92% placed, highest package 40 LPA, recruiters Zoho, TCS.")
        order = [p["url"].rsplit("/", 1)[-1] for p in FactExtractor(self.db, None).select_pages(cid)]
        self.assertEqual(order[-1], "me-cse-syllabus.pdf")
        self.assertIn(order[0], ("programmes-offered", "placement-statistics"))

    def test_messy_programme_names_collapse(self):
        facts = [
            {"category": "ug_program", "name": "B.E. Electronics and Communication Engineering", "details": "Duration 4 years, Intake 180"},
            {"category": "ug_program", "name": "B.E. Electronics and Communication Engineering (Aided)", "details": ""},
            {"category": "ug_program", "name": "B.E. Electronics and Communication Engineering (Accredited SS)", "details": "Code EM"},
            {"category": "ug_program", "name": "BE-Electronics-and-Communication-Engineering-2023-2024 onwards-appended on 28.07.2025", "details": "2023-2024 onwards"},
            {"category": "ug_program", "name": "B.E Computer Science and Engineering (AIML)", "details": ""},
            {"category": "ug_program", "name": "B.E. Computer Science and Engineering (Artificial Intelligence and Machine Learning)", "details": "Intake 60"},
            {"category": "ug_program", "name": "B.Tech. Fashion Technology", "details": ""},
        ]
        merged = merge_similar_facts(facts)
        names = [f["name"] for f in merged]
        self.assertEqual(len(merged), 3)
        self.assertIn("B.E. Electronics and Communication Engineering", names)
        ece = next(f for f in merged if f["name"].startswith("B.E. Electronics"))
        self.assertEqual(ece["details"], "Duration 4 years, Intake 180")    # junk details dropped
        self.assertEqual(names[-1], "B.Tech. Fashion Technology")           # unconfirmed programme listed last

    def test_official_programme_page_wins_and_strays_dropped(self):
        official = "https://x.edu/admission/programmes-offered"
        facts = [
            {"category": "ug_program", "name": "B.E Computer Science and Engineering (AIML)", "details": "4 Years, intake 180",
             "source_url": "https://x.edu/departments/cse/ug"},
            {"category": "ug_program", "name": "B.E. Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
             "details": "Duration 4 years, Intake 60", "source_url": official},
        ] + [{"category": "ug_program", "name": f"B.E. Branch {i}", "details": "Duration 4 years, Intake 60", "source_url": official}
             for i in range(5)] + [
            {"category": "ug_program", "name": "B.Tech. Computer and Bio", "details": "", "source_url": "https://x.edu/old-news"},
        ]
        merged = merge_similar_facts(facts)
        aiml = next(f for f in merged if "AI & ML" in f["name"])
        self.assertEqual(aiml["details"], "Duration 4 years, Intake 60")       # official number, no conflict stacked
        self.assertNotIn("B.Tech. Computer and Bio", [f["name"] for f in merged])  # stray mention left out

    def test_recruiter_noise_cleaned(self):
        merged = merge_similar_facts([
            {"category": "recruiter", "name": "Cognizant", "details": "Program Analyst Trainee, speaker Mr. X"},
            {"category": "recruiter", "name": "Infosys", "details": "Placed B.Tech graduate"},
            {"category": "recruiter", "name": "Over 100 recruiters", "details": "25+ core companies"},
        ])
        recruiters = {f["name"]: f["details"] for f in merged if f["category"] == "recruiter"}
        self.assertEqual(recruiters, {"Cognizant": "", "Infosys": "Placed B.Tech graduate"})
        self.assertIn("Over 100 recruiters", [f["name"] for f in merged if f["category"] == "placement_stat"])

    def test_question_intent(self):
        self.assertEqual(fact_categories_for("tell me about the companies hiring here"), ["recruiter", "placement_stat"])
        self.assertEqual(fact_categories_for("what are the UG courses offered here?"), ["ug_program"])
        self.assertEqual(fact_categories_for("list all courses"), ["ug_program", "pg_program", "doctoral_program"])
        self.assertEqual(fact_categories_for("when was the college founded?"), [])

    def test_answer_lists_all_facts_without_llm(self):
        cid = self.college["id"]
        programs = [{"category": "ug_program", "name": f"B.E. Branch {i}"} for i in range(20)]
        self.db.save_facts(cid, programs, "https://example.edu/courses")
        rag = RAGEngine(self.db)
        rag._get_active_provider = lambda: "fallback"
        ans = rag.answer_question(cid, "What are the UG courses offered here?")
        for i in range(20):
            self.assertIn(f"B.E. Branch {i}", ans["content"])
        self.assertIn("https://example.edu/courses", [c["url"] for c in ans["citations"]])


if __name__ == "__main__":
    unittest.main()
