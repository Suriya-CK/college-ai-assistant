# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
import unittest
from app.placement_report import (analyse_document, parse_nirf, parse_rows, report_year, save_document,
                                  summarise, summarise_nirf)

REPORT = """PLACEMENT AND 6 MONTHS INTERNSHIP (2025 - 2026)
S.
No
Reg. No. Name Degree Branch College Level Company
1 9100000001 Meena S B.Tech Apparel Technology ACT UG Aquarelle India
2 9100000002 Karthik R B.Tech Petroleum Engineering &
Technology ACT UG Petronash
3 9100000003 Vimal D B.E Computer Science and
Engineering CEG UG Amazon
4 9100000004 Lakshmi P M.E Computer Science and Engineering CEG PG Amazon
5 9100000005 Rahul T B.E Mechanical Engineering MIT UG Alstom (Drive 2)
6 9100000006 Nisha G B.E Mechanical Engineering MIT UG Alstom
"""


class TestPlacementReport(unittest.TestCase):
    def test_rows_including_wrapped_ones(self):
        rows = parse_rows(REPORT)
        self.assertEqual(len(rows), 6)
        self.assertEqual({k: rows[1][k] for k in ("college", "level", "company")},
                         {"college": "ACT", "level": "UG", "company": "Petronash"})
        self.assertEqual(rows[2]["company"], "Amazon")
        self.assertEqual(rows[4]["company"], "Alstom")   # "(Drive 2)" is the same company

    def test_summary_counts_and_no_personal_data(self):
        facts, page = summarise(parse_rows(REPORT), "2025-26", "report.pdf")
        recruiters = {f["name"]: f["details"] for f in facts if f["category"] == "recruiter"}
        self.assertEqual(recruiters["Amazon"], "2 offers (2025-26); by campus: CEG 2")
        self.assertEqual(recruiters["Alstom"], "2 offers (2025-26); by campus: MIT 2")
        stat = next(f for f in facts if f["category"] == "placement_stat" and f["name"].startswith("Placement offers"))
        self.assertIn("6 offers from 4 companies (5 UG, 1 PG)", stat["details"])
        stored = page + " ".join(f"{f['name']} {f['details']}" for f in facts)
        for personal in ("Meena", "Vimal", "Lakshmi", "9100000001", "9100000004"):
            self.assertNotIn(personal, stored)

    def test_year_from_title(self):
        self.assertEqual(report_year("PLACEMENT AND 6 MONTHS INTERNSHIP (2025 - 2026).pdf"), "2025-26")
        self.assertEqual(report_year("placements.pdf"), "")


DEPARTMENT_TABLE = """GOVERNMENT COLLEGE OF ENGINEERING - BARGUR
CAREER DEVELOPMENT CELL
PLACEMENT DETAILS - AY (2024-2025)
S.NO REGISTER
NUMBER
NAME COMPANY
MECHANICAL ENGINEERING
1 91000000101 MANI K S FOXCONN
2 91000000102 RAVI N ZOHO
COMPUTER SCIENCE AND ENGINEERING
1 91000000103 DEEPA M FOXCONN, QSPIDERS
2 91000000104 SURESH B CTS
(Dr.A.Kumar, AP / ECE)                 (Dr.B.Raman, Professor/ EEE)
"""

NIRF = """Placement & Higher Studies
UG [4 Years Program(s)]: Placement & higher studies for previous 3 years
Academic Year No. of first year students intake ... Median salary of placed graduates(Amount in Rs.) No. of students
selected for Higher Studies
2018-19 240 209 2019-20 56 2021-22 223 48 350000(Three lakhs
fifty thousand only)
2
2020-21 240 227 2021-22 45 2023-24 225 69 1000000(TEN LAKHS) 1
PG [2 Years Program(s)]: Placement & higher studies for previous 3 years
2022-23 54 3 2023-24 3 0 0(Zero) 1
Ph.D Student Details (including Integrated Ph.D)
"""


class TestOtherLayouts(unittest.TestCase):
    def test_department_table(self):
        rows = parse_rows(DEPARTMENT_TABLE)
        self.assertEqual([r["company"] for r in rows], ["FOXCONN", "ZOHO", "FOXCONN", "QSPIDERS", "CTS"])
        self.assertEqual(rows[0]["department"], "Mechanical Engineering")
        facts, page = summarise(rows, "2024-25", "placement details.pdf")
        stat = next(f for f in facts if f["name"].startswith("Placement offers"))
        self.assertEqual(stat["details"], "5 offers from 4 companies, to 4 students")
        foxconn = next(f for f in facts if f["name"] == "FOXCONN")
        self.assertIn("by department: Mechanical Engineering 1, Computer Science and Engineering 1", foxconn["details"])
        for personal in ("MANI", "DEEPA", "91000000101", "Raman"):
            self.assertNotIn(personal, page + str(facts))

    def test_nirf(self):
        rows = parse_nirf(NIRF)
        self.assertEqual([(r["level"], r["graduating_year"], r["placed"], r["graduated"], r["median"]) for r in rows],
                         [("UG", "2021-22", 48, 223, 350000), ("UG", "2023-24", 69, 225, 1000000),
                          ("PG", "2023-24", 0, 3, 0)])
        facts, _ = summarise_nirf(rows, "2025")
        self.assertEqual(facts[0]["details"],
                         "48 of 223 graduates placed (22%); median salary ₹3.5 lakh; 2 went for higher studies (NIRF 2025)")


class TestAnalyseAndSave(unittest.TestCase):
    def test_recognises_documents_and_ignores_others(self):
        self.assertEqual(analyse_document(NIRF, "Nirf 2025.pdf")["kind"], "nirf")
        self.assertEqual(analyse_document(NIRF, "Nirf 2025.pdf")["year"], "2025")
        self.assertEqual(analyse_document(REPORT, "PLACEMENT (2025 - 2026).pdf")["kind"], "table")
        self.assertIsNone(analyse_document("Academic regulations 2025. Attendance must be 75%.", "reg.pdf"))
        # A short table (e.g. 2 rows of contacts) is not a placement list
        self.assertIsNone(analyse_document("1 9100000001 Meena S B.Tech Apparel ACT UG Aquarelle", "x.pdf"))

    def test_newest_list_wins_and_no_names_are_stored(self):
        import tempfile
        from pathlib import Path
        from app.storage import Database
        from app.crawler import CollegeCrawler
        db = Database(Path(tempfile.mkdtemp()) / "t.db")
        cid = db.create_or_get_college("https://x.edu", "x.edu", "X")["id"]
        crawler = CollegeCrawler(db)
        new = analyse_document(DEPARTMENT_TABLE.replace("ZOHO", "TCS") + "3 91000000105 VIJAY L CTS\n", "list 2024-25.pdf")
        old = analyse_document(DEPARTMENT_TABLE, "list 2023-24.pdf")
        save_document(db, cid, "https://x.edu/2024.pdf", new, crawler)
        save_document(db, cid, "https://x.edu/2023.pdf", old, crawler)
        cts = next(f for f in db.get_facts(cid, ["recruiter"]) if f["name"] == "CTS")
        self.assertIn("(2024-25)", cts["details"])                     # the first-saved (newest) list kept
        pages = " ".join(p["clean_text"] for p in db.get_pages_with_text(cid))
        self.assertNotIn("MANI", pages)
        self.assertNotIn("91000000101", pages)


if __name__ == "__main__":
    unittest.main()
