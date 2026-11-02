import shutil
import unittest
from pathlib import Path

from app.importer import PageImporter
from app.storage import Database

MENU = "<nav><ul><li>Home</li><li>About</li><li>Admissions</li><li>Placements</li><li>Contact</li></ul></nav>"


def html_page(url, title, body):
    return (f"<!-- saved from url=({len(url):04d}){url} -->\n<html><head><title>{title}</title>"
            f"<script>var tracking = 1;</script></head><body>{MENU}{body}<footer>Copyright SSN</footer></body></html>")


def simple_pdf(text):
    """A minimal valid one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class TestImporter(unittest.TestCase):
    def setUp(self):
        self.dir = Path("data/test_import")
        self.files = self.dir / "saved"
        self.files.mkdir(parents=True, exist_ok=True)
        (self.files / "Page_files").mkdir(exist_ok=True)               # browsers' asset folder: must be ignored
        (self.files / "Page_files" / "style.css").write_text("body {}")
        self.db = Database(self.dir / "test.db")
        words = " ".join(["The college offers strong academic programmes."] * 8)

        (self.files / "Admissions.html").write_text(html_page(
            "https://www.ssn.edu.in/admissions", "Admissions | SSN College of Engineering",
            f"<h1>UG Admissions</h1><p>B.E. Computer Science and Engineering, B.E. Mechanical Engineering. {words}</p>"))
        (self.files / "Placements.html").write_text(html_page(
            "https://www.ssn.edu.in/placements", "Placements | SSN College of Engineering",
            f"<h1>Our Recruiters</h1><ul><li>Zoho</li><li>Infosys</li></ul><p>{words}</p>"))
        (self.files / "Contact.html").write_text(html_page(
            "https://www.ssn.edu.in/contact", "Contact | Campus &amp; Trust Office",
            f"<h1>Contact</h1><p>Rajiv Gandhi Salai, Kalavakkam 603110. {words}</p>"))
        (self.files / "Departments.mhtml").write_bytes((
            "From: <Saved by Blink>\r\nSnapshot-Content-Location: https://www.ssn.edu.in/departments\r\n"
            "Subject: Departments\r\nMIME-Version: 1.0\r\nContent-Type: multipart/related; boundary=\"B\"\r\n\r\n"
            "--B\r\nContent-Type: text/html\r\nContent-Location: https://www.ssn.edu.in/departments\r\n\r\n"
            f"<html><head><title>Departments | SSN</title></head><body>{MENU}<h1>Departments</h1>"
            f"<p>Civil, Chemical, Biomedical Engineering. {words}</p></body></html>\r\n--B--\r\n").encode())
        (self.files / "placement-report-2025.pdf").write_bytes(simple_pdf(
            "Placement report 2025: 412 students placed, highest package 41 LPA, recruiters include Zoho and TCS "
            "with many core and software companies visiting the campus every year for final placements "
            "and internships across all engineering departments of the college"))
        (self.files / "notes.txt").write_text("Hostel facilities\n" + " ".join(["Separate hostels for boys and girls."] * 6))
        (self.files / "empty.html").write_text("<html><body><p>Hi</p></body></html>")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_imports_all_file_types(self):
        result = PageImporter(self.db).import_folder(self.files, "https://www.ssn.edu.in", "SSN College of Engineering")
        self.assertEqual(len(result["imported"]), 6)
        self.assertEqual([name for name, _ in result["skipped"]], ["empty.html"])
        pages = {p["url"]: p for p in self.db.get_pages_with_text(result["college_id"])}

        # real page addresses come from the browser's "saved from" comment / MHTML header
        self.assertIn("https://www.ssn.edu.in/admissions", pages)
        self.assertIn("https://www.ssn.edu.in/departments", pages)
        admissions = pages["https://www.ssn.edu.in/admissions"]
        self.assertEqual(admissions["title"], "Admissions | SSN College of Engineering")
        self.assertIn("B.E. Computer Science and Engineering", admissions["clean_text"])
        self.assertNotIn("tracking", admissions["clean_text"])          # scripts removed

        pdf = next(p for p in pages.values() if p["url"].endswith(".pdf"))
        self.assertIn("412 students placed", pdf["clean_text"])

        self.assertEqual(pages["https://www.ssn.edu.in/contact"]["title"], "Contact | Campus & Trust Office")
        college = self.db.get_college(result["college_id"])
        self.assertEqual(college["name"], "SSN College of Engineering")
        self.assertEqual(college["page_count"], 6)
        self.assertEqual(college["status"], "completed")

    def test_menu_repeated_on_every_page_is_dropped(self):
        result = PageImporter(self.db).import_folder(self.files, "https://www.ssn.edu.in")
        contact = next(p for p in self.db.get_pages_with_text(result["college_id"]) if p["url"].endswith("/contact"))
        self.assertIn("Kalavakkam", contact["clean_text"])
        self.assertNotIn("Placements\n", contact["clean_text"])   # nav repeated across the saved pages is stripped
        self.assertNotIn("Copyright SSN", contact["clean_text"])
        everything = " ".join(p["clean_text"] for p in self.db.get_pages_with_text(result["college_id"]))
        self.assertEqual(everything.count("Placements\n"), 1)       # but one copy of the menu is kept

    def test_reimport_does_not_duplicate(self):
        importer = PageImporter(self.db)
        first = importer.import_folder(self.files, "https://www.ssn.edu.in")
        chunks_before = self.db.get_college(first["college_id"])["chunk_count"]
        importer.import_folder(self.files, "https://www.ssn.edu.in")
        college = self.db.get_college(first["college_id"])
        self.assertEqual(college["page_count"], 6)
        self.assertEqual(college["chunk_count"], chunks_before)


if __name__ == "__main__":
    unittest.main()
