import unittest
from collections import Counter
from app.crawler import CollegeCrawler

class TestCrawler(unittest.TestCase):
    def setUp(self):
        self.crawler = CollegeCrawler()

    def test_normalize_url(self):
        url = "https://www.stanford.edu/admissions/#deadline"
        normalized = self.crawler.normalize_url(url)
        self.assertEqual(normalized, "https://www.stanford.edu/admissions")

    def test_dedupe_key_merges_variants(self):
        key = self.crawler.dedupe_key
        self.assertEqual(key("http://cac.annauniv.edu/"), key("https://cac.annauniv.edu/index.php"))
        self.assertEqual(key("https://www.ssn.edu.in/about/"), key("http://ssn.edu.in/about"))
        self.assertNotEqual(key("https://x.edu/dept?id=1"), key("https://x.edu/dept?id=2"))

    def test_is_same_domain(self):
        base_domain = "mit.edu"
        self.assertTrue(self.crawler.is_same_domain(base_domain, "https://admissions.mit.edu/apply"))
        self.assertTrue(self.crawler.is_same_domain(base_domain, "https://www.mit.edu/about"))
        self.assertFalse(self.crawler.is_same_domain(base_domain, "https://harvard.edu/"))
        # A specific college subdomain does not leak into sibling portals
        self.assertFalse(self.crawler.is_same_domain("ceg.annauniv.edu", "https://acoe.annauniv.edu/"))
        self.assertTrue(self.crawler.is_same_domain("www.ssn.edu.in", "https://admissions.ssn.edu.in/"))

    def test_score_prefers_topic_pages(self):
        score = self.crawler.score_url
        self.assertGreater(score("https://x.edu/placements", 1), score("https://x.edu/gallery", 1))
        self.assertGreater(score("https://x.edu/page?id=7", 1, "UG Courses"), score("https://x.edu/page?id=8", 1, "Photos"))
        self.assertGreater(score("https://x.edu/admission", 1), score("https://x.edu/admission", 3))
        # "ug" must be a whole word, not part of "august"
        self.assertEqual(score("https://x.edu/august-update", 0), 0)

    def test_skips_logins_and_media(self):
        self.assertFalse(self.crawler._is_crawlable("https://x.edu/login?next=/"))
        self.assertFalse(self.crawler._is_crawlable("https://x.edu/campus.jpg"))
        self.assertTrue(self.crawler._is_crawlable("https://x.edu/brochure.pdf"))
        self.assertTrue(self.crawler._is_crawlable("https://x.edu/courses"))

    def test_hidden_links(self):
        html = """
        <meta http-equiv="refresh" content="0; url=/home/index.html">
        <frameset><frame src="menu.html"><frame src="main.php"></frameset>
        <script>window.location.href = "https://x.edu/new-site";</script>
        """
        links = self.crawler.extract_hidden_links(html, "https://x.edu/")
        self.assertIn("https://x.edu/home/index.html", links)
        self.assertIn("https://x.edu/menu.html", links)
        self.assertIn("https://x.edu/main.php", links)
        self.assertIn("https://x.edu/new-site", links)

    def test_soft_404(self):
        menu = "Home About Admissions Departments Placements Contact Library Hostel Research Alumni " * 3
        site = self.crawler._word_set(menu + " Welcome to our college")
        error = "Oops the page you requested could not be located please go home"
        not_found = self.crawler._word_set(menu + error) - site
        is_404 = self.crawler.is_soft_404
        self.assertTrue(is_404("Page Not Found", "Sorry", 200, set()))
        self.assertTrue(is_404("About", "x", 404, set()))
        self.assertTrue(is_404("College", menu + error, 200, not_found, site))
        # A short real page that shares the site menu is NOT an error page
        self.assertFalse(is_404("Contact", menu + "Phone 044 2745 0000 email principal@x.edu address Chennai Tamil Nadu",
                                200, not_found, site))
        self.assertFalse(is_404("Placements", menu + "Placement statistics 2024: Infosys, TCS and Zoho recruited 300 students.",
                                200, not_found, site))

    def test_college_name_from_title(self):
        name = self.crawler.college_name_from_title
        self.assertEqual(name("Home | CEG Campus"), "CEG Campus")
        self.assertEqual(name("Anna University, Chennai"), "Anna University, Chennai")
        self.assertEqual(name("SSN College of Engineering, Chennai | Top-Ranked Engineering College"),
                         "SSN College of Engineering, Chennai")
        self.assertEqual(name("Welcome - XYZ Institute of Technology"), "XYZ Institute of Technology")

    def test_is_blocked(self):
        class R:
            def __init__(self, code, err=None):
                self.status_code, self.error_message = code, err
        self.assertTrue(self.crawler.is_blocked(R(403)))
        self.assertTrue(self.crawler.is_blocked(R(None, "Blocked by anti-bot protection: HTTP 403")))
        self.assertFalse(self.crawler.is_blocked(R(404)))
        self.assertFalse(self.crawler.is_blocked(R(200)))

        class Page:
            def __init__(self, title, text):
                self.status_code, self.error_message = 200, None
                self.metadata = {"title": title}
                self.markdown = type("M", (), {"raw_markdown": text})()
        captcha = Page("Validation request", "User validation required to continue.. Please type the text you see in the image")
        self.assertTrue(self.crawler.is_blocked(captcha))
        self.assertTrue(self.crawler.is_blocked(Page("Just a moment...", "Checking your browser before accessing")))
        self.assertFalse(self.crawler.is_blocked(Page("Admissions", "Apply for B.E. programmes before 30 June.")))
        long_page = "Library rules. " + "Students must carry ID cards at all times. " * 40 + "Access denied to unregistered visitors."
        self.assertFalse(self.crawler.is_blocked(Page("Library", long_page)))

    def test_clean_markdown_and_sections(self):
        md = (
            "![logo](/logo.png)\n# Welcome to Test University\n"
            "We offer [Computer Science and Engineering](/cse) and more.\n\n"
            "## Tuition and Fees\nThe annual tuition for full-time students is $45,000.\n"
        )
        text = self.crawler.clean_markdown(md)
        self.assertNotIn("logo.png", text)
        self.assertIn("Computer Science and Engineering and more", text)
        sections = self.crawler.markdown_to_sections(text)
        self.assertEqual([s["heading"] for s in sections], ["Welcome to Test University", "Tuition and Fees"])
        self.assertIn("$45,000", sections[1]["text"])

    def test_boilerplate_lines_removed_after_repeats(self):
        counts = Counter()
        menu = "Home\nAbout Us\nContact"
        for i in range(3):
            self.crawler._strip_boilerplate(f"{menu}\nUnique page {i} content", counts)
        stripped = self.crawler._strip_boilerplate(f"{menu}\nPlacement records 2024", counts)
        self.assertNotIn("About Us", stripped)
        self.assertIn("Placement records 2024", stripped)

    def test_create_chunks(self):
        sections = [
            {"heading": "Admissions", "text": "Admissions are open for fall 2026. Submit your applications by December 15."},
            {"heading": "Tuition", "text": "Tuition fee is $30,000 per year with numerous scholarship options."}
        ]
        chunks = self.crawler.create_chunks(sections, "Test University")
        self.assertEqual(len(chunks), 2)
        self.assertIn("December 15", chunks[0]["text"])

    def test_long_section_is_split_with_overlap(self):
        words = " ".join(f"w{i}" for i in range(900))
        chunks = self.crawler.create_chunks([{"heading": "H", "text": words}], "T")
        self.assertEqual(len(chunks), 3)
        self.assertIn("w899", chunks[-1]["text"])

if __name__ == "__main__":
    unittest.main()
