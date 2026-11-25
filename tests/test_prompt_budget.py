# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""Prompt trimming: questions send only the facts and page lines they need (Groq free-tier token limits)."""
import unittest
from app.rag_engine import (
    MAX_CHUNK_CHARS, MAX_FACTS_FOR_SINGLE_ANSWER, MAX_FACTS_PER_CATEGORY_IN_LIST, RAGEngine, select_facts,
    trim_chunk_text,
)


def fact(category, name, details=""):
    return {"category": category, "name": name, "details": details, "source_url": "https://x.edu/a"}


class TestSelectFacts(unittest.TestCase):
    def setUp(self):
        self.facts = [fact("admission", f"Admission note {i}", "General information") for i in range(50)]
        self.facts.insert(30, fact("fee", "Application fee", "B.E. Tamil Nadu 850, Other States 850"))

    def test_single_answer_keeps_the_matching_fact(self):
        kept, complete = select_facts(self.facts, "What is the application fee for B.E. admission?")
        self.assertFalse(complete)
        self.assertEqual(len(kept), MAX_FACTS_FOR_SINGLE_ANSWER)
        self.assertIn("Application fee", [f["name"] for f in kept])

    def test_kept_facts_stay_in_original_order(self):
        kept, _ = select_facts(self.facts, "What is the application fee?")
        positions = [self.facts.index(f) for f in kept]
        self.assertEqual(positions, sorted(positions))

    def test_list_question_keeps_everything(self):
        for q in ("List all admission rules", "Which departments are there?", "What courses are offered?"):
            kept, complete = select_facts(self.facts, q)
            self.assertTrue(complete, q)
            self.assertEqual(len(kept), len(self.facts), q)

    def test_long_category_keeps_top_entries_in_a_list_question(self):
        companies = [fact("recruiter", f"Company {i}", f"{i % 20 + 1} offers (2025-26)") for i in range(250)]
        companies.append(fact("recruiter", "Amazon", "56 offers (2025-26)"))
        kept, complete = select_facts(companies, "Which companies hire here?")
        self.assertFalse(complete)
        self.assertEqual(len(kept), MAX_FACTS_PER_CATEGORY_IN_LIST)
        counts = [int(f["details"].split()[0]) for f in kept]
        self.assertEqual(counts, sorted(counts, reverse=True))        # biggest recruiters first
        self.assertIn("Amazon", [f["name"] for f in kept])
        # A named company is found even when it's not among the biggest
        kept, _ = select_facts(companies, "Does Company 96 recruit students?")
        self.assertIn("Company 96", [f["name"] for f in kept])
        header, _ = RAGEngine._build_facts_context(kept[:5], False, {"recruiter": 251})
        self.assertIn("top 5 of 251 items", header)

    def test_short_list_is_not_trimmed(self):
        few = self.facts[:10]
        self.assertEqual(select_facts(few, "What is the fee?"), (few, True))


class TestTrimChunk(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(trim_chunk_text("Title\nShort page", "fee"), "Title\nShort page")

    def test_long_text_keeps_title_and_matching_lines(self):
        filler = [f"Line {i} about campus life and student clubs and events." for i in range(100)]
        text = "\n".join(["Admissions 2026"] + filler[:70] + ["Application fee:", "Rs. 850 for all candidates"] + filler[70:])
        out = trim_chunk_text(text, "What is the application fee?")
        self.assertLessEqual(len(out), MAX_CHUNK_CHARS)
        self.assertTrue(out.startswith("Admissions 2026"))
        self.assertIn("Application fee:", out)
        self.assertIn("Rs. 850", out)   # the line after a matching line comes along


class TestFactsContextWording(unittest.TestCase):
    def test_complete_flag_changes_header(self):
        facts = [fact("fee", "Application fee", "850")]
        full, _ = RAGEngine._build_facts_context(facts, True)
        partial, _ = RAGEngine._build_facts_context(facts, False)
        self.assertIn("complete list", full)
        self.assertIn("most relevant", partial)


class TestSpokenSummarySplit(unittest.TestCase):
    def prep(self):
        return {"query": "q", "conversation_id": None, "target_lang": "en", "citations": [], "college_id": 1,
                "college_name": "X", "cache_key": "k",
                "lang_info": {"name": "English", "speech_locale": "en-IN", "default_voice": "v"}}

    def test_missing_closing_tag_is_not_shown(self):
        rag = RAGEngine.__new__(RAGEngine)
        rag._answer_cache = {}
        out = rag._finish_answer(self.prep(), "The fee is 850.\n\n---\n[SPOKEN_SUMMARY]\nThe fee is 850 rupees.", "groq")
        self.assertNotIn("SPOKEN_SUMMARY", out["content"])
        self.assertEqual(out["content"], "The fee is 850.")
        self.assertEqual(out["spoken_text"], "The fee is 850 rupees.")


class TestRepetitionGuard(unittest.TestCase):
    def test_loop_detected_on_repeated_lines(self):
        from app.rag_engine import is_looping
        self.assertTrue(is_looping("Intro\n* iGATE (core)\n* iGATE (core)\n* iGATE (core)\n* iG"))
        self.assertFalse(is_looping("* Amazon\n* Google\n* Microsoft\n"))
        self.assertFalse(is_looping("* iGATE\n* iGATE\n"))      # two repeats are not a loop yet

    def test_repeated_items_removed_but_structure_kept(self):
        from app.rag_engine import remove_repeated_lines
        text = "## Companies\n\n* Amazon\n* Google\n* Amazon\n\n| A | B |\n|---|---|\n| x | 1 |\n| x | 1 |\n\n## Next\n"
        self.assertEqual(remove_repeated_lines(text),
                         "## Companies\n\n* Amazon\n* Google\n\n| A | B |\n|---|---|\n| x | 1 |\n\n## Next\n")


class TestSpokenSummaryVariants(unittest.TestCase):
    def finish(self, raw):
        rag = RAGEngine.__new__(RAGEngine)
        rag._answer_cache = {}
        prep = TestSpokenSummarySplit.prep(None)
        return rag._finish_answer(prep, raw, "groq")

    def test_bold_and_heading_tags(self):
        for raw in ("Top offer is 58 LPA.\n\n---\n\n**SPOKEN_SUMMARY**\nThe top offer is 58 lakh.",
                    "Top offer is 58 LPA.\n\n## Spoken summary:\nThe top offer is 58 lakh.",
                    "Top offer is 58 LPA.\n[SPOKEN_SUMMARY]\nThe top offer is 58 lakh.\n[/SPOKEN_SUMMARY]",
                    "Top offer is 58 LPA.\n\n---\n\n**[SPOKEN_SUMMARY]**\nThe top offer is 58 lakh.\n**[/SPOKEN_SUMMARY]**"):
            out = self.finish(raw)
            self.assertEqual(out["content"], "Top offer is 58 LPA.", raw)
            self.assertEqual(out["spoken_text"], "The top offer is 58 lakh.", raw)


class TestAnswerLanguage(unittest.TestCase):
    def test_script_check(self):
        from app.rag_engine import in_language
        tamil = ("## இளநிலை படிப்புகள்\nஇந்த கல்லூரியில் ஐந்து படிப்புகள் உள்ளன.\n"
                 "- **B.E. Computer Science and Engineering**\n- **B.E. Cyber Security** (2025-2026)\n"
                 "| B.E. Electrical and Electronics Engineering | 1994-1995 |")
        english = "## Undergraduate Programmes\nThe college offers five courses.\n- **B.E. Cyber Security**"
        self.assertTrue(in_language(tamil, "ta"))          # English course names in lists are fine
        self.assertFalse(in_language(english, "ta"))       # the bug: an English answer to a Tamil question
        self.assertTrue(in_language(english, "en"))
        self.assertFalse(in_language(tamil, "en"))

    def test_language_line_names_the_language(self):
        from app.rag_engine import answer_language_line
        line = answer_language_line("ta", {"name": "Tamil", "native": "தமிழ்"})
        self.assertIn("Tamil (தமிழ்)", line)
        self.assertIn("English", answer_language_line("en", {}))


if __name__ == "__main__":
    unittest.main()
