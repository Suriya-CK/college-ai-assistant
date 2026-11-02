import unittest
from app.multilingual import (
    SUPPORTED_LANGUAGES,
    detect_language,
    get_english_search_terms,
    get_all_voices
)
from app.storage import Database
from app.rag_engine import RAGEngine
from app.voice_service import VoiceService


class TestMultilingual(unittest.TestCase):

    def test_language_detection(self):
        # Tamil
        self.assertEqual(detect_language("அண்ணா பல்கலைக்கழகம் எப்போது தொடங்கப்பட்டது?"), "ta")
        # Hindi
        self.assertEqual(detect_language("अन्ना विश्वविद्यालय में कौन से पाठ्यक्रम हैं?"), "hi")
        # Telugu
        self.assertEqual(detect_language("ఈ కళాశాల ప్రవేశ ప్రక్రియ ఏమిటి?"), "te")
        # Kannada
        self.assertEqual(detect_language("ಕಾಲೇಜಿನ ಬಗ್ಗೆ ಮಾಹಿತಿ ನೀಡಿ"), "kn")
        # Malayalam
        self.assertEqual(detect_language("ഈ കോളേജിനെക്കുറിച്ച് പറയുക"), "ml")
        # Bengali
        self.assertEqual(detect_language("এই কলেজ সম্পর্কে বলুন"), "bn")
        # Gujarati
        self.assertEqual(detect_language("કોલેજ વિશે જણાવો"), "gu")
        # English
        self.assertEqual(detect_language("Tell me about Anna University"), "en")

    def test_regional_keyword_mapping(self):
        # Tamil admissions
        ta_terms = get_english_search_terms("சேர்க்கை மற்றும் விண்ணப்பிக்கும் முறை", "ta")
        self.assertTrue(any(w in ta_terms for w in ["admission", "apply", "eligibility"]))

        # Tamil courses
        ta_courses = get_english_search_terms("இங்கு என்ன படிப்புகள் உள்ளன?", "ta")
        self.assertTrue(any(w in ta_courses for w in ["courses", "programs", "degrees"]))

        # Hindi fees
        hi_fees = get_english_search_terms("यहाँ फीस और शुल्क कितना है?", "hi")
        self.assertTrue(any(w in hi_fees for w in ["tuition", "fees", "cost"]))

        # Hindi contact
        hi_contact = get_english_search_terms("कॉलेज का पता और संपर्क विवरण क्या है?", "hi")
        self.assertTrue(any(w in hi_contact for w in ["contact", "address", "directory"]))

    def test_voice_service_multilingual_voices(self):
        vs = VoiceService()
        voices = vs.get_available_voices()
        self.assertGreaterEqual(len(voices), 15)

        # Check for Indian neural voices
        voice_ids = {v["id"] for v in voices}
        self.assertIn("ta-IN-PallaviNeural", voice_ids)
        self.assertIn("hi-IN-SwaraNeural", voice_ids)
        self.assertIn("te-IN-ShrutiNeural", voice_ids)
        self.assertIn("en-IN-NeerjaNeural", voice_ids)

        grouped = vs.get_voices_by_language()
        self.assertIn("ta", grouped)
        self.assertIn("hi", grouped)
        self.assertEqual(grouped["ta"]["speech_locale"], "ta-IN")
        self.assertEqual(grouped["hi"]["speech_locale"], "hi-IN")

    def test_rag_regional_response_metadata(self):
        db = Database()
        colleges = db.list_colleges()
        if not colleges:
            self.skipTest("No colleges in database to test RAG")

        cid = colleges[0]["id"]
        rag = RAGEngine(db)

        # Tamil query
        res_ta = rag.answer_question(cid, "அண்ணா பல்கலைக்கழகம் எப்போது தொடங்கப்பட்டது?", language="ta")
        self.assertEqual(res_ta["language"], "ta")
        self.assertEqual(res_ta["speech_locale"], "ta-IN")
        self.assertEqual(res_ta["default_voice"], "ta-IN-PallaviNeural")
        self.assertIn("அதிகாரப்பூர்வ", res_ta["content"])

        # Hindi query
        res_hi = rag.answer_question(cid, "विश्वविद्यालय के बारे में बताइए", language="hi")
        self.assertEqual(res_hi["language"], "hi")
        self.assertEqual(res_hi["speech_locale"], "hi-IN")
        self.assertEqual(res_hi["default_voice"], "hi-IN-SwaraNeural")
        self.assertIn("आधिकारिक", res_hi["content"])


if __name__ == "__main__":
    unittest.main()
