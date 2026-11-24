# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""Text is rewritten the way a person would say it before the voice reads it."""
import tempfile
import unittest
from pathlib import Path
from app.voice_service import VoiceService


class TestSpokenForms(unittest.TestCase):
    def setUp(self):
        self.voice = VoiceService(Path(tempfile.mkdtemp()))

    def say(self, text):
        return self.voice.clean_text_for_speech(text)

    def test_money_and_packages(self):
        self.assertEqual(self.say("The highest offer is 58 LPA."), "The highest offer is 58 lakh per annum.")
        self.assertEqual(self.say("Top package ₹ 1.17 Cr"), "Top package 1.17 crore rupees")
        self.assertEqual(self.say("Fee is ₹850 per year"), "Fee is 850 rupees per year")
        self.assertEqual(self.say("Dream offers pay 5-10 lakh"), "Dream offers pay 5 to 10 lakh")

    def test_degrees_and_symbols(self):
        self.assertEqual(self.say("B.E. and M.Tech courses"), "B E and M Tech courses")
        self.assertEqual(self.say("98% placed in R&D & IT"), "98 percent placed in R&D and IT")

    def test_dates_are_left_alone(self):
        self.assertEqual(self.say("Apply by 23-11-2026."), "Apply by 23-11-2026.")

    def test_same_text_same_cache_name(self):
        self.assertEqual(self.voice.cache_name("Hello", "en-IN-NeerjaNeural"),
                         self.voice.cache_name("Hello", "en-IN-NeerjaNeural"))
        self.assertNotEqual(self.voice.cache_name("Hello", "en-IN-NeerjaNeural"),
                            self.voice.cache_name("Hello", "en-IN-PrabhatNeural"))


if __name__ == "__main__":
    unittest.main()
