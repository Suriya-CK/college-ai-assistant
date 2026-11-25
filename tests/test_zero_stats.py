# Copyright (c) 2026 Suriya C K (github.com/Suriya-CK). All rights reserved.
# Part of College AI Voice Assistant. Unauthorized copying or reuse is prohibited. See LICENSE.
"""Animated counters read before they run ("0 LPA") must not become placement facts."""
import json
import unittest
from app.extractor import FactExtractor, is_zero_stat


class TestZeroStats(unittest.TestCase):
    def test_zero_counters_are_dropped(self):
        for details in ("0 LPA", "₹ 0 Cr", "0 %", "0 +", "0", "₹ 0 LAP", "0.00 LPA"):
            self.assertTrue(is_zero_stat("placement_stat", details), details)

    def test_real_values_are_kept(self):
        for details in ("58 LPA", "860", "47 students (₹5-10 Lakh)", "0.5 LPA", "Released", ""):
            self.assertFalse(is_zero_stat("placement_stat", details), details)
        self.assertFalse(is_zero_stat("fee", "NRI 0"))   # a 0 fee can be real

    def test_parse_facts_skips_zero_stats(self):
        raw = json.dumps({"facts": [
            {"category": "placement_stat", "name": "Highest offer 2023", "details": "0 LPA"},
            {"category": "placement_stat", "name": "Highest offer 2023", "details": "58 LPA"},
        ]})
        self.assertEqual([f["details"] for f in FactExtractor.parse_facts(raw)], ["58 LPA"])


class TestThinkTags(unittest.TestCase):
    def test_reasoning_before_json_is_ignored(self):
        raw = ('<think>The page lists {some} companies, e.g. {"x": 1}. I will output JSON.</think>\n'
               '{"facts": [{"category": "recruiter", "name": "Zoho", "details": "12 offers"}]}')
        self.assertEqual([f["name"] for f in FactExtractor.parse_facts(raw)], ["Zoho"])


if __name__ == "__main__":
    unittest.main()
