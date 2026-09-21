from __future__ import annotations

import unittest

from scripts.analyze_agent1_attribution import build_analysis


class AnalyzeAgent1AttributionTest(unittest.TestCase):
    def test_build_analysis_groups_counts_and_representatives(self) -> None:
        taxonomy = {
            "taxonomy_version": "v1",
            "factors": [
                {"factor_id": "factor_01", "factor_name": "Direct contradiction", "definition": "D1"},
                {"factor_id": "factor_02", "factor_name": "Replacement", "definition": "D2"},
            ],
        }
        records = [
            {
                "sample_id": "s1",
                "primary_factor_id": "factor_01",
                "secondary_factor_ids": ["factor_02"],
                "run_type": "adversarial",
                "dataset_name": "medmcqa",
                "factor_mapping_confidence": 0.9,
                "question_key": "q1",
                "supporting_evidence": ["e1"],
                "mapping_notes": "m1",
                "agent1_review_text": "r1",
            },
            {
                "sample_id": "s2",
                "primary_factor_id": "factor_01",
                "secondary_factor_ids": [],
                "run_type": "non_adversarial",
                "dataset_name": "truthfulqa",
                "factor_mapping_confidence": 0.8,
                "question_key": "q2",
                "supporting_evidence": ["e2"],
                "mapping_notes": "m2",
                "agent1_review_text": "r2",
            },
            {
                "sample_id": "s3",
                "primary_factor_id": "factor_02",
                "secondary_factor_ids": [],
                "run_type": "adversarial",
                "dataset_name": "medmcqa",
                "factor_mapping_confidence": 0.7,
                "question_key": "q3",
                "supporting_evidence": ["e3"],
                "mapping_notes": "m3",
                "agent1_review_text": "r3",
            },
        ]

        analysis = build_analysis(records, taxonomy, representative_limit=2)

        self.assertEqual(analysis["source_record_count"], 3)
        self.assertEqual(analysis["primary_factor_frequency"]["factor_01"], 2)
        self.assertEqual(analysis["primary_factor_by_run_type"]["adversarial"]["factor_01"], 1)
        self.assertEqual(analysis["primary_factor_by_dataset"]["medmcqa"]["factor_02"], 1)
        self.assertEqual(analysis["secondary_factor_frequency"]["factor_02"], 1)
        self.assertEqual(analysis["factor_details"][0]["factor_id"], "factor_01")
        self.assertEqual(len(analysis["factor_details"][0]["representative_samples"]), 2)


if __name__ == "__main__":
    unittest.main()
