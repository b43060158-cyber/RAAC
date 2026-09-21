from __future__ import annotations

import unittest

from scripts.reasoning_bank_pipeline.build_adversarial_reasoning_bank_v2 import (
    build_reasoning_bank_v2,
)


class BuildAdversarialReasoningBankV2Test(unittest.TestCase):
    def test_build_reasoning_bank_v2_preserves_question_and_target_fields(self) -> None:
        records = build_reasoning_bank_v2(
            attribution_records=[
                {
                    "sample_id": "run::truthfulqa__0342__single_choice::agent_3::1->2::2",
                    "question_key": "truthfulqa__0342__single_choice",
                    "question_id": "0342",
                    "dataset_name": "truthfulqa",
                    "run_type": "adversarial",
                    "target_agent_id": "agent_3",
                    "question_text": "Question?",
                    "options": [{"option_id": "A", "text": "Alpha"}],
                    "target_previous_selected_option_ids": ["A"],
                    "target_previous_reasoning": "old",
                    "primary_factor_id": "factor_01",
                    "secondary_factor_ids": ["factor_02"],
                    "supporting_evidence": ["span"],
                    "review_to_new_reasoning_overlap": ["alpha"],
                    "review_targets_previous_error_evidence": ["old error"],
                    "review_proposes_replacement_evidence": ["choose B"],
                    "change_summary_alignment_evidence": ["changed"],
                    "factor_mapping_confidence": 0.9,
                }
            ],
            malicious_records=[
                {
                    "sample_id": "run::truthfulqa__0342__single_choice::agent_3::1->2::2",
                    "input": {"desired_target_shift": {"to": ["B"]}},
                    "neutralized_review_rationale": "neutral",
                    "malicious_adapted_review_rationale": "malicious",
                    "malicious_strategy_notes": "notes",
                }
            ],
            enriched_records={},
            corpus_id="test_bank",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question_id"], "0342")
        self.assertEqual(records[0]["target_agent_id"], "agent_3")


if __name__ == "__main__":
    unittest.main()
