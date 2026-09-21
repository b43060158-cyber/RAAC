from __future__ import annotations

import unittest

from scripts.reasoning_bank_pipeline.build_agent1_attribution_candidates import (
    build_candidates,
)
from scripts.reasoning_bank_pipeline.build_agent1_review_generation_dataset import (
    _generation_record,
)


class Agent1InfluencePipelineTest(unittest.TestCase):
    def test_build_candidates_preserves_alignment_fields(self) -> None:
        non_payload = {
            "records": [
                {
                    "sample_id": "s1",
                    "run_id": "run1",
                    "run_type": "non_adversarial",
                    "dataset_name": "medmcqa",
                    "task_type": "single_choice",
                    "effective_task_type": "single_choice",
                    "question_key": "q1",
                    "question_id": "551",
                    "target_agent_id": "agent_2",
                    "question_text": "Question?",
                    "options": [{"option_id": "A", "text": "Alpha"}],
                    "target_previous_selected_option_ids": ["B"],
                    "target_previous_reasoning": "old",
                    "agent1_review_text": "review",
                    "agent1_review_stance": "oppose",
                    "agent1_review_score": 3,
                    "target_new_selected_option_ids": ["A"],
                    "target_new_reasoning": "new",
                    "target_change_summary": "changed",
                    "review_to_new_reasoning_overlap": ["alpha"],
                    "review_targets_previous_error_evidence": ["old wrong"],
                    "review_proposes_replacement_evidence": ["A better"],
                    "change_summary_alignment_evidence": ["changed"],
                    "alignment_notes": ["note"],
                }
            ]
        }
        adv_payload = {"records": []}
        candidates = build_candidates(non_payload, adv_payload)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_split"], "non_adversarial")
        self.assertEqual(candidates[0]["review_to_new_reasoning_overlap"], ["alpha"])
        self.assertEqual(candidates[0]["question_id"], "551")
        self.assertEqual(candidates[0]["target_agent_id"], "agent_2")

    def test_generation_record_uses_standardized_factor_fields(self) -> None:
        record = _generation_record(
            {
                "sample_id": "s2",
                "question_text": "Q",
                "options": [{"option_id": "A", "text": "Alpha"}],
                "target_previous_selected_option_ids": ["B"],
                "target_previous_reasoning": "old",
                "target_new_selected_option_ids": ["A"],
                "primary_factor_id": "factor_01",
                "secondary_factor_ids": ["factor_02"],
                "supporting_evidence": ["span"],
                "run_type": "adversarial",
                "dataset_name": "truthfulqa",
                "agent1_review_text": "review",
                "question_key": "qk",
                "target_agent_id": "agent_2",
                "factor_mapping_confidence": 0.8,
                "mapping_notes": "ok",
            }
        )
        self.assertEqual(record["input"]["primary_factor_id"], "factor_01")
        self.assertEqual(record["input"]["desired_target_shift"]["to"], ["A"])
        self.assertEqual(record["output"]["review_rationale"], "review")


if __name__ == "__main__":
    unittest.main()
