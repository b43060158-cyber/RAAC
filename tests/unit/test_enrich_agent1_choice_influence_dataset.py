from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.reasoning_bank_pipeline.enrich_agent1_choice_influence_dataset import (
    enrich_dataset,
    main,
)


def _dataset(record: dict) -> dict:
    return {
        "dataset_purpose": "test",
        "record_count": 1,
        "records": [record],
    }


class EnrichAgent1ChoiceInfluenceDatasetTest(unittest.TestCase):
    def test_enriches_record_with_alignment_evidence(self) -> None:
        payload = _dataset(
            {
                "sample_id": "sample-1",
                "agent1_review_text": (
                    "The peer answer is incorrect because option B ignores the clue; "
                    "the correct answer is option C because it matches the main symptom."
                ),
                "target_previous_reasoning": "Option B seems plausible because of one clue.",
                "target_new_reasoning": "Option C matches the main symptom and better fits the clue.",
                "target_change_summary": "Changed to option C after review pointed out that B ignored the clue.",
                "target_previous_selected_option_ids": ["B"],
                "target_new_selected_option_ids": ["C"],
            }
        )

        enriched = enrich_dataset(payload)
        record = enriched["records"][0]

        self.assertTrue(
            any("matches the main" in phrase or "main symptom" in phrase for phrase in record["review_to_new_reasoning_overlap"])
        )
        self.assertTrue(record["review_targets_previous_error_evidence"])
        self.assertTrue(record["review_proposes_replacement_evidence"])
        self.assertTrue(record["change_summary_alignment_evidence"])
        self.assertTrue(record["alignment_notes"])
        self.assertIn("gold labels", enriched["enrichment_note"])

    def test_handles_sparse_text_without_crashing(self) -> None:
        payload = _dataset(
            {
                "sample_id": "sample-2",
                "agent1_review_text": "",
                "target_previous_reasoning": "",
                "target_new_reasoning": "",
                "target_change_summary": "",
                "target_previous_selected_option_ids": [],
                "target_new_selected_option_ids": [],
            }
        )

        enriched = enrich_dataset(payload)
        record = enriched["records"][0]
        self.assertEqual(record["review_to_new_reasoning_overlap"], [])
        self.assertEqual(record["review_targets_previous_error_evidence"], [])
        self.assertEqual(record["review_proposes_replacement_evidence"], [])
        self.assertEqual(record["change_summary_alignment_evidence"], [])
        self.assertEqual(record["alignment_notes"], [])

    def test_main_writes_enriched_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_payload = _dataset(
                {
                    "sample_id": "sample-3",
                    "agent1_review_text": "Option D is incorrect; option A better matches the finding.",
                    "target_previous_reasoning": "I picked D based on one detail.",
                    "target_new_reasoning": "A better matches the finding.",
                    "target_change_summary": "Moved from D to A after review.",
                    "target_previous_selected_option_ids": ["D"],
                    "target_new_selected_option_ids": ["A"],
                }
            )
            non_input = tmp_path / "non.json"
            adv_input = tmp_path / "adv.json"
            non_output = tmp_path / "non_enriched.json"
            adv_output = tmp_path / "adv_enriched.json"
            non_input.write_text(json.dumps(input_payload, ensure_ascii=False), encoding="utf-8")
            adv_input.write_text(json.dumps(input_payload, ensure_ascii=False), encoding="utf-8")

            import sys

            argv_backup = sys.argv
            try:
                sys.argv = [
                    "scripts/reasoning_bank_pipeline/enrich_agent1_choice_influence_dataset.py",
                    "--non-adversarial-input",
                    str(non_input),
                    "--adversarial-input",
                    str(adv_input),
                    "--non-adversarial-output",
                    str(non_output),
                    "--adversarial-output",
                    str(adv_output),
                ]
                exit_code = main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            self.assertTrue(non_output.exists())
            self.assertTrue(adv_output.exists())


if __name__ == "__main__":
    unittest.main()
