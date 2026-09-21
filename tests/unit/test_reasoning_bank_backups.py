from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.reasoning_bank_pipeline import (
    build_adversarial_reasoning_bank_v2 as build_script,
)
from scripts.reasoning_bank_pipeline.reasoning_bank_backup_utils import (
    create_reasoning_bank_backup_bundle,
)


class ReasoningBankBackupTest(unittest.TestCase):
    def test_backup_utility_archives_target_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "adversarial_reasoning_bank_v2.jsonl"
            source = root / "agent1_influence_standardized_attribution.jsonl"
            target.write_text("old-bank\n", encoding="utf-8")
            source.write_text("source\n", encoding="utf-8")

            archive_path = create_reasoning_bank_backup_bundle(
                target_path=target,
                source_paths=[source],
                backup_dir=root / "backups",
                label="rebuild",
            )

            self.assertIsNotNone(archive_path)
            assert archive_path is not None
            self.assertTrue(archive_path.exists())

            with tarfile.open(archive_path, "r:gz") as tar:
                names = tar.getnames()
            self.assertIn("adversarial_reasoning_bank_v2.jsonl", names)
            self.assertIn("agent1_influence_standardized_attribution.jsonl", names)
            self.assertIn("backup_manifest.json", names)

    def test_build_script_creates_backup_before_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            attribution_input = root / "attribution.jsonl"
            malicious_input = root / "malicious.jsonl"
            enriched_input = root / "enriched.json"
            output = root / "adversarial_reasoning_bank_v2.jsonl"
            backup_dir = root / "backups"

            attribution_input.write_text(
                json.dumps(
                    {
                        "sample_id": "run::question::r1",
                        "dataset_name": "medmcqa",
                        "run_type": "adversarial",
                        "target_agent_id": "agent_2",
                        "question_text": "Question?",
                        "options": [{"option_id": "A", "text": "Alpha"}],
                        "target_previous_selected_option_ids": ["B"],
                        "target_previous_reasoning": "old reasoning",
                        "primary_factor_id": "factor_01",
                        "secondary_factor_ids": ["factor_02"],
                        "supporting_evidence": ["span"],
                        "review_to_new_reasoning_overlap": ["alpha"],
                        "review_targets_previous_error_evidence": ["old error"],
                        "review_proposes_replacement_evidence": ["choose A"],
                        "change_summary_alignment_evidence": ["changed because A"],
                        "factor_mapping_confidence": 0.9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            malicious_input.write_text(
                json.dumps(
                    {
                        "sample_id": "run::question::r1",
                        "input": {"desired_target_shift": {"to": ["A"]}},
                        "neutralized_review_rationale": "neutral",
                        "malicious_adapted_review_rationale": "malicious",
                        "malicious_strategy_notes": "notes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            enriched_input.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "sample_id": "run::question::r1",
                                "review_to_new_reasoning_overlap": ["alpha"],
                                "review_targets_previous_error_evidence": ["old error"],
                                "review_proposes_replacement_evidence": ["choose A"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output.write_text("old-bank\n", encoding="utf-8")

            argv = [
                "scripts/reasoning_bank_pipeline/build_adversarial_reasoning_bank_v2.py",
                "--attribution-input",
                str(attribution_input),
                "--malicious-input",
                str(malicious_input),
                "--enriched-input",
                str(enriched_input),
                "--output",
                str(output),
                "--backup-dir",
                str(backup_dir),
            ]

            with patch("sys.argv", argv):
                build_script.main()

            backup_files = list(backup_dir.glob("*.tar.gz"))
            self.assertEqual(len(backup_files), 1)
            self.assertIn("malicious", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
