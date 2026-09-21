from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.reasoning_bank_pipeline import (
    merge_incremental_reasoning_bank_candidate as merge_script,
)


class MergeIncrementalReasoningBankCandidateTest(unittest.TestCase):
    def test_merge_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_dir = Path(tmpdir)
            (candidate_dir / "candidate_manifest.json").write_text(
                json.dumps({"status": "pending_review"}),
                encoding="utf-8",
            )
            (candidate_dir / "adversarial_reasoning_bank_v2_candidate.jsonl").write_text(
                "",
                encoding="utf-8",
            )
            argv = [
                "scripts/reasoning_bank_pipeline/merge_incremental_reasoning_bank_candidate.py",
                "--candidate-dir",
                str(candidate_dir),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(ValueError):
                    merge_script.main()


if __name__ == "__main__":
    unittest.main()
