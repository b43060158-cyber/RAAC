from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autogen_mas.dataset_adapters import HumanEvalAdapter


class HumanEvalAdapterTest(unittest.TestCase):
    def test_loads_jsonl_without_canonical_solution(self) -> None:
        sample = {
            "task_id": "HumanEval/0",
            "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n",
            "entry_point": "add_one",
            "canonical_solution": "    return x + 1\n",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "HumanEval.jsonl"
            path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            records = HumanEvalAdapter(path).load()

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.question_id, "0")
        self.assertEqual(record.dataset_name, "humaneval")
        self.assertEqual(record.task_type, "code_generation")
        self.assertEqual(record.entry_point, "add_one")
        self.assertEqual(record.source_task_id, "HumanEval/0")
        serialized = record.to_dict()
        self.assertNotIn("canonical_solution", serialized)
        self.assertNotIn("options", serialized)
        self.assertNotIn("correct_option_ids", serialized)


if __name__ == "__main__":
    unittest.main()
