from pathlib import Path
import unittest

from autogen_mas.dataset_adapters import TruthfulQAAdapter


class TruthfulQAAdapterTest(unittest.TestCase):
    def test_load_defaults_to_mc1_single_choice(self) -> None:
        adapter = TruthfulQAAdapter(Path("data/truthfulqa/test.json"))

        records = adapter.load()

        self.assertEqual(len(records), 817)
        mc1_record = records[0]

        self.assertEqual(mc1_record.question_id, "0000")
        self.assertEqual(mc1_record.task_type, "single_choice")
        self.assertEqual(mc1_record.correct_option_ids, ["A"])
        self.assertEqual([option.option_id for option in mc1_record.options], ["A", "B", "C", "D"])
        self.assertEqual(mc1_record.metadata["source_task_key"], "mc1_targets")

    def test_load_rejects_multiple_choice(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports single_choice"):
            TruthfulQAAdapter(
                Path("data/truthfulqa/test.json"),
                task_types=("single_choice", "multiple_choice"),
            )


if __name__ == "__main__":
    unittest.main()
