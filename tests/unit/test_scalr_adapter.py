from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import SCALRAdapter


class SCALRAdapterTest(unittest.TestCase):
    def test_loads_jsonl_with_variable_choice_count(self) -> None:
        adapter = SCALRAdapter(Path("data/scalr/test.jsonl"))

        records = adapter.load()

        self.assertEqual(len(records), 571)
        self.assertEqual(records[0].question_id, "0000")
        self.assertEqual(records[0].dataset_name, "scalr")
        self.assertEqual(records[0].task_type, "single_choice")
        self.assertEqual(records[0].correct_option_ids, ["D"])
        self.assertEqual(
            [option.option_id for option in records[0].options],
            ["A", "B", "C", "D", "E"],
        )
        self.assertEqual(records[0].metadata["answer_index"], 3)
        self.assertEqual(records[0].metadata["source_id"], "0")

    def test_rejects_non_contiguous_choice_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scalr.jsonl"
            path.write_text(
                (
                    '{"index":"1","question":"Q","answer":"1",'
                    '"choice_0":"A","choice_2":"C"}\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "contiguous choice_\\* fields"):
                SCALRAdapter(path).load()

    def test_rejects_out_of_range_answer_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scalr.jsonl"
            path.write_text(
                (
                    '{"index":"1","question":"Q","answer":"3",'
                    '"choice_0":"A","choice_1":"B"}\n'
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "answer index 3"):
                SCALRAdapter(path).load()


if __name__ == "__main__":
    unittest.main()
