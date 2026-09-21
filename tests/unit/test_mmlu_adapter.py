from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import MMLUAdapter


class MMLUAdapterTest(unittest.TestCase):
    def test_loads_mmlu_csv_without_header(self) -> None:
        adapter = MMLUAdapter(Path("data/mmlu/test"))

        records = adapter.load()

        self.assertEqual(len(records), 1602)
        self.assertEqual(records[0].question_id, "0000")
        self.assertEqual(records[0].dataset_name, "mmlu")
        self.assertEqual(records[0].task_type, "single_choice")
        self.assertEqual([option.option_id for option in records[0].options], ["A", "B", "C", "D"])
        self.assertEqual(records[0].correct_option_ids, ["A"])
        self.assertIn("type-Ia", records[0].question)

    def test_skips_header_and_normalizes_lowercase_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmlu.csv"
            path.write_text(
                "question,A,B,C,D,answer\n"
                "\"Q?\",opt1,opt2,opt3,opt4,c\n",
                encoding="utf-8",
            )

            records = MMLUAdapter(path).load()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].correct_option_ids, ["C"])

    def test_aggregates_all_csv_files_under_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "astronomy_test.csv").write_text(
                "\"A1?\",a,b,c,d,A\n\"A2?\",a,b,c,d,B\n",
                encoding="utf-8",
            )
            (root / "ethics_test.csv").write_text(
                "\"E1?\",a,b,c,d,C\n",
                encoding="utf-8",
            )

            records = MMLUAdapter(root).load()

        # All questions across files are combined into one pool.
        self.assertEqual(len(records), 3)
        # question_id values are globally unique and contiguous.
        self.assertEqual([r.question_id for r in records], ["0000", "0001", "0002"])
        # Subject is recorded from the originating file.
        self.assertEqual(records[0].metadata["subject"], "astronomy_test")
        self.assertEqual(records[2].metadata["subject"], "ethics_test")

    def test_rejects_invalid_answer_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmlu.csv"
            path.write_text(
                "\"Q?\",opt1,opt2,opt3,opt4,E\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid answer"):
                MMLUAdapter(path).load()


if __name__ == "__main__":
    unittest.main()
