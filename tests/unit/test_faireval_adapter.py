from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import FairEvalAdapter


class FairEvalAdapterTest(unittest.TestCase):
    def test_rejects_unknown_answer_key_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "FairEval.json").write_text(
                '[{"question_id": 1, "question": "Q?", "category": "generic", '
                '"response": {"gpt35": "R1", "vicuna": "R2"}}]',
                encoding="utf-8",
            )
            (temp_path / "review_gpt35_vicuna-13b_human.txt").write_text(
                "UNKNOWN\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported label"):
                FairEvalAdapter(temp_path / "FairEval.json").load()

    def test_rejects_mismatched_answer_key_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "FairEval.json").write_text(
                '[{"question_id": 1, "question": "Q1?", "category": "generic", '
                '"response": {"gpt35": "R1", "vicuna": "R2"}}, '
                '{"question_id": 2, "question": "Q2?", "category": "generic", '
                '"response": {"gpt35": "R3", "vicuna": "R4"}}]',
                encoding="utf-8",
            )
            (temp_path / "review_gpt35_vicuna-13b_human.txt").write_text(
                "CHATGPT\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line count does not match"):
                FairEvalAdapter(temp_path / "FairEval.json").load()


if __name__ == "__main__":
    unittest.main()
