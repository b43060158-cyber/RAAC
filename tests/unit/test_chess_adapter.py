from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import ChessAdapter


class ChessAdapterTest(unittest.TestCase):
    def test_loads_chess_as_short_answer_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chess.json"
            path.write_text(
                """
{
  "Chess_0": {
    "input": "a2a4 a7a5 d2",
    "target": ["e3", "c3", "d1"]
  }
}
""",
                encoding="utf-8",
            )

            records = ChessAdapter(path).load()

        self.assertEqual(len(records), 1)
        first = records[0]
        self.assertEqual(first.question_id, "Chess_0")
        self.assertEqual(first.dataset_name, "chess")
        self.assertEqual(first.task_type, "chess_move")
        self.assertEqual(first.legal_target_squares, ["e3", "c3", "d1"])
        self.assertEqual(first.game, "a2a4 a7a5")
        self.assertEqual(first.source_square, "d2")
        self.assertEqual(first.metadata["source_square"], "d2")
        self.assertEqual(first.metadata["answer_format"], "chess_square")
        self.assertIn('Given the chess game "a2a4 a7a5"', first.question)
        self.assertIn('piece at "d2"', first.question)
        self.assertEqual(first.rendered_question, first.question)

    def test_rejects_malformed_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chess.json"
            path.write_text(
                """
{
  "Chess_0": {
    "input": "a2a4 a7a5",
    "target": ["e3"]
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source square"):
                ChessAdapter(path).load()


if __name__ == "__main__":
    unittest.main()
