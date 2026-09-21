import unittest

from scripts.run_adversarial_question_list import (
    parse_question_indices,
    select_questions_by_index,
)


class RunAdversarialQuestionListTests(unittest.TestCase):
    def test_parse_question_indices_supports_spaces_and_commas(self) -> None:
        self.assertEqual(
            parse_question_indices(["1046,1244", "1992 2968"]),
            [1046, 1244, 1992, 2968],
        )

    def test_parse_question_indices_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate question indices"):
            parse_question_indices(["1046", "1046"])

    def test_select_questions_by_index_preserves_requested_order(self) -> None:
        questions = ["q0", "q1", "q2", "q3"]
        self.assertEqual(
            select_questions_by_index(questions, [3, 1]),
            ["q3", "q1"],
        )

    def test_select_questions_by_index_rejects_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_questions_by_index(["q0"], [1])


if __name__ == "__main__":
    unittest.main()
