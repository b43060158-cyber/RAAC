from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import MedMCQAAdapter


class MedMCQAAdapterTest(unittest.TestCase):
    def test_loads_jsonl_with_cop_as_one_based_correct_option(self) -> None:
        adapter = MedMCQAAdapter(Path("data/medmcqa/dev.json"))

        records = adapter.load()

        self.assertEqual(len(records), 4183)
        self.assertEqual(records[0].question_id, "0000")
        self.assertEqual(records[0].dataset_name, "medmcqa")
        self.assertEqual(records[0].task_type, "single_choice")
        self.assertEqual([option.option_id for option in records[0].options], ["A", "B", "C", "D"])

        self.assertEqual(records[0].correct_option_ids, ["A"])
        self.assertEqual(
            records[0].options[0].text,
            "Impulse through myelinated fibers is slower than non-myelinated fibers",
        )
        self.assertEqual(records[0].metadata["cop"], 1)

        self.assertEqual(records[1].correct_option_ids, ["A"])
        self.assertEqual(records[1].metadata["cop"], 1)

        self.assertEqual(records[2].correct_option_ids, ["C"])
        self.assertEqual(records[2].metadata["cop"], 3)
        self.assertEqual(
            records[2].options[2].text,
            "Amniotic fluid samples plus chromosomal analysis will definitely tell her that next baby will be down syndromic or not",
        )

    def test_missing_cop_reports_unlabeled_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.json"
            path.write_text(
                '{"question":"Q","opa":"A","opb":"B","opc":"C","opd":"D"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "answer-hidden test split"):
                MedMCQAAdapter(path).load()


if __name__ == "__main__":
    unittest.main()
