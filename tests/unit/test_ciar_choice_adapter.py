import json
import tempfile
import unittest
from pathlib import Path

from autogen_mas.dataset_adapters import CIARChoiceAdapter


_SAMPLES = [
    {
        "question": "When Alice walks up the hill, her speed is 1 m/s and down 3 m/s. Average speed?",
        "answer": ["1.5", "3/2"],
        "explanation": "harmonic mean",
        "incorrect answer": ["2"],
        "incorrect explanation": "arithmetic mean",
    },
    {
        "question": "Distribute balls to maximize chance of picking white.",
        "answer": ["0.75", "75%", "3/4"],
        "incorrect answer": ["0.5", "50%", "1/2"],
    },
]


def _write_samples(samples: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(samples, tmp)
    tmp.close()
    return Path(tmp.name)


class CIARChoiceAdapterTest(unittest.TestCase):
    def test_produces_binary_single_choice_records(self) -> None:
        path = _write_samples(_SAMPLES)
        records = CIARChoiceAdapter(path).load()

        self.assertEqual(len(records), 2)
        for index, record in enumerate(records):
            self.assertEqual(record.dataset_name, "ciar_choice")
            self.assertEqual(record.task_type, "single_choice")
            self.assertEqual(record.question_id, f"{index:04d}")
            # Exactly two options, ids A and B.
            self.assertEqual([o.option_id for o in record.options], ["A", "B"])
            # Exactly one correct option.
            self.assertEqual(len(record.correct_option_ids), 1)
            self.assertIn(record.correct_option_ids[0], {"A", "B"})
            self.assertEqual(
                record.question_key, f"ciar_choice__{index:04d}__single_choice"
            )

    def test_option_text_joins_full_alias_cluster(self) -> None:
        path = _write_samples(_SAMPLES)
        records = CIARChoiceAdapter(path).load()

        record = records[1]
        texts = {o.text for o in record.options}
        self.assertIn("0.75 / 75% / 3/4", texts)
        self.assertIn("0.5 / 50% / 1/2", texts)
        # The correct option text must be the answer cluster.
        correct = next(
            o for o in record.options if o.option_id == record.correct_option_ids[0]
        )
        self.assertEqual(correct.text, "0.75 / 75% / 3/4")

    def test_metadata_retains_alias_clusters(self) -> None:
        path = _write_samples(_SAMPLES)
        records = CIARChoiceAdapter(path).load()

        meta = records[0].metadata
        self.assertEqual(meta["correct_answer_aliases"], ["1.5", "3/2"])
        self.assertEqual(meta["incorrect_answer_aliases"], ["2"])
        self.assertEqual(meta["explanation"], "harmonic mean")
        self.assertEqual(meta["incorrect_explanation"], "arithmetic mean")

    def test_correct_position_is_deterministic(self) -> None:
        path = _write_samples(_SAMPLES)
        first = CIARChoiceAdapter(path).load()
        second = CIARChoiceAdapter(path).load()

        self.assertEqual(
            [r.correct_option_ids for r in first],
            [r.correct_option_ids for r in second],
        )


if __name__ == "__main__":
    unittest.main()
