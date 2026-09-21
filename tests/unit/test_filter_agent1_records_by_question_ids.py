from __future__ import annotations

import unittest

from scripts.filter_agent1_records_by_question_ids import filter_payload


class FilterAgent1RecordsByQuestionIdsTest(unittest.TestCase):
    def test_filters_json_records_payload_and_updates_count(self) -> None:
        payload = {
            "dataset_purpose": "demo",
            "record_count": 3,
            "records": [
                {"question_id": "0551", "sample_id": "s1"},
                {"question_key": "medmcqa__0868__single_choice", "sample_id": "s2"},
                {"sample_id": "run::medmcqa__1328__single_choice::agent_2::1->2::1"},
            ],
        }

        filtered = filter_payload(payload, {"0551", "1328"})

        self.assertEqual(filtered["record_count"], 2)
        self.assertEqual(
            [record["sample_id"] for record in filtered["records"]],
            ["s1", "run::medmcqa__1328__single_choice::agent_2::1->2::1"],
        )

    def test_filters_jsonl_style_list_payload(self) -> None:
        payload = [
            {"source_question_key": "medmcqa__1577__single_choice", "sample_id": "s1"},
            {"question_id": "1885", "sample_id": "s2"},
            {"question_key": "medmcqa__3799__single_choice", "sample_id": "s3"},
        ]

        filtered = filter_payload(payload, {"1885", "3799"})

        self.assertEqual([record["sample_id"] for record in filtered], ["s2", "s3"])

    def test_filters_by_run_id_when_provided(self) -> None:
        payload = {
            "record_count": 3,
            "records": [
                {
                    "run_id": "adversarial-competition-20260519-123820",
                    "question_id": "0551",
                    "sample_id": "s1",
                },
                {
                    "run_id": "competition-20260519-170057",
                    "question_id": "0551",
                    "sample_id": "s2",
                },
                {
                    "sample_id": "adversarial-competition-20260519-123820::medmcqa__1885__single_choice::agent_2::1->2::1",
                },
            ],
        }

        filtered = filter_payload(
            payload,
            {"0551", "1885"},
            {"adversarial-competition-20260519-123820"},
        )

        self.assertEqual(filtered["record_count"], 2)
        self.assertEqual([record["sample_id"] for record in filtered["records"]], ["s1", "adversarial-competition-20260519-123820::medmcqa__1885__single_choice::agent_2::1->2::1"])


if __name__ == "__main__":
    unittest.main()
