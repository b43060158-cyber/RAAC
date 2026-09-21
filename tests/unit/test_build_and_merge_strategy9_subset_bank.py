from __future__ import annotations

import unittest

from scripts.reasoning_bank_pipeline.build_and_merge_strategy9_subset_bank import (
    merge_bank_records,
)


class BuildAndMergeStrategy9SubsetBankTest(unittest.TestCase):
    def test_merge_appends_new_sample_ids(self) -> None:
        existing = [
            {"sample_id": "s1", "value": "old-1"},
            {"sample_id": "s2", "value": "old-2"},
        ]
        incoming = [
            {"sample_id": "s3", "value": "new-3"},
        ]

        merged = merge_bank_records(existing, incoming)

        self.assertEqual(
            [record["sample_id"] for record in merged],
            ["s1", "s2", "s3"],
        )

    def test_merge_renames_incoming_record_on_sample_id_collision(self) -> None:
        existing = [
            {"sample_id": "s1", "value": "old-1"},
            {"sample_id": "s2", "value": "old-2"},
        ]
        incoming = [
            {"sample_id": "s2", "value": "new-2"},
            {"sample_id": "s3", "value": "new-3"},
        ]

        merged = merge_bank_records(existing, incoming, merge_tag="20260521-180000")

        self.assertEqual(
            merged,
            [
                {"sample_id": "s1", "value": "old-1"},
                {"sample_id": "s2", "value": "old-2"},
                {
                    "sample_id": "s2::merged_20260521-180000",
                    "original_sample_id": "s2",
                    "value": "new-2",
                },
                {"sample_id": "s3", "value": "new-3"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
