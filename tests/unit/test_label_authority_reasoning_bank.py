from __future__ import annotations

import unittest

from scripts.label_authority_reasoning_bank import (
    LABEL_VERSION,
    is_labeled,
    labeled_record,
    normalize_authority_response,
)


class LabelAuthorityReasoningBankTests(unittest.TestCase):
    def test_normalize_authority_response_accepts_flagged_examples(self) -> None:
        normalized = normalize_authority_response(
            {
                "authority_packaging_flag": True,
                "authority_packaging_risk": "HEAVY",
                "authority_packaging_reason": "Leans on a named institute and theatrical statistic.",
            }
        )

        self.assertEqual(
            normalized,
            {
                "authority_packaging_flag": True,
                "authority_packaging_risk": "heavy",
                "authority_packaging_reason": "Leans on a named institute and theatrical statistic.",
            },
        )

    def test_normalize_authority_response_clears_risk_for_non_authority_examples(self) -> None:
        normalized = normalize_authority_response(
            {
                "authority_packaging_flag": False,
                "authority_packaging_risk": "medium",
                "authority_packaging_reason": "This is a direct correction without decorative authority cues.",
            }
        )

        self.assertEqual(
            normalized,
            {
                "authority_packaging_flag": False,
                "authority_packaging_risk": "",
                "authority_packaging_reason": "This is a direct correction without decorative authority cues.",
            },
        )

    def test_normalize_authority_response_requires_risk_for_flagged_examples(self) -> None:
        with self.assertRaises(ValueError):
            normalize_authority_response(
                {
                    "authority_packaging_flag": True,
                    "authority_packaging_reason": "Missing risk should fail.",
                }
            )

    def test_is_labeled_rejects_incomplete_flagged_records(self) -> None:
        self.assertFalse(
            is_labeled(
                {
                    "authority_packaging_flag": True,
                    "authority_packaging_reason": "Reason without risk is incomplete.",
                }
            )
        )

    def test_labeled_record_writes_model_and_version(self) -> None:
        output = labeled_record(
            {
                "sample_id": "sample-1",
                "authority_packaging_flag": False,
            },
            {
                "authority_packaging_flag": True,
                "authority_packaging_risk": "medium",
                "authority_packaging_reason": "Uses an obscure expert name to prop up the review.",
            },
            model="deepseek-chat",
        )

        self.assertEqual(output["sample_id"], "sample-1")
        self.assertTrue(output["authority_packaging_flag"])
        self.assertEqual(output["authority_packaging_risk"], "medium")
        self.assertEqual(output["authority_label_model"], "deepseek-chat")
        self.assertEqual(output["authority_label_version"], LABEL_VERSION)


if __name__ == "__main__":
    unittest.main()
