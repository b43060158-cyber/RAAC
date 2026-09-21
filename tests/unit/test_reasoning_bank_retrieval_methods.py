"""Tests for hot-pluggable retrieval + model-family routing of the bank."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autogen_mas.config import AdversarialAgentBehaviorConfig
from autogen_mas.models import OptionRecord, QuestionRecord
from autogen_mas.runtime.adversarial.reasoning_bank import ReasoningBank, model_family
from autogen_mas.runtime.adversarial.retrieval import (
    RetrievalContext,
    RetrievalError,
    ScoredRecord,
    get_scorer,
    get_selector,
    register_scorer,
    register_selector,
)


# Register test-only methods once at import time (ids are unique).
@register_scorer("test_reverse_pin")
def _test_reverse_pin(records, ctx):
    # Score purely by reverse sample_id so the *last* record wins, proving the
    # configured scorer (not the default lexical one) is in effect.
    return [
        ScoredRecord(score=float(index), record=record)
        for index, record in enumerate(records)
    ]


@register_selector("test_take_one")
def _test_take_one(scored, ctx, top_k):
    ordered = sorted(scored, key=lambda item: -item.score)
    return [item.record for item in ordered[:1]]


def _record(sample_id, *, dataset="medmcqa", family=None, retrieval_text="alpha beta"):
    record = {
        "sample_id": sample_id,
        "corpus_id": "bank_test",
        "dataset_name": dataset,
        "compatible_modes": ["convert"],
        "target_agent_id": "agent_x",
        "retrieval_text": retrieval_text,
        "desired_target_shift": {"to": ["B"]},
    }
    if family is not None:
        record["build_model_family"] = family
    return record


def _question(dataset="medmcqa"):
    return QuestionRecord(
        question_id="q1",
        dataset_name=dataset,
        task_type="single_choice",
        question="alpha beta gamma?",
        options=[OptionRecord(option_id="A", text="A"), OptionRecord(option_id="B", text="B")],
        correct_option_ids=["A"],
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _retrieve(bank: ReasoningBank, *, dataset="medmcqa", strategy_name="strategy9"):
    return bank.retrieve_examples(
        question_record=_question(dataset),
        target_agent_id="agent_2",
        target_current_option_ids=["A"],
        target_current_reasoning="alpha beta reasoning",
        desired_target_shift=["B"],
        review_phase_mode="convert",
        strategy_name=strategy_name,
    )


class ModelFamilyTests(unittest.TestCase):
    def test_buckets(self) -> None:
        self.assertEqual(model_family("qwen3.5-flash"), "qwen")
        self.assertEqual(model_family("qwen-plus"), "qwen")
        self.assertEqual(model_family("deepseek-chat"), "deepseek")
        self.assertEqual(model_family(""), "")
        self.assertEqual(model_family("mystery-model"), "other")


class HotSwapTests(unittest.TestCase):
    def test_per_strategy_scorer_selector_override_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            _write_jsonl(path, [_record("s1"), _record("s2"), _record("s3")])
            config = AdversarialAgentBehaviorConfig(
                attack_strategy={"choice": "strategy9", "short_answer": "0", "code": "0"},
                reasoning_bank_path=str(path),
                reasoning_bank_scorer_by_strategy={"strategy9": "test_reverse_pin"},
                reasoning_bank_selector_by_strategy={"strategy9": "test_take_one"},
            ).validate()
            bank = ReasoningBank(behavior_config=config)
            results = _retrieve(bank)
            # test_reverse_pin scores by position, test_take_one keeps 1 -> last record.
            self.assertEqual([r["sample_id"] for r in results], ["s3"])

    def test_resolution_precedence(self) -> None:
        config = AdversarialAgentBehaviorConfig(
            attack_strategy={"choice": "strategy9", "short_answer": "0", "code": "0"},
            reasoning_bank_scorer="global_scorer",
            reasoning_bank_scorer_by_strategy={"strategy9": "strat_scorer"},
            reasoning_bank_retrieval_params={"a": 1},
            reasoning_bank_retrieval_params_by_strategy={"strategy9": {"b": 2}},
        ).validate()
        scorer, selector, params = config.resolve_retrieval_method(strategy_name="strategy9")
        self.assertEqual(scorer, "strat_scorer")
        self.assertIsNone(selector)
        self.assertEqual(params, {"a": 1, "b": 2})
        # Unknown strategy falls back to global default.
        scorer2, _, _ = config.resolve_retrieval_method(strategy_name="other")
        self.assertEqual(scorer2, "global_scorer")


class FamilyRoutingTests(unittest.TestCase):
    def _config(self, tmp, *, fallback=True, cross_dataset=False, families=None):
        return AdversarialAgentBehaviorConfig(
            attack_strategy={"choice": "strategy9", "short_answer": "0", "code": "0"},
            reasoning_bank_paths_by_model_family=families,
            reasoning_bank_allow_cross_family_fallback=fallback,
            reasoning_bank_allow_cross_dataset=cross_dataset,
        ).validate()

    def test_routes_to_matching_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qwen = Path(tmp) / "qwen.jsonl"
            deepseek = Path(tmp) / "deepseek.jsonl"
            _write_jsonl(qwen, [_record("qwen1")])
            _write_jsonl(deepseek, [_record("ds1")])
            config = self._config(
                tmp,
                families={"qwen": [str(qwen)], "deepseek": [str(deepseek)]},
            )
            qwen_bank = ReasoningBank(behavior_config=config, runtime_model="qwen3.5-flash")
            ds_bank = ReasoningBank(behavior_config=config, runtime_model="deepseek-chat")
            self.assertEqual([r["sample_id"] for r in _retrieve(qwen_bank)], ["qwen1"])
            self.assertEqual([r["sample_id"] for r in _retrieve(ds_bank)], ["ds1"])

    def test_cross_family_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            deepseek = Path(tmp) / "deepseek.jsonl"
            _write_jsonl(deepseek, [_record("ds1")])
            # qwen runtime, but only a deepseek bank exists.
            families = {"deepseek": [str(deepseek)]}
            fb_on = self._config(tmp, fallback=True, families=families)
            bank_on = ReasoningBank(behavior_config=fb_on, runtime_model="qwen-plus")
            self.assertEqual([r["sample_id"] for r in _retrieve(bank_on)], ["ds1"])

            fb_off = self._config(tmp, fallback=False, families=families)
            bank_off = ReasoningBank(behavior_config=fb_off, runtime_model="qwen-plus")
            with self.assertRaises(RetrievalError):
                _retrieve(bank_off)

    def test_never_empty_under_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qwen = Path(tmp) / "qwen.jsonl"
            # qwen bank only has a different-dataset record; cross-dataset off.
            _write_jsonl(qwen, [_record("qwen_other", dataset="scalr")])
            config = self._config(
                tmp, fallback=False, cross_dataset=False, families={"qwen": [str(qwen)]}
            )
            bank = ReasoningBank(behavior_config=config, runtime_model="qwen3.5-flash")
            with self.assertRaises(RetrievalError):
                _retrieve(bank, dataset="medmcqa")

    def test_cross_dataset_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            qwen = Path(tmp) / "qwen.jsonl"
            _write_jsonl(qwen, [_record("qwen_other", dataset="scalr")])
            config = self._config(
                tmp, fallback=False, cross_dataset=True, families={"qwen": [str(qwen)]}
            )
            bank = ReasoningBank(behavior_config=config, runtime_model="qwen3.5-flash")
            results = _retrieve(bank, dataset="medmcqa")
            self.assertEqual([r["sample_id"] for r in results], ["qwen_other"])


def _ctx(*, mode="convert", desired_shift=None, params=None):
    return RetrievalContext(
        question_record=_question(),
        dataset_name="medmcqa",
        target_agent_id="agent_2",
        target_current_option_ids=["A"],
        target_current_reasoning="alpha beta reasoning",
        target_current_final_answer="",
        desired_target_shift=list(desired_shift or ["B"]),
        desired_target_final_answer="",
        review_phase_mode=mode,
        params=params or {},
    )


class SituationalScorerTests(unittest.TestCase):
    def test_situational_beats_topical(self) -> None:
        scorer = get_scorer("situational_transfer_v1")
        situational = {
            "sample_id": "situational",
            "corpus_id": "b",
            "desired_target_shift": {"to": ["B"]},
            "target_previous_selected_option_ids": ["A"],
            "borrowable_style": "replacement_path",
            "retrieval_text": "totally unrelated words",
        }
        topical = {
            "sample_id": "topical",
            "corpus_id": "b",
            "desired_target_shift": {"to": ["C"]},
            "borrowable_style": "",
            # Heavy overlap with the question text but wrong situation.
            "retrieval_text": "alpha beta gamma",
        }
        scored = scorer([topical, situational], _ctx())
        best = max(scored, key=lambda s: s.score)
        self.assertEqual(best.record["sample_id"], "situational")


class ThresholdSelectorTests(unittest.TestCase):
    def test_floor_filters_but_never_empties(self) -> None:
        selector = get_selector("threshold_topk_v1")
        scored = [
            ScoredRecord(0.1, {"sample_id": "low", "corpus_id": "b"}),
            ScoredRecord(0.9, {"sample_id": "high", "corpus_id": "b"}),
        ]
        kept = selector(scored, _ctx(params={"min_score": 0.5}), 4)
        self.assertEqual([r["sample_id"] for r in kept], ["high"])
        # Nothing clears the floor -> still returns the single best.
        none_pass = selector(scored, _ctx(params={"min_score": 5.0}), 4)
        self.assertEqual([r["sample_id"] for r in none_pass], ["high"])


class MmrSelectorTests(unittest.TestCase):
    def test_diversifies_styles(self) -> None:
        selector = get_selector("mmr_diversity_v1")
        scored = [
            ScoredRecord(1.0, {"sample_id": "r1", "corpus_id": "b", "borrowable_style": "replacement_path"}),
            ScoredRecord(0.9, {"sample_id": "r2", "corpus_id": "b", "borrowable_style": "replacement_path"}),
            ScoredRecord(0.8, {"sample_id": "r3", "corpus_id": "b", "borrowable_style": "alignment_transfer"}),
        ]
        chosen = selector(scored, _ctx(params={"mmr_lambda": 0.5}), 2)
        # Top relevance is r1; MMR should prefer the distinct-style r3 over the
        # near-duplicate r2 for the second slot.
        self.assertEqual([r["sample_id"] for r in chosen], ["r1", "r3"])


class CliOverrideTests(unittest.TestCase):
    def test_cli_overrides_scorer_selector_params_and_routing(self) -> None:
        from autogen_mas.cli import _build_parser, _with_adversarial_overrides
        from autogen_mas.config import load_experiment_config

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run-adversarial-dataset",
                "--dataset",
                "scalr",
                "--reasoning-bank-scorer",
                "situational_transfer_v1",
                "--reasoning-bank-selector",
                "mmr_diversity_v1",
                "--reasoning-bank-retrieval-param",
                "mmr_lambda=0.7",
                "--reasoning-bank-retrieval-param",
                "min_score=0.4",
                "--reasoning-bank-retrieval-param",
                "w_question_overlap=0.1",
                "--reasoning-bank-allow-cross-dataset",
                "--no-reasoning-bank-cross-family-fallback",
            ]
        )
        config = load_experiment_config("config/paper.yaml")
        updated = _with_adversarial_overrides(config, args)
        behavior = updated.adversarial_mas.adversarial_agent
        self.assertEqual(behavior.reasoning_bank_scorer, "situational_transfer_v1")
        self.assertEqual(behavior.reasoning_bank_selector, "mmr_diversity_v1")
        self.assertEqual(
            behavior.reasoning_bank_retrieval_params,
            {"mmr_lambda": 0.7, "min_score": 0.4, "w_question_overlap": 0.1},
        )
        self.assertTrue(behavior.reasoning_bank_allow_cross_dataset)
        self.assertFalse(behavior.reasoning_bank_allow_cross_family_fallback)


if __name__ == "__main__":
    unittest.main()
