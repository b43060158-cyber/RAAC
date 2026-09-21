from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from autogen_mas.cli import (
    _batch_runtime_options,
    _build_batch_progress,
    _filter_questions_by_failed_questions_file,
    _select_questions,
    main,
)
from autogen_mas.config import (
    AgentConfig,
    AlignmentJudgeConfig,
    DashScopeSettings,
    ExperimentConfig,
    RuntimeConfig,
    SingleAgentConfig,
)
from autogen_mas.models import OptionRecord, QuestionRecord


class FakeSingleAgentRunner:
    init_config = None
    init_settings = None
    received_questions = None
    received_progress_callback = None
    received_resume_run_path = None

    def __init__(self, experiment_config, llm_settings):
        self.__class__.init_config = experiment_config
        self.__class__.init_settings = llm_settings

    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/single-agent-run")


class FakeCompetitionRunner:
    init_config = None
    init_settings = None
    init_debug_mode = None
    init_persist_feedback_summary = None
    received_questions = None
    received_progress_callback = None
    received_resume_run_path = None

    def __init__(
        self,
        experiment_config,
        llm_settings,
        debug_mode=False,
        persist_feedback_summary=False,
    ):
        self.__class__.init_config = experiment_config
        self.__class__.init_settings = llm_settings
        self.__class__.init_debug_mode = debug_mode
        self.__class__.init_persist_feedback_summary = persist_feedback_summary

    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/mas-run")


class FakeChessCompetitionRunner(FakeCompetitionRunner):
    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/chess-mas-run")


class FakeBatchCompetitionRunner:
    init_config = None
    init_settings = None
    init_batch_options = None
    received_questions = None
    received_progress_callback = None
    received_resume_run_path = None

    def __init__(self, experiment_config, llm_settings, batch_options):
        self.__class__.init_config = experiment_config
        self.__class__.init_settings = llm_settings
        self.__class__.init_batch_options = batch_options

    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/batched-mas-run")


class FakeAdversarialCompetitionRunner(FakeCompetitionRunner):
    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/adversarial-mas-run")


class FakeBatchSingleAgentRunner(FakeBatchCompetitionRunner):
    def run_dataset(self, questions, progress_callback=None, resume_run_path=None):
        self.__class__.received_questions = questions
        self.__class__.received_progress_callback = progress_callback
        self.__class__.received_resume_run_path = resume_run_path
        return SimpleNamespace(run_path="/tmp/batched-single-agent-run")


class FakeEvaluator:
    call = None
    evaluate_call = None
    evaluate_round_sweep_call = None

    def evaluate(self, run_path, *, selection_rule="top-agent"):
        self.__class__.evaluate_call = {
            "run_path": run_path,
            "selection_rule": selection_rule,
        }
        return SimpleNamespace(
            summary={
                "total_questions": 4,
                "correct_questions": 3,
                "accuracy": 0.75,
                "by_task_type": {},
            }
        )

    def evaluate_single_agent(self, run_path, *, agent_id, model, temperature):
        self.__class__.call = {
            "run_path": run_path,
            "agent_id": agent_id,
            "model": model,
            "temperature": temperature,
        }
        return {
            "run_path": run_path,
            "agent_id": agent_id,
            "model": model,
            "temperature": temperature,
            "selection_rule": "single_agent_direct_answer",
            "total_questions": 2,
            "correct_questions": 1,
            "accuracy": 0.5,
            "by_task_type": {},
        }

    def evaluate_round_sweep(
        self,
        run_path,
        *,
        selection_rule="top-agent",
        max_round=None,
    ):
        self.__class__.evaluate_round_sweep_call = {
            "run_path": run_path,
            "selection_rule": selection_rule,
            "max_round": max_round,
        }
        return {
            "run_path": run_path,
            "experiment": "round_sweep",
            "max_round": max_round,
            "rounds": [],
        }


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeSingleAgentRunner.init_config = None
        FakeSingleAgentRunner.init_settings = None
        FakeSingleAgentRunner.received_questions = None
        FakeSingleAgentRunner.received_progress_callback = None
        FakeSingleAgentRunner.received_resume_run_path = None
        FakeCompetitionRunner.init_config = None
        FakeCompetitionRunner.init_settings = None
        FakeCompetitionRunner.init_debug_mode = None
        FakeCompetitionRunner.init_persist_feedback_summary = None
        FakeCompetitionRunner.received_questions = None
        FakeCompetitionRunner.received_progress_callback = None
        FakeCompetitionRunner.received_resume_run_path = None
        FakeChessCompetitionRunner.init_config = None
        FakeChessCompetitionRunner.init_settings = None
        FakeChessCompetitionRunner.init_debug_mode = None
        FakeChessCompetitionRunner.init_persist_feedback_summary = None
        FakeChessCompetitionRunner.received_questions = None
        FakeChessCompetitionRunner.received_progress_callback = None
        FakeChessCompetitionRunner.received_resume_run_path = None
        FakeBatchCompetitionRunner.init_config = None
        FakeBatchCompetitionRunner.init_settings = None
        FakeBatchCompetitionRunner.init_batch_options = None
        FakeBatchCompetitionRunner.received_questions = None
        FakeBatchCompetitionRunner.received_progress_callback = None
        FakeBatchCompetitionRunner.received_resume_run_path = None
        FakeAdversarialCompetitionRunner.init_config = None
        FakeAdversarialCompetitionRunner.init_settings = None
        FakeAdversarialCompetitionRunner.init_debug_mode = None
        FakeAdversarialCompetitionRunner.init_persist_feedback_summary = None
        FakeAdversarialCompetitionRunner.received_questions = None
        FakeAdversarialCompetitionRunner.received_progress_callback = None
        FakeAdversarialCompetitionRunner.received_resume_run_path = None
        FakeBatchSingleAgentRunner.init_config = None
        FakeBatchSingleAgentRunner.init_settings = None
        FakeBatchSingleAgentRunner.init_batch_options = None
        FakeBatchSingleAgentRunner.received_questions = None
        FakeBatchSingleAgentRunner.received_progress_callback = None
        FakeBatchSingleAgentRunner.received_resume_run_path = None
        FakeEvaluator.call = None
        FakeEvaluator.evaluate_call = None
        FakeEvaluator.evaluate_round_sweep_call = None

    def _questions(self, count: int = 6) -> list[QuestionRecord]:
        return [
            QuestionRecord(
                question_id=f"q{index}",
                dataset_name="demo",
                task_type="single_choice",
                question=f"Question {index}?",
                options=[OptionRecord("A", "Alpha"), OptionRecord("B", "Beta")],
                correct_option_ids=["A"],
            )
            for index in range(count)
        ]

    def _enable_adversarial_runner_config(self, config: ExperimentConfig, *, total_agents: int = 1) -> None:
        config.adversarial_mas.total_agents = total_agents
        config.adversarial_mas.agents = [
            AgentConfig(agent_id=f"agent_{index}")
            for index in range(1, total_agents + 1)
        ]
        config.adversarial_mas.adversarial_count = min(1, total_agents)

    def test_select_questions_samples_by_seed_then_restores_original_order(self) -> None:
        questions = self._questions(10)

        selected_a, selection_a = _select_questions(questions, limit=4, seed=42)
        selected_b, selection_b = _select_questions(questions, limit=4, seed=42)
        first_n, first_n_selection = _select_questions(questions, limit=4, seed=None)

        self.assertEqual([item.question_id for item in selected_a], [item.question_id for item in selected_b])
        self.assertEqual(
            [questions.index(item) for item in selected_a],
            sorted(questions.index(item) for item in selected_a),
        )
        self.assertNotEqual([item.question_id for item in selected_a], ["q0", "q1", "q2", "q3"])
        self.assertEqual(selection_a.strategy, "random_sample_original_order")
        self.assertEqual(selection_b.seed, 42)
        self.assertEqual([item.question_id for item in first_n], ["q0", "q1", "q2", "q3"])
        self.assertEqual(first_n_selection.strategy, "first_n")

    def test_dataset_load_errors_are_reported_as_cli_errors(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        stderr = io.StringIO()

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._load_questions",
                    side_effect=ValueError("MedMCQA sample is missing the 'cop' answer field."),
                )
            )
            stack.enter_context(redirect_stderr(stderr))
            with self.assertRaises(SystemExit) as raised:
                main(["run-dataset", "--dataset", "medmcqa"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("MedMCQA sample is missing the 'cop' answer field.", stderr.getvalue())

    def test_batch_progress_resets_total_per_stage_and_shows_batch_context(self) -> None:
        created_progress = []

        class FakeTqdm:
            def __init__(self, *, total, desc, unit):
                self.total = total
                self.desc = desc
                self.unit = unit
                self.n = 0
                self.postfixes = []
                self.closed = False
                created_progress.append(self)

            def refresh(self):
                pass

            def set_postfix_str(self, value, refresh=True):
                self.postfixes.append(value)

            def update(self, amount):
                self.n += amount

            def close(self):
                self.closed = True

        fake_tqdm_module = ModuleType("tqdm")
        fake_tqdm_module.tqdm = FakeTqdm
        question = self._questions(1)[0]

        with patch.dict(sys.modules, {"tqdm": fake_tqdm_module}):
            on_progress, close_progress = _build_batch_progress("Batched LLM-MAS")

        progress = created_progress[0]
        self.assertEqual(progress.unit, "call")
        on_progress(
            1,
            100,
            question,
            "round 1/3 answer agent_batches 40 question_batches 10 planned",
        )
        self.assertEqual(progress.total, 40)
        self.assertEqual(progress.n, 0)
        self.assertEqual(
            progress.postfixes[-1],
            "round=1/3 stage=answer question_batches=10 api_calls=40",
        )

        on_progress(1, 100, question, "round 1/3 batch answer agent_1 8q started")
        self.assertEqual(progress.n, 0)
        self.assertIn("stage=answer", progress.postfixes[-1])
        self.assertIn("batch_size=8q", progress.postfixes[-1])
        self.assertIn("status=started", progress.postfixes[-1])

        on_progress(1, 100, question, "round 1/3 batch answer agent_1 8q completed")
        self.assertEqual(progress.n, 1)

        on_progress(
            1,
            100,
            question,
            "round 1/3 review agent_batches 24 question_batches 6 planned",
        )
        self.assertEqual(progress.total, 24)
        self.assertEqual(progress.n, 0)
        self.assertEqual(
            progress.postfixes[-1],
            "round=1/3 stage=review question_batches=6 api_calls=24",
        )

        on_progress(1, 100, question, "round 1/3 batch review agent_2 4q completed")
        self.assertEqual(progress.n, 1)
        self.assertIn("stage=review", progress.postfixes[-1])
        self.assertIn("batch_size=4q", progress.postfixes[-1])
        close_progress()
        self.assertTrue(progress.closed)

    def test_list_models_uses_anthropic_route_profile(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=2, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(
            api_key="deepseek-key",
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            client_backend="direct_http",
            model_routes={
                "anthropic": DashScopeSettings(
                    api_key="any",
                    provider="anthropic",
                    model="claude-sonnet-5",
                    base_url="https://api.anthropic.com",
                    client_backend="direct_http",
                )
            },
        )
        stdout = io.StringIO()
        captured_requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"data": [{"id": "claude-sonnet-5"}, {"id": "claude-haiku-4-5"}]}
                ).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            captured_requests.append((request, timeout))
            return FakeResponse()

        with ExitStack() as stack:
            stack.enter_context(
                patch("autogen_mas.cli.load_settings", return_value=(config, settings))
            )
            stack.enter_context(patch("urllib.request.urlopen", fake_urlopen))
            stack.enter_context(redirect_stdout(stdout))
            exit_code = main(["list-models", "--llm-provider", "anthropic"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["provider"], "anthropic")
        self.assertEqual(
            payload["models"],
            ["claude-haiku-4-5", "claude-sonnet-5"],
        )
        self.assertEqual(captured_requests[0][0].full_url, "https://api.anthropic.com/v1/models")
        self.assertEqual(captured_requests[0][0].headers["X-api-key"], "any")

    def test_run_single_agent_uses_single_agent_config_and_limit(self) -> None:
        config = ExperimentConfig(
            agents=[
                AgentConfig(agent_id="mas_agent", model="mas-model", temperature=0.9),
            ],
            single_agent=SingleAgentConfig(
                agent_id="baseline",
                model="baseline-model",
                temperature=0.1,
                run_id_prefix="baseline-run",
            ),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.SingleAgentRunner", FakeSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-single-agent", "--limit", "2", "--num-rounds", "2"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeSingleAgentRunner.init_config.single_agent.agent_id, "baseline")
        self.assertEqual(FakeSingleAgentRunner.init_config.question_selection.limit, 2)
        self.assertEqual(FakeSingleAgentRunner.init_config.question_selection.strategy, "first_n")
        self.assertEqual(FakeSingleAgentRunner.init_config.runtime.num_rounds, 2)
        self.assertIs(FakeSingleAgentRunner.init_settings, settings)
        self.assertEqual(len(FakeSingleAgentRunner.received_questions), 2)
        self.assertEqual(FakeSingleAgentRunner.received_progress_callback, "progress")
        self.assertEqual(FakeEvaluator.call["agent_id"], "baseline")
        self.assertEqual(FakeEvaluator.call["model"], "baseline-model")
        self.assertEqual(FakeEvaluator.call["temperature"], 0.1)

    def test_run_single_agent_passes_resume_run_path_to_non_batched_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.SingleAgentRunner", FakeSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "run-single-agent",
                    "--limit",
                    "2",
                    "--resume-run-path",
                    "runs/single-agent-old",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeSingleAgentRunner.received_resume_run_path, "runs/single-agent-old")

    def test_run_single_agent_disables_self_reflection_with_single_round(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.SingleAgentRunner", FakeSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-single-agent", "--no-self-reflection", "--limit", "2"])

        self.assertEqual(exit_code, 0)
        self.assertFalse(FakeSingleAgentRunner.init_config.single_agent.enable_self_reflection)
        self.assertEqual(FakeSingleAgentRunner.init_config.runtime.num_rounds, 1)

    def test_run_single_agent_can_enable_alignment_judge_via_cli(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
            alignment_judge=AlignmentJudgeConfig(enabled=False),
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.SingleAgentRunner", FakeSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-single-agent", "--limit", "2", "--alignment-judge"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(FakeSingleAgentRunner.init_config.alignment_judge.enabled)

    def test_run_single_agent_batched_uses_batch_single_agent_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.BatchSingleAgentRunner", FakeBatchSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_batch_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "run-single-agent",
                    "--batched",
                    "--limit",
                    "3",
                    "--num-rounds",
                    "2",
                    "--answer-batch-size",
                    "2",
                    "--review-batch-size",
                    "2",
                    "--max-workers",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeBatchSingleAgentRunner.init_config.runtime.num_rounds, 2)
        self.assertEqual(FakeBatchSingleAgentRunner.init_config.runtime.max_workers, 1)
        self.assertEqual(len(FakeBatchSingleAgentRunner.received_questions), 3)
        self.assertEqual(FakeBatchSingleAgentRunner.received_progress_callback, "progress")
        self.assertEqual(
            FakeBatchSingleAgentRunner.init_batch_options.answer_max_questions_per_batch,
            2,
        )
        self.assertEqual(
            FakeBatchSingleAgentRunner.init_batch_options.review_max_questions_per_batch,
            2,
        )
        self.assertEqual(
            FakeBatchSingleAgentRunner.init_batch_options.batch_timeout_seconds,
            None,
        )
        self.assertEqual(FakeEvaluator.call["run_path"], "/tmp/batched-single-agent-run")

    def test_run_single_agent_batched_passes_resume_run_path(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.BatchSingleAgentRunner", FakeBatchSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_batch_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "run-single-agent",
                    "--batched",
                    "--limit",
                    "3",
                    "--resume-run-path",
                    "runs/batched-single-agent-old",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            FakeBatchSingleAgentRunner.received_resume_run_path,
            "runs/batched-single-agent-old",
        )

    def test_run_single_agent_batched_disables_self_reflection_with_single_round(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.BatchSingleAgentRunner", FakeBatchSingleAgentRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_batch_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-single-agent", "--batched", "--no-self-reflection"])

        self.assertEqual(exit_code, 0)
        self.assertFalse(FakeBatchSingleAgentRunner.init_config.single_agent.enable_self_reflection)
        self.assertEqual(FakeBatchSingleAgentRunner.init_config.runtime.num_rounds, 1)

    def test_batch_runtime_options_reads_env_file_timeout_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = f"{tmp_dir}/.env"
            with open(env_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "ANSWER_BATCH_TOKENS=7000",
                            "REVIEW_BATCH_TOKENS=5000",
                            "ANSWER_BATCH_SIZE=9",
                            "REVIEW_BATCH_SIZE=6",
                            "BATCH_TIMEOUT_SECONDS=75",
                        ]
                    )
                    + "\n"
                )
            args = SimpleNamespace(
                answer_batch_tokens=None,
                review_batch_tokens=None,
                answer_batch_size=None,
                review_batch_size=None,
                batch_timeout_seconds=None,
            )

            with patch.dict(os.environ, {}, clear=True):
                options = _batch_runtime_options(args, env_path)

        self.assertEqual(options.answer_max_input_tokens, 7000)
        self.assertEqual(options.review_max_input_tokens, 5000)
        self.assertEqual(options.answer_max_questions_per_batch, 9)
        self.assertEqual(options.review_max_questions_per_batch, 6)
        self.assertEqual(options.batch_timeout_seconds, 75)

    def test_single_agent_batch_runtime_defaults_use_token_dynamic_caps(self) -> None:
        args = SimpleNamespace(
            command="run-single-agent",
            answer_batch_tokens=None,
            review_batch_tokens=None,
            answer_batch_size=None,
            review_batch_size=None,
            batch_timeout_seconds=None,
        )

        with patch.dict(os.environ, {}, clear=True):
            options = _batch_runtime_options(args, env_path="/tmp/missing-env-file")

        self.assertEqual(options.answer_max_input_tokens, 4000)
        self.assertEqual(options.review_max_input_tokens, 3000)
        self.assertEqual(options.answer_max_questions_per_batch, 8)
        self.assertEqual(options.review_max_questions_per_batch, 4)
        self.assertIsNone(options.batch_timeout_seconds)

    def test_single_agent_batch_runtime_reads_single_agent_token_env(self) -> None:
        args = SimpleNamespace(
            command="run-single-agent",
            answer_batch_tokens=None,
            review_batch_tokens=None,
            answer_batch_size=None,
            review_batch_size=None,
            batch_timeout_seconds=None,
        )

        with patch.dict(
            os.environ,
            {
                "SINGLE_AGENT_ANSWER_BATCH_TOKENS": "5000",
                "SINGLE_AGENT_REVIEW_BATCH_TOKENS": "3500",
            },
            clear=True,
        ):
            options = _batch_runtime_options(args, env_path="/tmp/missing-env-file")

        self.assertEqual(options.answer_max_input_tokens, 5000)
        self.assertEqual(options.review_max_input_tokens, 3500)

    def test_run_dataset_and_single_agent_share_seeded_question_selection(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(10)
        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(patch("autogen_mas.cli._build_question_progress", return_value=("mas-progress", lambda: None)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            mas_exit_code = main(["run-dataset", "--limit", "4", "--seed", "42"])

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(patch("autogen_mas.cli.SingleAgentRunner", FakeSingleAgentRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(patch("autogen_mas.cli._build_question_progress", return_value=("single-progress", lambda: None)))
            stack.enter_context(redirect_stdout(io.StringIO()))
            single_exit_code = main(["run-single-agent", "--limit", "4", "--seed", "42"])

        self.assertEqual(mas_exit_code, 0)
        self.assertEqual(single_exit_code, 0)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/mas-run")
        mas_ids = [question.question_id for question in FakeCompetitionRunner.received_questions]
        single_ids = [question.question_id for question in FakeSingleAgentRunner.received_questions]
        self.assertEqual(mas_ids, single_ids)
        self.assertEqual(FakeCompetitionRunner.received_progress_callback, "mas-progress")
        self.assertEqual(FakeSingleAgentRunner.received_progress_callback, "single-progress")
        self.assertEqual(FakeCompetitionRunner.init_config.question_selection.seed, 42)
        self.assertEqual(FakeSingleAgentRunner.init_config.question_selection.seed, 42)
        self.assertEqual(
            FakeCompetitionRunner.init_config.question_selection.strategy,
            "random_sample_original_order",
        )
        self.assertEqual(
            FakeSingleAgentRunner.init_config.question_selection.strategy,
            "random_sample_original_order",
        )

    def test_run_dataset_can_disable_alignment_judge_via_cli(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1")],
            ),
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-dataset", "--limit", "2", "--no-alignment-judge"])

        self.assertEqual(exit_code, 0)
        self.assertFalse(FakeCompetitionRunner.init_config.alignment_judge.enabled)

    def test_run_dataset_can_override_num_rounds_via_cli(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/dev.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-dataset", "--dataset", "medmcqa", "--num-rounds", "9"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeCompetitionRunner.init_config.runtime.num_rounds, 9)

    def test_run_adversarial_dataset_disables_alignment_judge_for_baseline1_by_default(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"demo": "data/demo/test.json"},
            alignment_judge=AlignmentJudgeConfig(
                enabled=True,
                judges=[AgentConfig(agent_id="judge_1")],
            ),
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(2)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("adversarial-progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-adversarial-dataset",
                    "--dataset",
                    "demo",
                    "--attack-strategy",
                    "baseline-1",
                    "--limit",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(FakeAdversarialCompetitionRunner.init_config.alignment_judge.enabled)

    def test_run_adversarial_dataset_explicit_alignment_judge_overrides_baseline1_default(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"demo": "data/demo/test.json"},
            alignment_judge=AlignmentJudgeConfig(
                enabled=False,
                judges=[AgentConfig(agent_id="judge_1")],
            ),
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(2)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("adversarial-progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-adversarial-dataset",
                    "--dataset",
                    "demo",
                    "--attack-strategy",
                    "baseline-1",
                    "--limit",
                    "2",
                    "--alignment-judge",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(FakeAdversarialCompetitionRunner.init_config.alignment_judge.enabled)

    def test_run_dataset_rejects_batched_mode(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["run-dataset", "--batched"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --batched", stderr.getvalue())

    def test_load_questions_supports_chess_dataset(self) -> None:
        from autogen_mas.cli import _load_questions

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "chess.json"
            dataset_path.write_text(
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

            questions = _load_questions("chess", str(dataset_path))

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].dataset_name, "chess")
        self.assertEqual(questions[0].task_type, "chess_move")

    def test_load_questions_supports_scalr_dataset(self) -> None:
        from autogen_mas.cli import _load_questions

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "scalr.jsonl"
            dataset_path.write_text(
                (
                    '{"index":"7","question":"Q?",'
                    '"choice_0":"opt1","choice_1":"opt2","choice_2":"opt3",'
                    '"answer":"1"}\n'
                ),
                encoding="utf-8",
            )

            questions = _load_questions("scalr", str(dataset_path))

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].dataset_name, "scalr")
        self.assertEqual(questions[0].task_type, "single_choice")
        self.assertEqual(questions[0].correct_option_ids, ["B"])

    def test_run_dataset_uses_chess_competition_runner_for_chess(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"chess": "data/chess/problems.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(2)
        for question in questions:
            question.dataset_name = "chess"
            question.task_type = "chess_move"

        with (
            patch("autogen_mas.cli.load_settings", return_value=(config, settings)),
            patch("autogen_mas.cli._load_questions", return_value=questions),
            patch("autogen_mas.cli.ChessCompetitionRunner", FakeChessCompetitionRunner),
            patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner),
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            patch("autogen_mas.cli._build_question_progress", return_value=("progress", lambda: None)),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["run-dataset", "--dataset", "chess", "--limit", "2"])

        self.assertEqual(exit_code, 0)
        self.assertIsNone(FakeCompetitionRunner.init_config)
        self.assertIsNotNone(FakeChessCompetitionRunner.init_config)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/chess-mas-run")

    def test_run_adversarial_dataset_uses_independent_runner_and_overrides(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent", temperature=0.9)],
            single_agent=SingleAgentConfig(agent_id="baseline"),
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"truthfulqa": "data/truthfulqa/test.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(10)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("adversarial-progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-adversarial-dataset",
                    "--limit",
                    "4",
                    "--seed",
                    "42",
                    "--num-rounds",
                    "9",
                    "--no-consensus-short-circuit",
                    "--total-agents",
                    "3",
                    "--adversarial-count",
                    "1",
                    "--normal-agent-model",
                    "normal-model",
                    "--adversarial-model",
                    "adversarial-model",
                    "--adversarial-answer-prompt",
                    "Adversarial answer prompt",
                    "--adversarial-review-prompt",
                    "Adversarial review prompt",
                    "--attack-strategy",
                    "7",
                    "--attack-intensity",
                    "high",
                    "--adversarial-code-comment-lines",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIsNone(FakeCompetitionRunner.received_questions)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/adversarial-mas-run")
        self.assertEqual(len(FakeAdversarialCompetitionRunner.received_questions), 4)
        self.assertEqual(
            FakeAdversarialCompetitionRunner.received_progress_callback,
            "adversarial-progress",
        )
        init_config = FakeAdversarialCompetitionRunner.init_config
        self.assertEqual(init_config.agents[0].agent_id, "normal_mas_agent")
        self.assertEqual(init_config.agents[0].temperature, 0.9)
        self.assertEqual(init_config.runtime.num_rounds, 9)
        self.assertFalse(init_config.runtime.consensus_short_circuit)
        self.assertEqual(init_config.adversarial_mas.total_agents, 3)
        self.assertEqual(init_config.adversarial_mas.adversarial_count, 1)
        self.assertEqual(init_config.adversarial_mas.normal_agent_model, "normal-model")
        self.assertEqual(init_config.adversarial_mas.adversarial_agent.model, "adversarial-model")
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.answer_prompt,
            "Adversarial answer prompt",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.review_prompt,
            "Adversarial review prompt",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.attack_strategy.choice,
            "cognitive_manipulation",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.attack_strategy.short_answer,
            "cognitive_manipulation",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.attack_strategy.code,
            "cognitive_manipulation",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.attack_intensity,
            "high",
        )
        self.assertEqual(
            init_config.adversarial_mas.adversarial_agent.code_comment_lines,
            3,
        )

    def test_strategy8_adversarial_dataset_excludes_sample_question_keys_before_selection(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"demo": "data/demo/test.json"},
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.strategy8_sample_question_keys",
                    return_value={"demo__q0__single_choice", "demo__q1__single_choice"},
                )
            )
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("adversarial-progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-adversarial-dataset",
                    "--dataset",
                    "demo",
                    "--attack-strategy",
                    "8",
                    "--limit",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [question.question_id for question in FakeAdversarialCompetitionRunner.received_questions],
            ["q2", "q3"],
        )

    def test_run_adversarial_dataset_can_skip_reasoning_bank_questions_before_selection(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"demo": "data/demo/test.json"},
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            json.dump(
                {
                    "corpus_id": "bank_v2",
                    "sample_id": "run::demo__q0__single_choice::agent_1::1->2::1",
                    "dataset_name": "demo",
                },
                handle,
            )
            handle.write("\n")
            json.dump(
                {
                    "corpus_id": "bank_v2",
                    "question_key": "demo__q1__single_choice",
                    "dataset_name": "demo",
                },
                handle,
            )
            handle.write("\n")
            json.dump(
                {
                    "corpus_id": "other_bank",
                    "question_key": "demo__q4__single_choice",
                    "dataset_name": "demo",
                },
                handle,
            )
            handle.write("\n")
            bank_path = handle.name

        config.adversarial_mas.adversarial_agent.reasoning_bank_path = bank_path
        config.adversarial_mas.adversarial_agent.reasoning_bank_paths = [bank_path]
        config.adversarial_mas.adversarial_agent.reasoning_bank_corpora = ["bank_v2"]

        try:
            with ExitStack() as stack:
                stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
                stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli.AdversarialCompetitionRunner",
                        FakeAdversarialCompetitionRunner,
                    )
                )
                stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli._build_question_progress",
                        return_value=("adversarial-progress", lambda: None),
                    )
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                exit_code = main(
                    [
                        "run-adversarial-dataset",
                        "--dataset",
                        "demo",
                        "--attack-strategy",
                        "9",
                        "--skip-corpus-questions",
                        "--limit",
                        "2",
                    ]
                )
        finally:
            os.unlink(bank_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [question.question_id for question in FakeAdversarialCompetitionRunner.received_questions],
            ["q2", "q3"],
        )

    def test_run_dataset_skips_reasoning_bank_questions_before_selection(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"demo": "data/demo.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump(
                {
                    "corpus_id": "bank_v2",
                    "sample_id": "bank_v2::demo__q0__single_choice::0",
                    "question_key": "demo__q0__single_choice",
                    "dataset_name": "demo",
                },
                handle,
            )
            handle.write("\n")
            json.dump(
                {
                    "corpus_id": "bank_v2",
                    "sample_id": "bank_v2::demo__q1__single_choice::1",
                    "question_key": "demo__q1__single_choice",
                    "dataset_name": "demo",
                },
                handle,
            )
            handle.write("\n")
            bank_path = handle.name

        config.adversarial_mas.adversarial_agent.reasoning_bank_path = bank_path
        config.adversarial_mas.adversarial_agent.reasoning_bank_paths = [bank_path]
        config.adversarial_mas.adversarial_agent.reasoning_bank_corpora = ["bank_v2"]

        try:
            with ExitStack() as stack:
                stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
                stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
                stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
                stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli._build_question_progress",
                        return_value=("progress", lambda: None),
                    )
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                exit_code = main(
                    [
                        "run-dataset",
                        "--dataset",
                        "demo",
                        "--skip-corpus-questions",
                        "--limit",
                        "2",
                    ]
                )
        finally:
            os.unlink(bank_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [question.question_id for question in FakeCompetitionRunner.received_questions],
            ["q2", "q3"],
        )

    def test_run_adversarial_dataset_passes_resume_run_path_to_non_batched_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"ciar": "data/CIAR/CIAR.json"},
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-adversarial-dataset",
                    "--dataset",
                    "ciar",
                    "--limit",
                    "3",
                    "--resume-run-path",
                    "runs/adversarial-competition-old",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            FakeAdversarialCompetitionRunner.received_resume_run_path,
            "runs/adversarial-competition-old",
        )
        self.assertFalse(FakeAdversarialCompetitionRunner.init_debug_mode)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/adversarial-mas-run")

    def test_run_dataset_passes_resume_run_path_to_non_batched_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/dev.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-dataset",
                    "--dataset",
                    "medmcqa",
                    "--limit",
                    "3",
                    "--resume-run-path",
                    "runs/competition-old",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeCompetitionRunner.received_resume_run_path, "runs/competition-old")
        self.assertFalse(FakeCompetitionRunner.init_debug_mode)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/mas-run")

    def test_run_dataset_filters_questions_via_failed_questions_file(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/dev.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(5)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump(
                [
                    {"question_index": 2, "question_key": "demo__q1__single_choice"},
                    {"question_index": 4, "question_key": "demo__q3__single_choice"},
                ],
                handle,
            )
            failed_path = handle.name
        try:
            with ExitStack() as stack:
                stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
                stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
                stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
                stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli._build_question_progress",
                        return_value=("progress", lambda: None),
                    )
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                exit_code = main(
                    [
                        "run-dataset",
                        "--dataset",
                        "medmcqa",
                        "--failed-questions-file",
                        failed_path,
                    ]
                )
        finally:
            os.unlink(failed_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [question.question_id for question in FakeCompetitionRunner.received_questions],
            ["q1", "q3"],
        )

    def test_run_adversarial_dataset_filters_questions_via_failed_questions_file(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"ciar": "data/CIAR/CIAR.json"},
        )
        config.adversarial_mas.total_agents = 1
        config.adversarial_mas.agents = [AgentConfig(agent_id="agent_1")]
        config.adversarial_mas.adversarial_count = 1
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(4)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump(
                [
                    {"question_index": 1, "question_key": "demo__q0__single_choice"},
                    {"question_index": 3, "question_key": "demo__q2__single_choice"},
                ],
                handle,
            )
            failed_path = handle.name
        try:
            with ExitStack() as stack:
                stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
                stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli.AdversarialCompetitionRunner",
                        FakeAdversarialCompetitionRunner,
                    )
                )
                stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
                stack.enter_context(
                    patch(
                        "autogen_mas.cli._build_question_progress",
                        return_value=("progress", lambda: None),
                    )
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                exit_code = main(
                    [
                        "run-adversarial-dataset",
                        "--dataset",
                        "ciar",
                        "--failed-questions-file",
                        failed_path,
                    ]
                )
        finally:
            os.unlink(failed_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [question.question_id for question in FakeAdversarialCompetitionRunner.received_questions],
            ["q0", "q2"],
        )

    def test_failed_questions_file_prefers_stable_identifiers_over_indexes(self) -> None:
        questions = self._questions(4)

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "question_index": 1,
                        "question_id": "q2",
                        "question_key": "demo__q2__single_choice",
                    }
                ],
                handle,
            )
            failed_path = handle.name
        try:
            filtered = _filter_questions_by_failed_questions_file(questions, failed_path)
        finally:
            os.unlink(failed_path)

        self.assertEqual([question.question_id for question in filtered], ["q2"])

    def test_run_dataset_debug_passes_debug_mode_to_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/dev.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(["run-dataset", "--dataset", "medmcqa", "--debug"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(FakeCompetitionRunner.init_debug_mode)
        self.assertFalse(FakeCompetitionRunner.init_persist_feedback_summary)

    def test_run_dataset_debug_can_persist_feedback_summary(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"medmcqa": "data/medmcqa/dev.json"},
        )
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(patch("autogen_mas.cli.CompetitionRunner", FakeCompetitionRunner))
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                [
                    "run-dataset",
                    "--dataset",
                    "medmcqa",
                    "--debug",
                    "--persist-feedback-summary",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(FakeCompetitionRunner.init_debug_mode)
        self.assertTrue(FakeCompetitionRunner.init_persist_feedback_summary)

    def test_persist_feedback_summary_requires_debug(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["run-dataset", "--dataset", "medmcqa", "--persist-feedback-summary"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--persist-feedback-summary requires --debug", stderr.getvalue())

    def test_run_adversarial_dataset_debug_passes_debug_mode_to_runner(self) -> None:
        config = ExperimentConfig(
            agents=[AgentConfig(agent_id="normal_mas_agent")],
            runtime=RuntimeConfig(num_rounds=3, output_dir="runs"),
            datasets={"ciar": "data/CIAR/CIAR.json"},
        )
        self._enable_adversarial_runner_config(config)
        settings = DashScopeSettings(api_key="test-key", model="default-model")
        questions = self._questions(3)

        with ExitStack() as stack:
            stack.enter_context(patch("autogen_mas.cli.load_settings", return_value=(config, settings)))
            stack.enter_context(patch("autogen_mas.cli._load_questions", return_value=questions))
            stack.enter_context(
                patch(
                    "autogen_mas.cli.AdversarialCompetitionRunner",
                    FakeAdversarialCompetitionRunner,
                )
            )
            stack.enter_context(patch("autogen_mas.cli.Evaluator", FakeEvaluator))
            stack.enter_context(
                patch(
                    "autogen_mas.cli._build_question_progress",
                    return_value=("progress", lambda: None),
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = main(
                ["run-adversarial-dataset", "--dataset", "ciar", "--debug"]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(FakeAdversarialCompetitionRunner.init_debug_mode)
        self.assertFalse(
            FakeAdversarialCompetitionRunner.init_persist_feedback_summary
        )

    def test_run_dataset_rejects_batch_size_arguments(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["run-dataset", "--answer-batch-size", "4"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --answer-batch-size 4", stderr.getvalue())

    def test_run_adversarial_dataset_rejects_batched_mode(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["run-adversarial-dataset", "--batched"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --batched", stderr.getvalue())

    def test_run_adversarial_dataset_rejects_batch_size_arguments(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["run-adversarial-dataset", "--answer-batch-size", "4"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --answer-batch-size 4", stderr.getvalue())

    def test_evaluate_passes_top_agent_selection_rule(self) -> None:
        with (
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "evaluate",
                    "--run-path",
                    "/tmp/mas-run",
                    "--selection-rule",
                    "top-agent",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(FakeEvaluator.evaluate_call["run_path"], "/tmp/mas-run")
        self.assertEqual(FakeEvaluator.evaluate_call["selection_rule"], "top-agent")

    def test_evaluate_round_sweep_passes_options(self) -> None:
        with (
            patch("autogen_mas.cli.Evaluator", FakeEvaluator),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "evaluate-round-sweep",
                    "--run-path",
                    "/tmp/mas-run",
                    "--selection-rule",
                    "top-agent",
                    "--max-round",
                    "9",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            FakeEvaluator.evaluate_round_sweep_call,
            {
                "run_path": "/tmp/mas-run",
                "selection_rule": "top-agent",
                "max_round": 9,
            },
        )


if __name__ == "__main__":
    unittest.main()
