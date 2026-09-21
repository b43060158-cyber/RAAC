from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autogen_mas.config import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DASHSCOPE_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
    load_dashscope_settings,
    load_experiment_config,
    load_llm_settings,
)


class ConfigLoadingTest(unittest.TestCase):
    def test_loads_deepseek_settings_from_current_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
DEEPSEEK_API_KEY=deepseek-test-key
DEEPSEEK_MODEL=deepseek-reasoner
DEEPSEEK_TIMEOUT_SECONDS=90
DEEPSEEK_MAX_RETRIES=1
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.api_key, "deepseek-test-key")
        self.assertEqual(settings.base_url, DEFAULT_DEEPSEEK_BASE_URL)
        self.assertEqual(settings.model, "deepseek-reasoner")
        self.assertEqual(settings.timeout_seconds, 90)
        self.assertEqual(settings.max_retries, 1)
        self.assertEqual(settings.client_backend, "direct_http")

    def test_unified_llm_settings_override_provider_specific_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=deepseek
LLM_API_KEY=unified-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://custom.deepseek.example/v1
LLM_TIMEOUT_SECONDS=45
LLM_MAX_RETRIES=0
LLM_CLIENT_BACKEND=direct_http
DEEPSEEK_API_KEY=provider-key
DEEPSEEK_MODEL=deepseek-reasoner
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.api_key, "unified-key")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertEqual(settings.base_url, "https://custom.deepseek.example/v1")
        self.assertEqual(settings.timeout_seconds, 45)
        self.assertEqual(settings.max_retries, 0)
        self.assertEqual(settings.client_backend, "direct_http")

    def test_duplicate_llm_profiles_create_model_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=dashscope
LLM_API_KEY=dashscope-test-key
LLM_MODEL=qwen3-32b
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_CLIENT_BACKEND=direct_http

LLM_PROVIDER=deepseek
LLM_API_KEY=deepseek-test-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_CLIENT_BACKEND=direct_http
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertIn("dashscope", settings.model_routes)
        self.assertEqual(settings.model_routes["dashscope"].model, "qwen3-32b")
        self.assertEqual(
            settings.model_routes["dashscope"].base_url,
            DEFAULT_DASHSCOPE_BASE_URL,
        )

    def test_duplicate_llm_profiles_support_anthropic_compatible_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=anthropic
LLM_API_KEY=proxy-test-key
LLM_MODEL=claude-sonnet-4-6
LLM_BASE_URL=https://api.anthropic.com
LLM_CLIENT_BACKEND=direct_http

LLM_PROVIDER=deepseek
LLM_API_KEY=deepseek-test-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_CLIENT_BACKEND=direct_http
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-chat")
        self.assertIn("anthropic", settings.model_routes)
        self.assertEqual(settings.model_routes["anthropic"].model, "claude-sonnet-4-6")
        self.assertEqual(
            settings.model_routes["anthropic"].base_url,
            "https://api.anthropic.com",
        )

    def test_anthropic_provider_defaults_to_public_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=anthropic
LLM_API_KEY=anthropic-test-key
LLM_MODEL=claude-sonnet-4-6
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "anthropic")
        self.assertEqual(settings.api_key, "anthropic-test-key")
        self.assertEqual(settings.base_url, DEFAULT_ANTHROPIC_BASE_URL)
        self.assertEqual(settings.model, "claude-sonnet-4-6")
        self.assertEqual(settings.client_backend, "direct_http")

    def test_ollama_provider_defaults_to_local_openai_compatible_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.api_key, "any")
        self.assertEqual(settings.base_url, DEFAULT_OLLAMA_BASE_URL)
        self.assertEqual(settings.model, "gemma3:4b")
        self.assertEqual(settings.client_backend, "direct_http")

    def test_duplicate_llm_profiles_support_ollama_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b

LLM_PROVIDER=deepseek
LLM_API_KEY=deepseek-test-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_CLIENT_BACKEND=direct_http
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertIn("ollama", settings.model_routes)
        self.assertEqual(settings.model_routes["ollama"].api_key, "any")
        self.assertEqual(settings.model_routes["ollama"].model, "gemma3:4b")
        self.assertEqual(settings.model_routes["ollama"].base_url, DEFAULT_OLLAMA_BASE_URL)

    def test_legacy_dashscope_settings_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                """
DASHSCOPE_API_KEY=dashscope-test-key
QWEN_MODEL_ID=qwen-test
MAS_LLM_CLIENT=direct_http
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_dashscope_settings(env_path)

        self.assertEqual(settings.provider, "dashscope")
        self.assertEqual(settings.api_key, "dashscope-test-key")
        self.assertEqual(settings.base_url, DEFAULT_DASHSCOPE_BASE_URL)
        self.assertEqual(settings.model, "qwen-test")
        self.assertEqual(settings.client_backend, "direct_http")

    def test_loads_single_agent_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "prompts": {
    "answer": "Global answer prompt",
    "review": "Global review prompt"
  },
  "agents": [
    {
      "agent_id": "agent_1",
      "model": "mas-model",
      "temperature": 0.9
    }
  ],
  "single_agent": {
    "agent_id": "baseline",
    "model": "baseline-model",
    "temperature": 0.1,
    "answer_prompt": "Baseline answer prompt",
    "run_id_prefix": "baseline-run",
    "enable_self_reflection": false
  },
  "runtime": {
    "num_rounds": 3,
    "output_dir": "runs",
    "max_workers": 2,
    "run_id_prefix": "competition",
    "consensus_short_circuit": false
  },
  "datasets": {
    "truthfulqa": "data/truthfulqa/test.json"
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(config.single_agent.agent_id, "baseline")
        self.assertEqual(config.single_agent.model, "baseline-model")
        self.assertEqual(config.single_agent.temperature, 0.1)
        self.assertEqual(config.single_agent.answer_prompt, "Baseline answer prompt")
        self.assertEqual(config.single_agent.run_id_prefix, "baseline-run")
        self.assertFalse(config.single_agent.enable_self_reflection)
        self.assertFalse(config.runtime.consensus_short_circuit)
        self.assertEqual(config.agents[0].model, "mas-model")
        self.assertEqual(config.agents[0].temperature, 0.9)

    def test_loads_single_agent_dataset_specific_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [
    {
      "agent_id": "agent_1"
    }
  ],
  "single_agent": {
    "answer_prompt": "Default baseline prompt",
    "answer_prompts_by_dataset": {
      "ciar": "CIAR baseline prompt",
      "truthfulqa": "TruthfulQA baseline prompt"
    }
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(config.single_agent.answer_prompt, "Default baseline prompt")
        self.assertEqual(
            config.single_agent.answer_prompts_by_dataset,
            {
                "ciar": "CIAR baseline prompt",
                "truthfulqa": "TruthfulQA baseline prompt",
            },
        )
        self.assertEqual(
            config.single_agent.resolved_answer_prompt_for_dataset("ciar", "Fallback prompt"),
            "CIAR baseline prompt",
        )
        self.assertEqual(
            config.single_agent.resolved_answer_prompt_for_dataset(
                "medmcqa",
                "Fallback prompt",
            ),
            "Default baseline prompt",
        )

    def test_repo_config_keeps_faireval_single_agent_comparison_prompt(self) -> None:
        config = load_experiment_config(Path("config/paper.yaml"))

        prompt = config.single_agent.answer_prompts_by_dataset["faireval"]

        self.assertIn("Choose A if response 1 is better", prompt)
        self.assertIn("choose B if the two responses are equally good overall", prompt)
        self.assertIn("choose C if response 2 is better", prompt)

    def test_single_agent_defaults_inherit_global_prompt_and_llm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "prompts": {
    "answer": "Global answer prompt",
    "review": "Global review prompt"
  },
  "agents": [
    {
      "agent_id": "agent_1"
    }
  ]
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        agent_config = config.single_agent.to_agent_config()
        self.assertEqual(config.single_agent.agent_id, "single_agent")
        self.assertIsNone(config.single_agent.model)
        self.assertEqual(agent_config.resolved_answer_prompt(config.answer_prompt), "Global answer prompt")
        self.assertEqual(config.single_agent.temperature, 0.2)
        self.assertEqual(config.single_agent.run_id_prefix, "single-agent")
        self.assertFalse(config.driver_attribution.enabled)

    def test_loads_normal_agent_answer_mitigation_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "normal_agent_answer_mitigation": {
    "enabled": true,
    "start_round": 2,
    "message": "Mitigation message override."
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertTrue(config.normal_agent_answer_mitigation.enabled)
        self.assertEqual(config.normal_agent_answer_mitigation.start_round, 2)
        self.assertEqual(
            config.normal_agent_answer_mitigation.message,
            "Mitigation message override.",
        )

    def test_loads_driver_attribution_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "driver_attribution": {
    "enabled": true,
    "prompt": "Driver prompt",
    "judges": [
      {"agent_id": "driver_judge_1", "model": null, "temperature": 0.0},
      {"agent_id": "driver_judge_2", "model": "judge-model", "temperature": 0.3}
    ],
    "strong_driver_threshold": 0.8,
    "plausible_driver_threshold": 0.4,
    "max_workers": 1,
    "fallback_attribution": "self_decision_change"
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertTrue(config.driver_attribution.enabled)
        self.assertEqual(config.driver_attribution.prompt, "Driver prompt")
        self.assertEqual(len(config.driver_attribution.judges), 2)
        self.assertIsNone(config.driver_attribution.judges[0].model)
        self.assertEqual(config.driver_attribution.judges[1].model, "judge-model")
        self.assertEqual(config.driver_attribution.judges[1].temperature, 0.3)
        self.assertEqual(config.driver_attribution.strong_driver_threshold, 0.8)
        self.assertEqual(config.driver_attribution.plausible_driver_threshold, 0.4)
        self.assertEqual(config.driver_attribution.max_workers, 1)

    def test_loads_alignment_judge_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "alignment_judge": {
    "enabled": true,
    "prompt": "Alignment judge prompt",
    "judges": [
      {"agent_id": "alignment_judge_1", "model": null, "temperature": 0.0},
      {"agent_id": "alignment_judge_2", "model": "judge-model", "temperature": 0.3}
    ],
    "fallback_policy": "keep_structured"
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertTrue(config.alignment_judge.enabled)
        self.assertEqual(config.alignment_judge.prompt, "Alignment judge prompt")
        self.assertEqual(len(config.alignment_judge.judges), 2)
        self.assertIsNone(config.alignment_judge.judges[0].model)
        self.assertEqual(config.alignment_judge.judges[1].model, "judge-model")
        self.assertEqual(config.alignment_judge.judges[1].temperature, 0.3)
        self.assertEqual(config.alignment_judge.fallback_policy, "keep_structured")

    def test_loads_independent_adversarial_mas_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [
    {"agent_id": "normal_agent_1", "temperature": 0.9}
  ],
  "adversarial_mas": {
    "total_agents": 2,
    "adversarial_count": 1,
    "normal_agent_model": "normal-model",
    "seed": 123,
    "run_id_prefix": "adv-run",
    "agents": [
      {"agent_id": "agent_1", "temperature": 0.1},
      {"agent_id": "agent_2", "temperature": 0.2}
            ],
            "adversarial_agent": {
              "model": "adversarial-model",
              "answer_prompt": "Adversarial answer prompt",
              "review_prompt": "Adversarial review prompt",
              "attack_strategy": "7",
              "attack_intensity": "high",
              "wrong_answer_strategy": "random_single_wrong",
              "high_score": 10,
      "low_score": 1
    }
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(config.agents[0].agent_id, "normal_agent_1")
        self.assertEqual(config.agents[0].temperature, 0.9)
        self.assertEqual(config.adversarial_mas.total_agents, 2)
        self.assertEqual(config.adversarial_mas.adversarial_count, 1)
        self.assertEqual(config.adversarial_mas.normal_agent_model, "normal-model")
        self.assertEqual(config.adversarial_mas.seed, 123)
        self.assertEqual(config.adversarial_mas.run_id_prefix, "adv-run")
        self.assertEqual(config.adversarial_mas.agents[0].agent_id, "agent_1")
        self.assertEqual(config.adversarial_mas.agents[0].temperature, 0.1)
        self.assertEqual(config.adversarial_mas.adversarial_agent.model, "adversarial-model")
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.answer_prompt,
            "Adversarial answer prompt",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.choice,
            "cognitive_manipulation",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.short_answer,
            "cognitive_manipulation",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.code,
            "cognitive_manipulation",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_intensity,
            "high",
        )

    def test_removed_attack_strategy_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "adversarial_mas": {
    "total_agents": 2,
    "adversarial_count": 1,
    "agents": [
      {"agent_id": "agent_1"},
      {"agent_id": "agent_2"}
    ],
    "adversarial_agent": {
      "attack_strategy": "6"
    }
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "attack_strategy"):
                load_experiment_config(config_path)

    def test_strategy8_alias_only_changes_choice_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "adversarial_mas": {
    "total_agents": 2,
    "adversarial_count": 1,
    "agents": [
      {"agent_id": "agent_1"},
      {"agent_id": "agent_2"}
    ],
    "adversarial_agent": {
      "attack_strategy": "8"
    }
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.choice,
            "few_shot_rerank",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.short_answer,
            "random_target_answer",
        )
        self.assertEqual(
            config.adversarial_mas.adversarial_agent.attack_strategy.code,
            "comment_lines",
        )

    def test_top_level_adversarial_agent_model_alias_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "adversarial_mas": {
    "total_agents": 2,
    "adversarial_count": 1,
    "adversarial_agent_model": "adversarial-alias-model",
    "agents": [
      {"agent_id": "agent_1"},
      {"agent_id": "agent_2"}
    ],
    "adversarial_agent": {
      "model": null
    }
  }
}
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(
            config.adversarial_mas.adversarial_agent.model,
            "adversarial-alias-model",
        )

    def test_adversarial_agent_behavior_rejects_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "adversarial_mas": {
    "total_agents": 1,
    "adversarial_count": 1,
    "agents": [{"agent_id": "agent_1", "temperature": 0.4}],
    "adversarial_agent": {
      "temperature": 0.8
    }
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not define temperature"):
                load_experiment_config(config_path)

    def test_adversarial_count_cannot_exceed_total_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "adversarial_mas": {
    "total_agents": 1,
    "adversarial_count": 2,
    "agents": [{"agent_id": "agent_1"}]
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not exceed total_agents"):
                load_experiment_config(config_path)

    def test_enabled_driver_attribution_requires_judges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "driver_attribution": {
    "enabled": true,
    "judges": []
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "driver_attribution.judges"):
                load_experiment_config(config_path)

    def test_enabled_alignment_judge_requires_judges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "agents.json"
            config_path.write_text(
                """
{
  "agents": [{"agent_id": "agent_1"}],
  "alignment_judge": {
    "enabled": true,
    "judges": []
  }
}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "alignment_judge.judges"):
                load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
