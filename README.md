# DAI 2026 Multi-Agent Reasoning Artifact

This directory is the self-contained research artifact for an accepted [DAI 2026](https://www.adai.ai/dai/2026/index.html) paper on adversarial robustness in LLM-based multi-agent reasoning. It contains the experiment runtime, the four evaluation datasets used in the paper, the released reasoning bank, paper-aligned configuration, evaluation utilities, and unit tests.

The artifact supports three paper settings:

- **Clean MAS**: four agents answer and peer-review over three rounds.
- **RAAC (Strategy 13)**: one adversarial agent uses retrieval-augmented, target-specific persuasion.
- **MCA (Baseline 1)**: the comparison attack without the RAAC retrieval strategy.

## Repository layout

```text
autogen_mas/          Core runtime and evaluators
config/paper.yaml     Paper-aligned experiment configuration
data/                 Evaluation data and released reasoning bank
docs/                 Prompt specification and reported-results appendix
results/              Machine-readable reported aggregate results
scripts/              Corpus, analysis, plotting, and viewer utilities
tests/unit/            Offline unit test suite
viewer/                Static run-inspection interface
```

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

Edit `.env` with an API key and model. The runtime supports DeepSeek, DashScope-compatible Qwen, Anthropic-compatible endpoints, and local Ollama. Never commit `.env`.

For a local Ollama endpoint, the example values work after changing `LLM_MODEL` to an installed model:

```dotenv
LLM_PROVIDER=ollama
LLM_API_KEY=any
LLM_MODEL=your-local-model
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_CLIENT_BACKEND=direct_http
```

## Offline verification

These checks do not call an LLM API:

```bash
python scripts/validate_artifact.py
python -m pytest -q tests/unit
```

The validator checks every released dataset file by SHA-256, loads all four datasets through the public adapters, and validates the reasoning-bank record count.

## Quick start

Run these commands from the repository root. Start with a small limit because a full four-agent, three-round experiment makes many model calls.

`--seed` controls which questions are drawn from a dataset, so use the same value across the settings you want to compare. Any integer works:

```bash
export SEED=12345
```

`--skip-corpus-questions` removes the questions that the released reasoning bank lists as source questions (matched by question key) before sampling, which keeps the evaluated questions separate from the bank. Use it for all runs, including Clean MAS, so that all settings are evaluated on the same questions.

Clean MAS:

```bash
python -m autogen_mas --config config/paper.yaml run-dataset \
  --dataset truthfulqa --limit 5 --seed "$SEED" --skip-corpus-questions
```

RAAC / Strategy 13:

```bash
python -m autogen_mas --config config/paper.yaml run-adversarial-dataset \
  --dataset truthfulqa --limit 5 --seed "$SEED" --skip-corpus-questions \
  --attack-strategy 13
```

MCA / Baseline 1:

```bash
python -m autogen_mas --config config/paper.yaml run-adversarial-dataset \
  --dataset truthfulqa --limit 5 --seed "$SEED" --skip-corpus-questions \
  --attack-strategy baseline_1
```

Replace `truthfulqa` with `medmcqa`, `mmlu`, or `scalr`. Results are written under `runs/`. Evaluate a completed run with:

```bash
python -m autogen_mas evaluate --run-path runs/<run-id>
```

The paper used 100 questions per dataset, three rounds, four agents, and three independent runs per setting on the same question set. For example:

```bash
python -m autogen_mas --config config/paper.yaml run-adversarial-dataset \
  --dataset mmlu --limit 100 --seed "$SEED" --skip-corpus-questions \
  --attack-strategy 13
```

To use a different model without editing the configuration, add `--llm-provider`, `--llm-model`, and, when needed, `--normal-agent-model` and `--adversarial-model`.

## Data

The repository includes only the evaluation material required by the paper, not complete upstream training corpora. See [data/README.md](data/README.md) for file hashes, provenance, licenses, and citations. In summary:

| Dataset | Released examples | Upstream license |
| --- | ---: | --- |
| MedMCQA | 4,183 | MIT |
| MMLU | 1,602 across 7 subject files | MIT |
| SCALR / LegalBench | 571 | CC BY 4.0 |
| TruthfulQA | 817 | Apache-2.0 |
| RAAC reasoning bank | 803 | Project-generated artifact; see data notice |

## Ethical-use note

This artifact studies how a malicious participant can influence peer agents in a controlled evaluation setting. It is released for reproducibility, auditing, and defensive research. Do not deploy the attack components against systems or users without authorization.

## License and citation

Project code, configuration, documentation, and scripts are released under the MIT License; see `LICENSE`. Bundled benchmark data remains under its upstream license, documented in `data/README.md` and `LICENSES/`.

If you use this artifact, please cite the DAI 2026 paper:

```bibtex
@inproceedings{dai2026_raac,
  title     = {RAAC: A State-Aware Retrieval-Aided Adversarial Commentary Attack on Discussion-Based LLM Multi-Agent Systems},
  booktitle = {Proceedings of the 8th International Conference on Distributed Artificial Intelligence},
  year      = {2026}
}
```
