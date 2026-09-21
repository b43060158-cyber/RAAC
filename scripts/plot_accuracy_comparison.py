from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TITLE = "LLM-MAS Accuracy Comparison"
DEFAULT_MODEL_ID = "deepseekv4-flash"


class SummaryLoadError(RuntimeError):
    """Raised when an evaluation summary cannot be loaded for plotting."""


@dataclass(frozen=True)
class RunAccuracy:
    run_path: Path
    label: str
    dataset_names: tuple[str, ...]
    total_questions: int
    accuracy: float
    correct_questions: int
    evaluated_questions: int

    @property
    def accuracy_percent(self) -> float:
        return self.accuracy * 100.0


def load_run_accuracy(run_path: str | Path, *, label: str) -> RunAccuracy:
    path = Path(run_path)
    summary_path = path / "evaluation" / "summary.json"
    if not summary_path.exists():
        raise SummaryLoadError(
            f"Evaluation summary not found: {summary_path}. "
            "Please run evaluation first so <run_path>/evaluation/summary.json exists."
        )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryLoadError(f"Invalid JSON in evaluation summary: {summary_path}") from exc

    return RunAccuracy(
        run_path=path,
        label=label,
        dataset_names=_load_dataset_names(path),
        total_questions=_required_int(summary, "total_questions", summary_path),
        accuracy=_required_float(summary, "accuracy", summary_path),
        correct_questions=_required_int(summary, "correct_questions", summary_path),
        evaluated_questions=_optional_int(
            summary,
            "evaluated_questions",
            fallback_key="total_questions",
            summary_path=summary_path,
        ),
    )


def default_output_path(normal_run_path: str | Path, adversarial_run_path: str | Path) -> Path:
    normal_name = Path(normal_run_path).name
    adversarial_name = Path(adversarial_run_path).name
    return PROJECT_ROOT / "runs" / f"accuracy_comparison_{normal_name}_vs_{adversarial_name}.png"


def default_multi_output_path(
    normal_run_path: str | Path,
    adversarial_run_paths: Sequence[str | Path],
) -> Path:
    normal_name = Path(normal_run_path).name
    adversarial_names = [Path(run_path).name for run_path in adversarial_run_paths]
    joined_adversarial_names = "_".join(adversarial_names)
    return PROJECT_ROOT / "runs" / f"accuracy_comparison_{normal_name}_vs_{joined_adversarial_names}.png"


def plot_accuracy_comparison(
    normal: RunAccuracy,
    adversarial: RunAccuracy | Sequence[RunAccuracy],
    *,
    output: str | Path,
    title: str = DEFAULT_TITLE,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    adversarial_runs = _as_adversarial_runs(adversarial)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = [normal, *adversarial_runs]
    accuracies = [run.accuracy_percent for run in runs]
    labels = [run.label for run in runs]
    palette = ["#B85042", "#4D6FA9", "#C9862B", "#6E5A8A", "#8A6E4D", "#4D8A7C"]
    colors = ["#2E7D68", *[palette[index % len(palette)] for index in range(len(adversarial_runs))]]

    fig_width = max(8.0, 1.7 * len(runs) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.4), dpi=160)
    bars = ax.bar(labels, accuracies, color=colors[: len(runs)], width=0.55)

    dataset_label = _comparison_dataset_label(runs)
    total_label = _comparison_total_label(runs)
    y_limit = min(110.0, max(100.0, max(accuracies) + 12.0))
    ax.set_ylim(0, y_limit)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"{title}\n"
        f"Model: {model_id} | Dataset: {dataset_label} | Total questions: {total_label}"
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, run in zip(bars, runs, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2.0,
            f"{height:.1f}%\n{run.correct_questions}/{run.evaluated_questions}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    drop_points_by_label = {
        run.label: normal.accuracy_percent - run.accuracy_percent
        for run in adversarial_runs
    }
    for bar, run in zip(bars[1:], adversarial_runs, strict=True):
        drop_points = drop_points_by_label[run.label]
        annotation_label = (
            f"Drop {drop_points:.1f} pp"
            if drop_points >= 0
            else f"Increase {abs(drop_points):.1f} pp"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(y_limit - 3.0, bar.get_height() + 12.0),
            annotation_label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if len(runs) > 3:
        ax.tick_params(axis="x", labelrotation=18)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    result = {
        "output_path": str(output_path),
        "model_id": model_id,
        "dataset": dataset_label,
        "total_questions": total_label,
        "normal_accuracy": normal.accuracy,
        "adversarial_accuracies": {
            run.label: run.accuracy
            for run in adversarial_runs
        },
        "accuracy_drop_points": drop_points_by_label,
    }
    if len(adversarial_runs) == 1:
        only_run = adversarial_runs[0]
        result["adversarial_accuracy"] = only_run.accuracy
        result["accuracy_drop_points"] = drop_points_by_label[only_run.label]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot accuracy comparison between normal and adversarial LLM-MAS runs."
    )
    parser.add_argument("normal_run_path", help="Path to the normal LLM-MAS run result.")
    parser.add_argument(
        "adversarial_run_path",
        nargs="?",
        help="Backward-compatible path to one adversarial LLM-MAS run result.",
    )
    parser.add_argument(
        "--adversarial",
        action="append",
        default=[],
        metavar="STRATEGY_NAME=RUN_PATH",
        help="Adversarial strategy name and run path. Repeat for multiple strategies.",
    )
    parser.add_argument("--output", help="Output PNG path.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Chart title.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model identifier shown in the chart and CLI output. Defaults to {DEFAULT_MODEL_ID}.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normal = load_run_accuracy(args.normal_run_path, label="Normal LLM-MAS")
        adversarial_specs = _parse_adversarial_specs(
            args.adversarial_run_path,
            args.adversarial,
        )
        adversarial_runs = [
            load_run_accuracy(run_path, label=strategy_name)
            for strategy_name, run_path in adversarial_specs
        ]
        for adversarial_run in adversarial_runs:
            if normal.dataset_names != adversarial_run.dataset_names:
                print(
                    "warning: dataset names differ between runs "
                    f"({normal.label}: {', '.join(normal.dataset_names)} vs "
                    f"{adversarial_run.label}: {', '.join(adversarial_run.dataset_names)}); plotting anyway.",
                    file=sys.stderr,
                )
            if normal.total_questions != adversarial_run.total_questions:
                print(
                    "warning: total_questions differs between runs "
                    f"({normal.label}: {normal.total_questions} vs "
                    f"{adversarial_run.label}: {adversarial_run.total_questions}); plotting anyway.",
                    file=sys.stderr,
                )
            if normal.evaluated_questions != adversarial_run.evaluated_questions:
                print(
                    "warning: evaluated_questions differs between runs "
                    f"({normal.label}: {normal.evaluated_questions} vs "
                    f"{adversarial_run.label}: {adversarial_run.evaluated_questions}); plotting anyway.",
                    file=sys.stderr,
                )

        output = args.output or (
            default_output_path(args.normal_run_path, args.adversarial_run_path)
            if args.adversarial_run_path and not args.adversarial
            else default_multi_output_path(
                args.normal_run_path,
                [run_path for _strategy_name, run_path in adversarial_specs],
            )
        )
        result = plot_accuracy_comparison(
            normal,
            adversarial_runs,
            output=output,
            title=args.title,
            model_id=args.model_id,
        )
    except SummaryLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"dataset={result['dataset']}")
    print(f"model_id={result['model_id']}")
    print(f"total_questions={result['total_questions']}")
    print(f"normal_accuracy={normal.accuracy_percent:.1f}%")
    for run in adversarial_runs:
        drop_points = result["accuracy_drop_points"]
        run_drop_points = drop_points[run.label] if isinstance(drop_points, dict) else drop_points
        print(f"adversarial_accuracy[{run.label}]={run.accuracy_percent:.1f}%")
        print(f"accuracy_drop_points[{run.label}]={run_drop_points:.1f}")
    print(f"output_path={result['output_path']}")
    return 0


def _parse_adversarial_specs(
    positional_run_path: str | None,
    named_specs: list[str],
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if positional_run_path:
        specs.append(("Adversarial LLM-MAS", positional_run_path))
    for raw_spec in named_specs:
        if "=" not in raw_spec:
            raise ValueError(
                "--adversarial must use STRATEGY_NAME=RUN_PATH, "
                f"got: {raw_spec}"
            )
        strategy_name, run_path = raw_spec.split("=", 1)
        strategy_name = strategy_name.strip()
        run_path = run_path.strip()
        if not strategy_name:
            raise ValueError(
                "--adversarial strategy name cannot be empty; "
                f"got: {raw_spec}"
            )
        if not run_path:
            raise ValueError(
                "--adversarial run path cannot be empty; "
                f"got: {raw_spec}"
            )
        specs.append((strategy_name, run_path))
    if not specs:
        raise ValueError(
            "Provide either adversarial_run_path or at least one "
            "--adversarial STRATEGY_NAME=RUN_PATH."
        )
    labels = [label for label, _path in specs]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(f"Duplicate adversarial strategy name(s): {', '.join(duplicate_labels)}")
    return specs


def _as_adversarial_runs(adversarial: RunAccuracy | Sequence[RunAccuracy]) -> list[RunAccuracy]:
    if isinstance(adversarial, RunAccuracy):
        return [adversarial]
    runs = list(adversarial)
    if not runs:
        raise SummaryLoadError("At least one adversarial run is required for plotting.")
    return runs


def _required_float(summary: dict[str, Any], key: str, summary_path: Path) -> float:
    if key not in summary:
        raise SummaryLoadError(f"Missing required field '{key}' in evaluation summary: {summary_path}")
    value = summary[key]
    if not isinstance(value, int | float):
        raise SummaryLoadError(f"Field '{key}' must be numeric in evaluation summary: {summary_path}")
    return float(value)


def _required_int(summary: dict[str, Any], key: str, summary_path: Path) -> int:
    if key not in summary:
        raise SummaryLoadError(f"Missing required field '{key}' in evaluation summary: {summary_path}")
    value = summary[key]
    if not isinstance(value, int):
        raise SummaryLoadError(f"Field '{key}' must be an integer in evaluation summary: {summary_path}")
    return value


def _optional_int(
    summary: dict[str, Any],
    key: str,
    *,
    fallback_key: str,
    summary_path: Path,
) -> int:
    if key in summary:
        return _required_int(summary, key, summary_path)
    return _required_int(summary, fallback_key, summary_path)


def _load_dataset_names(run_path: Path) -> tuple[str, ...]:
    question_dir = run_path / "questions"
    question_files = sorted(question_dir.glob("*.json"))
    names: set[str] = set()
    for question_file in question_files:
        try:
            payload = json.loads(question_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SummaryLoadError(f"Invalid JSON in question result: {question_file}") from exc
        dataset_name = str(payload.get("dataset_name", "")).strip()
        if dataset_name:
            names.add(dataset_name)
    if not names:
        raise SummaryLoadError(
            f"Could not determine dataset name from question results in: {question_dir}"
        )
    return tuple(sorted(names))


def _comparison_dataset_label(runs: Sequence[RunAccuracy]) -> str:
    first_dataset_names = runs[0].dataset_names
    if all(run.dataset_names == first_dataset_names for run in runs):
        return ", ".join(first_dataset_names)
    return "; ".join(f"{run.label}: {', '.join(run.dataset_names)}" for run in runs)


def _comparison_total_label(runs: Sequence[RunAccuracy]) -> str:
    first_total_questions = runs[0].total_questions
    if all(run.total_questions == first_total_questions for run in runs):
        return str(first_total_questions)
    return "; ".join(f"{run.label}: {run.total_questions}" for run in runs)


if __name__ == "__main__":
    raise SystemExit(main())
