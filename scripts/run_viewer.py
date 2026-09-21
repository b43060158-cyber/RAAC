from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = PROJECT_ROOT / "viewer"
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_RUN_PATH = RUNS_ROOT


class ViewerError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ViewerError(f"找不到文件：{path}", HTTPStatus.NOT_FOUND) from exc
    except json.JSONDecodeError as exc:
        raise ViewerError(f"JSON 解析失败：{path} ({exc})") from exc


def resolve_run_path(raw_path: str | None) -> Path:
    raw = (raw_path or str(DEFAULT_RUN_PATH)).strip()
    if not raw:
        raise ViewerError("请输入 run 目录路径。")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise ViewerError(f"路径不存在：{path}", HTTPStatus.NOT_FOUND)
    if not path.is_dir():
        raise ViewerError(f"路径不是目录：{path}")
    if not (path / "questions").is_dir():
        raise ViewerError(f"缺少 questions 目录：{path / 'questions'}")
    return path


def question_key_from_payload(payload: dict[str, Any], fallback: str) -> str:
    return str(
        payload.get("question_key")
        or f"{payload.get('dataset_name', '')}__{payload.get('question_id', '')}__{payload.get('task_type', '')}"
        or fallback
    )


def load_question_payloads(run_path: Path) -> list[tuple[Path, dict[str, Any]]]:
    question_dir = run_path / "questions"
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(question_dir.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, dict):
            payloads.append((path, payload))
    if not payloads:
        raise ViewerError(f"questions 目录下没有可读取的题目 JSON：{question_dir}", HTTPStatus.NOT_FOUND)
    return payloads


def build_runs_response(runs_root: Path = RUNS_ROOT) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if not runs_root.exists():
        return {"runs_root": str(runs_root), "runs": []}
    for path in sorted(runs_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_dir() or not (path / "questions").is_dir():
            continue
        question_paths = sorted((path / "questions").glob("*.json"))
        if not question_paths:
            continue
        _, results_by_key = load_evaluation(path)
        if not results_by_key:
            continue
        question_keys: set[str] = set()
        for question_path in question_paths:
            try:
                payload = read_json(question_path)
            except ViewerError:
                continue
            if isinstance(payload, dict):
                question_keys.add(question_key_from_payload(payload, question_path.stem))
        evaluated_question_count = sum(1 for key in question_keys if key in results_by_key)
        if evaluated_question_count == 0:
            continue
        manifest: dict[str, Any] = {}
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            try:
                loaded_manifest = read_json(manifest_path)
                if isinstance(loaded_manifest, dict):
                    manifest = loaded_manifest
            except ViewerError:
                manifest = {}
        summary: dict[str, Any] = {}
        summary_path = path / "evaluation" / "summary.json"
        if summary_path.exists():
            try:
                loaded_summary = read_json(summary_path)
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            except ViewerError:
                summary = {}
        question_count = len(question_paths)
        runs.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "run_id": str(manifest.get("run_id") or path.name),
                "created_at": str(manifest.get("created_at") or ""),
                "question_count": question_count,
                "evaluated_question_count": evaluated_question_count,
                "missing_evaluation_count": max(question_count - evaluated_question_count, 0),
                "accuracy": summary.get("accuracy"),
                "has_evaluation": True,
                "modified_at": path.stat().st_mtime,
            }
        )
    return {"runs_root": str(runs_root.resolve()), "runs": runs}


def load_evaluation(run_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evaluation_dir = run_path / "evaluation"
    summary_path = evaluation_dir / "summary.json"
    results_path = evaluation_dir / "question_results.json"

    summary: dict[str, Any] = {}
    results_by_key: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        loaded_summary = read_json(summary_path)
        if isinstance(loaded_summary, dict):
            summary = loaded_summary
    if results_path.exists():
        loaded_results = read_json(results_path)
        if isinstance(loaded_results, list):
            for item in loaded_results:
                if isinstance(item, dict) and item.get("question_key"):
                    results_by_key[str(item["question_key"])] = item
    return summary, results_by_key


def option_map(payload: dict[str, Any]) -> dict[str, str]:
    options = payload.get("options") or []
    mapping: dict[str, str] = {}
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict):
                option_id = str(option.get("option_id", ""))
                if option_id:
                    mapping[option_id] = str(option.get("text", ""))
    return mapping


def answer_label(answer: dict[str, Any], options_by_id: dict[str, str]) -> str:
    selected = answer.get("selected_option_ids") or []
    if isinstance(selected, list) and selected:
        parts = []
        for option_id in selected:
            key = str(option_id)
            text = options_by_id.get(key, "")
            parts.append(f"{key}. {text}" if text else key)
        return "; ".join(parts)
    final_answer = str(answer.get("final_answer") or "").strip()
    if final_answer:
        return final_answer
    code = str(answer.get("code") or "").strip()
    if code:
        first_line = code.splitlines()[0] if code.splitlines() else code
        return first_line[:100] + ("..." if len(first_line) > 100 else "")
    return "未提供答案"


def collect_agents(manifest: dict[str, Any], payloads: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    agents: set[str] = set()
    config_agents = (
        manifest.get("experiment_config", {}).get("agents", [])
        if isinstance(manifest.get("experiment_config"), dict)
        else []
    )
    if isinstance(config_agents, list):
        for agent in config_agents:
            if isinstance(agent, dict) and agent.get("agent_id"):
                agents.add(str(agent["agent_id"]))

    for _, payload in payloads:
        for round_data in payload.get("rounds", []) or []:
            for result in round_data.get("agent_results", []) or []:
                if isinstance(result, dict) and result.get("agent_id"):
                    agents.add(str(result["agent_id"]))
    return sorted(agents)


def collect_round_indexes(payloads: list[tuple[Path, dict[str, Any]]]) -> list[int]:
    indexes: set[int] = set()
    for _, payload in payloads:
        for round_data in payload.get("rounds", []) or []:
            try:
                indexes.add(int(round_data.get("round_index")))
            except (TypeError, ValueError):
                continue
    return sorted(indexes)


def build_question_summary(
    path: Path,
    payload: dict[str, Any],
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    key = question_key_from_payload(payload, path.stem)
    options_by_id = option_map(payload)
    changed_agents: set[str] = set()
    round_count = 0
    final_answers: dict[str, str] = {}
    final_confidences: dict[str, Any] = {}
    final_round = None

    rounds = payload.get("rounds") or []
    if isinstance(rounds, list):
        round_count = len(rounds)
        final_round = rounds[-1] if rounds else None
        for round_data in rounds:
            for result in round_data.get("agent_results", []) or []:
                answer = result.get("answer") or {}
                if answer.get("changed_answer"):
                    changed_agents.add(str(result.get("agent_id") or answer.get("agent_id") or ""))
        if isinstance(final_round, dict):
            for result in final_round.get("agent_results", []) or []:
                agent_id = str(result.get("agent_id", ""))
                answer = result.get("answer") or {}
                final_answers[agent_id] = answer_label(answer, options_by_id)
                final_confidences[agent_id] = answer.get("confidence")

    selected_agent_id = str((evaluation or {}).get("selected_agent_id") or "")
    selected_answer = final_answers.get(selected_agent_id, "")
    question_text = str(payload.get("question") or payload.get("prompt") or "")

    return {
        "question_key": key,
        "question_id": str(payload.get("question_id", "")),
        "dataset_name": str(payload.get("dataset_name", "")),
        "task_type": str(payload.get("task_type", "")),
        "question_preview": question_text[:220],
        "round_count": round_count,
        "selected_agent_id": selected_agent_id,
        "selected_answer": selected_answer,
        "selected_confidence": final_confidences.get(selected_agent_id),
        "correct_option_ids": (evaluation or {}).get("correct_option_ids")
        or payload.get("correct_option_ids")
        or [],
        "predicted_option_ids": (evaluation or {}).get("predicted_option_ids") or [],
        "predicted_final_answer": (evaluation or {}).get("predicted_final_answer") or "",
        "is_correct": (evaluation or {}).get("is_correct"),
        "is_tie": bool((evaluation or {}).get("is_tie")),
        "excluded_from_accuracy": bool((evaluation or {}).get("excluded_from_accuracy")),
        "changed_agents": sorted(agent for agent in changed_agents if agent),
        "has_changes": bool(changed_agents),
    }


def build_run_response(run_path: Path) -> dict[str, Any]:
    manifest_path = run_path / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    summary, results_by_key = load_evaluation(run_path)
    payloads = load_question_payloads(run_path)
    questions = [
        build_question_summary(path, payload, results_by_key.get(question_key_from_payload(payload, path.stem)))
        for path, payload in payloads
    ]

    return {
        "run_path": str(run_path),
        "run_id": str(manifest.get("run_id") or run_path.name),
        "manifest": manifest,
        "summary": summary,
        "agents": collect_agents(manifest, payloads),
        "round_indexes": collect_round_indexes(payloads),
        "question_count": len(questions),
        "questions": questions,
    }


def build_question_response(run_path: Path, question_key: str) -> dict[str, Any]:
    summary, results_by_key = load_evaluation(run_path)
    _ = summary
    for path, payload in load_question_payloads(run_path):
        key = question_key_from_payload(payload, path.stem)
        if key == question_key or path.stem == question_key:
            manifest_path = run_path / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            return {
                "run_path": str(run_path),
                "question_path": str(path),
                "question": payload,
                "evaluation": results_by_key.get(key, {}),
                "agents": collect_agents(manifest if isinstance(manifest, dict) else {}, [(path, payload)]),
                "round_indexes": collect_round_indexes([(path, payload)]),
            }
    raise ViewerError(f"找不到题目：{question_key}", HTTPStatus.NOT_FOUND)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "MASViewer/1.0"

    def do_HEAD(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.serve_static(parsed.path, head_only=True)
        except ViewerError as exc:
            self.send_response(exc.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self.send_json({"ok": True, "default_run_path": str(DEFAULT_RUN_PATH)})
                return
            if parsed.path == "/api/runs":
                self.send_json(build_runs_response())
                return
            if parsed.path == "/api/run":
                query = parse_qs(parsed.query)
                run_path = resolve_run_path(query.get("path", [None])[0])
                self.send_json(build_run_response(run_path))
                return
            if parsed.path == "/api/question":
                query = parse_qs(parsed.query)
                run_path = resolve_run_path(query.get("path", [None])[0])
                question_key = (query.get("question_key", [""])[0] or "").strip()
                if not question_key:
                    raise ViewerError("缺少 question_key 参数。")
                self.send_json(build_question_response(run_path, question_key))
                return
            self.serve_static(parsed.path)
        except ViewerError as exc:
            self.send_json({"error": str(exc)}, status=exc.status)
        except Exception as exc:  # pragma: no cover - defensive for local UI diagnostics
            self.send_json({"error": f"服务内部错误：{exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path: str, *, head_only: bool = False) -> None:
        relative = unquote(request_path).lstrip("/") or "index.html"
        target = (VIEWER_ROOT / relative).resolve()
        if not str(target).startswith(str(VIEWER_ROOT.resolve())):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local visual viewer for MAS competition results.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"MAS result viewer running at {url}")
    print(f"Default run path: {DEFAULT_RUN_PATH}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
