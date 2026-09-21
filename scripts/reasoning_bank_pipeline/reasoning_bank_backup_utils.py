"""Shared backup helpers for reasoning-bank rebuild and merge operations."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_BACKUP_DIR = RUNS_DIR / "reasoning_bank_backups"


def _dedupe_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def create_reasoning_bank_backup_bundle(
    *,
    target_path: Path,
    source_paths: Iterable[Path],
    backup_dir: Path | None = None,
    label: str = "rebuild",
) -> Path | None:
    files_to_archive = _dedupe_existing_paths([target_path, *source_paths])
    if not files_to_archive:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination_dir = (backup_dir or DEFAULT_BACKUP_DIR).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"{target_path.stem}_{label}_{timestamp}.tar.gz"
    archive_path = destination_dir / archive_name

    manifest = {
        "created_at": timestamp,
        "label": label,
        "target_path": str(target_path.expanduser().resolve()),
        "archived_files": [str(path) for path in files_to_archive],
    }

    with tarfile.open(archive_path, "w:gz") as tar:
        for path in files_to_archive:
            try:
                arcname = path.relative_to(PROJECT_ROOT)
            except ValueError:
                arcname = Path(path.name)
            tar.add(path, arcname=str(arcname))

        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        tarinfo = tarfile.TarInfo(name="backup_manifest.json")
        tarinfo.size = len(manifest_bytes)
        tar.addfile(tarinfo, io.BytesIO(manifest_bytes))

    return archive_path
