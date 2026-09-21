"""Offline integrity and loadability checks for the public artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autogen_mas.dataset_adapters import (
    MMLUAdapter,
    MedMCQAAdapter,
    SCALRAdapter,
    TruthfulQAAdapter,
)


EXPECTED_HASHES = {
    "data/medmcqa/dev.json": "a2b5f53890f4e742b728378fb847bc5338088d48ea01218676379f587270af31",
    "data/mmlu/test/astronomy_test.csv": "87c9c3cb5a55395c70dd9ed521dd7d2629fd4ff236c5ab647102e007f52a9341",
    "data/mmlu/test/business_ethics_test.csv": "cd3fc3a85a34cb17e6c5b2fb68ea61e7800b31120d92934019f1e17f4832fbd4",
    "data/mmlu/test/electrical_engineering_test.csv": "8b0d747a97cb2cbbd8926c8deb2cda6073da6d5d3cc099693aa7761be938f5e4",
    "data/mmlu/test/global_facts_test.csv": "a81058ee8e2385edaf42b8847ec9db01221fa4b16d262db415f3370884745ff9",
    "data/mmlu/test/moral_scenarios_test.csv": "128d7bd823d793f8b7faf28f98f4b3c737b3e84d4b42c06c13458d36075a423a",
    "data/mmlu/test/public_relations_test.csv": "3712f955d0c38b073b188a175a8d9a6ce64601f0ae9c98d8cfc37461c4cb5933",
    "data/mmlu/test/us_foreign_policy_test.csv": "2821ee009a2b1b7d4adc5a212a0bad7fe05d377455de0fe0841beab8f00142e1",
    "data/scalr/test.jsonl": "98a4e0933dc1beb4df24bfeaf356456ea3741590cf3857d8373d3b78ed3d4fd0",
    "data/truthfulqa/test.json": "b7ef813f7a7bedd957c6db2ef23f660d5ba74250784b6d6b021456823c254c77",
    "data/reasoning_bank/adversarial_reasoning_bank_v2.jsonl": "237697175f84034f344bcfe1943c499a7aa6b026db6701f9ca16b262589eefdf",
}

EXPECTED_COUNTS = {
    "medmcqa": 4_183,
    "mmlu": 1_602,
    "scalr": 571,
    "truthfulqa": 817,
    "reasoning_bank": 803,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    for relative_path, expected in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        actual = sha256(path) if path.is_file() else "missing"
        if actual != expected:
            failures.append(f"{relative_path}: expected {expected}, got {actual}")

    datasets = {
        "medmcqa": MedMCQAAdapter(ROOT / "data/medmcqa/dev.json").load(),
        "mmlu": MMLUAdapter(ROOT / "data/mmlu/test").load(),
        "scalr": SCALRAdapter(ROOT / "data/scalr/test.jsonl").load(),
        "truthfulqa": TruthfulQAAdapter(ROOT / "data/truthfulqa/test.json").load(),
    }
    for name, records in datasets.items():
        actual = len(records)
        expected = EXPECTED_COUNTS[name]
        if actual != expected:
            failures.append(f"{name}: expected {expected} records, got {actual}")
        print(f"{name}: {actual} records")

    bank_path = ROOT / "data/reasoning_bank/adversarial_reasoning_bank_v2.jsonl"
    with bank_path.open(encoding="utf-8") as handle:
        bank_count = sum(1 for line in handle if line.strip())
    print(f"reasoning_bank: {bank_count} records")
    if bank_count != EXPECTED_COUNTS["reasoning_bank"]:
        failures.append(
            f"reasoning_bank: expected {EXPECTED_COUNTS['reasoning_bank']} records, "
            f"got {bank_count}"
        )

    if failures:
        print("\nArtifact validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nArtifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
