# Released data

This directory contains the exact evaluation inputs used by the artifact and the released RAAC reasoning bank. The benchmark files are redistributed under their upstream licenses; the project does not relicense them.

## Inventory

| Path | Examples | SHA-256 |
| --- | ---: | --- |
| `medmcqa/dev.json` | 4,183 | `a2b5f53890f4e742b728378fb847bc5338088d48ea01218676379f587270af31` |
| `mmlu/test/astronomy_test.csv` | 152 | `87c9c3cb5a55395c70dd9ed521dd7d2629fd4ff236c5ab647102e007f52a9341` |
| `mmlu/test/business_ethics_test.csv` | 100 | `cd3fc3a85a34cb17e6c5b2fb68ea61e7800b31120d92934019f1e17f4832fbd4` |
| `mmlu/test/electrical_engineering_test.csv` | 145 | `8b0d747a97cb2cbbd8926c8deb2cda6073da6d5d3cc099693aa7761be938f5e4` |
| `mmlu/test/global_facts_test.csv` | 100 | `a81058ee8e2385edaf42b8847ec9db01221fa4b16d262db415f3370884745ff9` |
| `mmlu/test/moral_scenarios_test.csv` | 895 | `128d7bd823d793f8b7faf28f98f4b3c737b3e84d4b42c06c13458d36075a423a` |
| `mmlu/test/public_relations_test.csv` | 110 | `3712f955d0c38b073b188a175a8d9a6ce64601f0ae9c98d8cfc37461c4cb5933` |
| `mmlu/test/us_foreign_policy_test.csv` | 100 | `2821ee009a2b1b7d4adc5a212a0bad7fe05d377455de0fe0841beab8f00142e1` |
| `scalr/test.jsonl` | 571 | `98a4e0933dc1beb4df24bfeaf356456ea3741590cf3857d8373d3b78ed3d4fd0` |
| `truthfulqa/test.json` | 817 | `b7ef813f7a7bedd957c6db2ef23f660d5ba74250784b6d6b021456823c254c77` |
| `reasoning_bank/adversarial_reasoning_bank_v2.jsonl` | 803 | `237697175f84034f344bcfe1943c499a7aa6b026db6701f9ca16b262589eefdf` |

The MMLU count is 1,602 across seven paper-selected subject files. CSV row counts above include no header because the upstream files are headerless.

## Provenance, licenses, and citations

### MedMCQA

- Source: <https://github.com/medmcqa/medmcqa>
- License: MIT; see `../LICENSES/MedMCQA-MIT.txt`.
- Citation: Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. “MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering.” CHIL/PMLR 174, 2022.

Only the 4,183-example paper evaluation file is included. The 147 MB training split and the answer-hidden split are intentionally excluded.

### MMLU

- Source: <https://github.com/hendrycks/test>
- License: MIT; see `../LICENSES/MMLU-MIT.txt`.
- Citation: Dan Hendrycks et al. “Measuring Massive Multitask Language Understanding.” ICLR 2021.

Only the seven subjects used by this artifact are included.

### SCALR / LegalBench

- Source and task card: <https://hazyresearch.stanford.edu/legalbench/tasks/scalr.html>
- License: Creative Commons Attribution 4.0; see `../LICENSES/CC-BY-4.0.txt`.
- Citation: Neel Guha et al. “LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models.” NeurIPS Datasets and Benchmarks, 2023.

The released file contains all 571 SCALR examples.

### TruthfulQA

- Source: <https://github.com/sylinrl/TruthfulQA>
- License: Apache License 2.0; see `../LICENSES/Apache-2.0.txt`.
- Citation: Stephanie Lin, Jacob Hilton, and Owain Evans. “TruthfulQA: Measuring How Models Mimic Human Falsehoods.” ACL 2022.

The released file is the 817-example multiple-choice evaluation set in the format consumed by this runtime.

### RAAC reasoning bank

`reasoning_bank/adversarial_reasoning_bank_v2.jsonl` is a project-generated research artifact built from successful influence cases and model-generated analysis. It is released under CC BY 4.0, subject to the licenses of any benchmark question text embedded in individual records. Cite the DAI 2026 paper when using it.
