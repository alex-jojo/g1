#!/usr/bin/env python3
"""Download math benchmarks and evaluate GradAlign Qwen3-0.6B checkpoints.

The default evaluation suite covers all requested benchmarks: GSM8K, MATH-500,
Minerva Math, the Qwen/OlympiadBench 675-problem English text-only mathematics
subset, AIME 2024/2025/2026, HMMT February 2026, and AMC 2023. The default
sampling configuration uses eight responses per problem and reports AVG@8,
P@8, and maj@8.

Examples:
    # Backward-compatible download commands
    python scripts/download_math_benchmarks.py
    python scripts/download_math_benchmarks.py olympiadbench
    python scripts/download_math_benchmarks.py download gsm8k math500

    # Inspect checkpoints without loading a model
    python scripts/download_math_benchmarks.py checkpoints --selection both

    # Eight-sample evaluation of the latest SVD and random checkpoints
    python scripts/download_math_benchmarks.py evaluate --selection both

    # Evaluate every complete checkpoint (10, 20, ..., 100)
    python scripts/download_math_benchmarks.py evaluate \
        --selection both --steps all

Evaluation reuses the repository's Ray/vLLM inference implementation and the
same ``rewards/grpo_math_verify_reward.py`` mathematical-equivalence verifier
used for training. VERL FSDP shards are merged automatically only when a usable
Hugging Face ``merged`` or ``merged_for_eval`` directory is absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    expected_rows: int
    required_keys: frozenset[str]
    urls: tuple[str, ...]
    homepage: str
    description: str
    hf_dataset: str | None = None
    hf_split: str = "train"


@dataclass(frozen=True)
class CheckpointTarget:
    selection: str
    checkpoint_root: Path
    experiment: str
    step: int

    @property
    def experiment_dir(self) -> Path:
        return self.checkpoint_root / self.experiment

    @property
    def step_dir(self) -> Path:
        return self.experiment_dir / f"global_step_{self.step}"

    @property
    def actor_dir(self) -> Path:
        return self.step_dir / "actor"


SPECS: Mapping[str, BenchmarkSpec] = {
    "aime24": BenchmarkSpec(
        name="aime24",
        expected_rows=30,
        required_keys=frozenset({"id", "problem", "answer"}),
        urls=(
            "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
            "main/evaluation/data/aime24/test.jsonl",
        ),
        homepage="https://huggingface.co/datasets/math-ai/aime24",
        description="AIME I and AIME II 2024 (15 + 15 problems).",
    ),
    "aime25": BenchmarkSpec(
        name="aime25",
        expected_rows=30,
        required_keys=frozenset({"id", "problem", "answer"}),
        urls=(
            "https://huggingface.co/datasets/math-ai/aime25/"
            "resolve/main/test.jsonl?download=true",
        ),
        homepage="https://huggingface.co/datasets/math-ai/aime25",
        description="AIME I and AIME II 2025 (15 + 15 problems).",
    ),
    "aime26": BenchmarkSpec(
        name="aime26",
        expected_rows=30,
        required_keys=frozenset({"id", "problem", "answer"}),
        urls=(
            "https://huggingface.co/datasets/math-ai/aime26/"
            "resolve/main/aime2026.jsonl?download=true",
        ),
        homepage="https://huggingface.co/datasets/math-ai/aime26",
        description="AIME I and AIME II 2026 (15 + 15 problems).",
    ),
    "hmmt26": BenchmarkSpec(
        name="hmmt26",
        expected_rows=33,
        required_keys=frozenset({"problem_idx", "problem", "answer"}),
        urls=(),
        homepage="https://huggingface.co/datasets/MathArena/hmmt_feb_2026",
        description="The 33 final-answer problems from HMMT February 2026.",
        hf_dataset="MathArena/hmmt_feb_2026",
        hf_split="train",
    ),
    "amc23": BenchmarkSpec(
        name="amc23",
        expected_rows=40,
        required_keys=frozenset({"id", "problem", "answer"}),
        urls=(
            "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
            "main/evaluation/data/amc23/test.jsonl",
        ),
        homepage="https://github.com/QwenLM/Qwen2.5-Math/tree/main/"
        "evaluation/data/amc23",
        description="The common 40-problem AMC 12A/12B 2023 subset.",
    ),
    "gsm8k": BenchmarkSpec(
        name="gsm8k",
        expected_rows=1_319,
        required_keys=frozenset({"question", "answer"}),
        urls=(
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "master/grade_school_math/data/test.jsonl",
        ),
        homepage="https://github.com/openai/grade-school-math",
        description="Official GSM8K test split.",
    ),
    "math500": BenchmarkSpec(
        name="math500",
        expected_rows=500,
        required_keys=frozenset({"problem", "solution", "answer"}),
        urls=(
            "https://github.com/openai/prm800k/raw/main/"
            "prm800k/math_splits/test.jsonl",
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
            "resolve/main/test.jsonl?download=true",
        ),
        homepage=(
            "https://github.com/openai/prm800k/tree/main/"
            "prm800k/math_splits"
        ),
        description="OpenAI's representative 500-problem MATH test subset.",
    ),
    "minerva_math": BenchmarkSpec(
        name="minerva_math",
        expected_rows=272,
        required_keys=frozenset({"problem", "solution", "type", "idx"}),
        urls=(
            "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
            "main/evaluation/data/minerva_math/test.jsonl",
        ),
        homepage=(
            "https://research.google/pubs/"
            "solving-quantitative-reasoning-problems-with-language-models/"
        ),
        description=(
            "The 272 undergraduate STEM problems from MIT OpenCourseWare "
            "introduced with Minerva."
        ),
    ),
    "olympiadbench": BenchmarkSpec(
        name="olympiadbench",
        expected_rows=675,
        required_keys=frozenset(
            {"id", "question", "solution", "final_answer"}
        ),
        urls=(
            "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/"
            "main/evaluation/data/olympiadbench/test.jsonl",
        ),
        homepage="https://github.com/OpenBMB/OlympiadBench",
        description=(
            "The Qwen 675-problem OE_TO_maths_en_COMP snapshot: English, "
            "text-only, open-ended competition mathematics."
        ),
    ),
}

ALIASES = {
    "aime-24": "aime24",
    "aime2024": "aime24",
    "aime-2024": "aime24",
    "aime-25": "aime25",
    "aime2025": "aime25",
    "aime-2025": "aime25",
    "aime-26": "aime26",
    "aime2026": "aime26",
    "aime-2026": "aime26",
    "hmmt-26": "hmmt26",
    "hmmt2026": "hmmt26",
    "hmmt-2026": "hmmt26",
    "hmmt-feb-2026": "hmmt26",
    "amc-23": "amc23",
    "amc2023": "amc23",
    "amc-2023": "amc23",
    "gsm-8k": "gsm8k",
    "math-500": "math500",
    "minerva": "minerva_math",
    "minerva-math": "minerva_math",
    "olympiad-bench": "olympiadbench",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_ROOT = Path(
    os.environ.get("GRADALIGN_STORAGE_ROOT", "/media/chenzhipeng/cll")
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "GRADALIGN_BENCHMARK_ROOT",
        str(DEFAULT_STORAGE_ROOT / "datasets" / "math_benchmarks"),
    )
).expanduser()
DEFAULT_HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT", "https://hf-mirror.com"
).rstrip("/")
DEFAULT_GITHUB_MIRROR = os.environ.get(
    "GRADALIGN_GITHUB_MIRROR", "https://ghfast.top"
).rstrip("/")
DEFAULT_RESULTS_ROOT = (
    DEFAULT_STORAGE_ROOT / "runtime" / "evaluations" / "math_benchmarks"
)
DEFAULT_SVD_ROOT = (
    DEFAULT_STORAGE_ROOT / "runtime" / "checkpoints" / "gradalign_svd"
)
DEFAULT_RANDOM_ROOT = (
    DEFAULT_STORAGE_ROOT / "runtime" / "checkpoints" / "random"
)
DEFAULT_SVD_EXPERIMENT = (
    "svd_erank_qwen3_0_6b_unified_reward_v1_"
    "qwen3-0.6b-base_1to8_gsm_math_dsr_test_no_val"
)
DEFAULT_RANDOM_EXPERIMENT = (
    "random_qwen3_0_6b_unified_reward_v1_"
    "qwen3-0.6b-base_1to8_qwen3_0_6b_pass_0.125_0.875_no_val"
)
DEFAULT_EVAL_BENCHMARKS = (
    "gsm8k",
    "math500",
    "minerva_math",
    "olympiadbench",
    "aime24",
    "aime25",
    "aime26",
    "hmmt26",
    "amc23",
)
# Every evaluated benchmark has equal status in the leaderboard and macro
# statistics. Keep one canonical ordering instead of maintaining a subset.
LEADERBOARD_BENCHMARKS = DEFAULT_EVAL_BENCHMARKS
DEFAULT_REWARD_PATH = REPO_ROOT / "rewards" / "grpo_math_verify_reward.py"
DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
DEFAULT_CUDA_VISIBLE_DEVICES = os.environ.get(
    "CUDA_VISIBLE_DEVICES", "4,5,6,7"
)
USER_AGENT = "GradAlign-math-benchmark-evaluator/2.0"
ACTOR_SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
STEP_DIR_RE = re.compile(r"global_step_(\d+)$")
COMMANDS = frozenset({"download", "sources", "checkpoints", "evaluate"})


def normalize_names(
    names: Sequence[str] | None,
    *,
    default: Sequence[str] | None = None,
) -> list[str]:
    """Normalize aliases and reject unknown or duplicate benchmark names."""
    if not names:
        return list(default if default is not None else SPECS)

    normalized: list[str] = []
    for raw_name in names:
        name = ALIASES.get(raw_name.lower(), raw_name.lower())
        if name not in SPECS:
            choices = ", ".join(SPECS)
            raise ValueError(f"unknown benchmark {raw_name!r}; choose from: {choices}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def benchmark_path(output_dir: Path, name: str) -> Path:
    return output_dir / name / "test.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded objects from a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on physical line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}: line {line_number} is {type(item).__name__}, "
                    "expected a JSON object"
                )
            yield item


def validate_file(path: Path, spec: BenchmarkSpec) -> int:
    """Validate JSONL syntax, schema and exact row count."""
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"{path}: downloaded a Git LFS pointer, not the dataset")

    count = 0
    for count, item in enumerate(iter_jsonl(path), start=1):
        missing = spec.required_keys.difference(item)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"{path}: record {count} is missing required keys: {missing_text}"
            )

    if count != spec.expected_rows:
        raise ValueError(
            f"{path}: expected {spec.expected_rows} records, found {count}"
        )
    return count


def endpoint_is_enabled(value: str | None) -> bool:
    return bool(value and value.strip().lower() not in {"direct", "none", "off"})


def resolve_download_url(url: str) -> str:
    """Route canonical GitHub/Hugging Face URLs through configured mirrors."""
    hf_endpoint = os.environ.get(
        "GRADALIGN_HF_DOWNLOAD_ENDPOINT",
        os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
    ).rstrip("/")
    github_mirror = os.environ.get(
        "GRADALIGN_GITHUB_MIRROR", DEFAULT_GITHUB_MIRROR
    ).rstrip("/")

    hf_prefix = "https://huggingface.co"
    if url.startswith(hf_prefix) and endpoint_is_enabled(hf_endpoint):
        return hf_endpoint + url[len(hf_prefix) :]

    github_prefixes = (
        "https://github.com/",
        "https://raw.githubusercontent.com/",
    )
    if url.startswith(github_prefixes) and endpoint_is_enabled(github_mirror):
        return f"{github_mirror}/{url}"
    return url


def download_to_temp(url: str, destination_dir: Path, timeout: float) -> Path:
    """Stream a URL to a temporary file in the destination filesystem."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, temporary_name = tempfile.mkstemp(
        prefix=".benchmark-download-", suffix=".part", dir=destination_dir
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def export_hf_dataset_to_temp(
    spec: BenchmarkSpec, destination_dir: Path
) -> Path:
    """Load a small Hugging Face split and export it as canonical JSONL."""
    if spec.hf_dataset is None:
        raise ValueError(f"{spec.name} has no Hugging Face dataset configured")
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            f"downloading {spec.name} requires the 'datasets' package"
        ) from exc

    dataset = datasets.load_dataset(spec.hf_dataset, split=spec.hf_split)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".benchmark-hf-", suffix=".jsonl.part", dir=destination_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in dataset:
                json.dump(
                    dict(row),
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                handle.write("\n")
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def install_benchmark(
    spec: BenchmarkSpec,
    destination: Path,
    *,
    repair: bool,
    timeout: float,
) -> Path:
    """Keep a valid local file, or download and atomically install it."""
    if destination.exists():
        try:
            rows = validate_file(destination, spec)
        except (OSError, ValueError) as exc:
            if not repair:
                raise RuntimeError(
                    f"existing file failed validation: {exc}. "
                    "Re-run with --repair to replace it."
                ) from exc
            print(f"[repair] {spec.name}: {exc}")
        else:
            print(f"[skip]   {spec.name}: {destination} ({rows} records)")
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in spec.urls:
        resolved_url = resolve_download_url(url)
        temporary_path: Path | None = None
        try:
            print(f"[fetch]  {spec.name}: {resolved_url}")
            temporary_path = download_to_temp(
                resolved_url, destination.parent, timeout
            )
            rows = validate_file(temporary_path, spec)
            os.replace(temporary_path, destination)
            print(f"[saved]  {spec.name}: {destination} ({rows} records)")
            return destination
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"{resolved_url}: {exc}")
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    if spec.hf_dataset is not None:
        temporary_path = None
        source = f"hf://datasets/{spec.hf_dataset}/{spec.hf_split}"
        try:
            print(f"[fetch]  {spec.name}: {source}")
            temporary_path = export_hf_dataset_to_temp(spec, destination.parent)
            rows = validate_file(temporary_path, spec)
            os.replace(temporary_path, destination)
            print(f"[saved]  {spec.name}: {destination} ({rows} records)")
            return destination
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    details = "\n  - ".join(errors)
    raise RuntimeError(f"all download sources failed for {spec.name}:\n  - {details}")


def ensure_benchmarks(
    names: Sequence[str] | None = None,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repair: bool = False,
    timeout: float = 60.0,
    default: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Ensure selected benchmarks exist locally and return their paths."""
    selected = normalize_names(names, default=default)
    root = Path(output_dir).expanduser().resolve()
    paths: dict[str, Path] = {}
    for name in selected:
        spec = SPECS[name]
        destination = benchmark_path(root, name)
        paths[name] = install_benchmark(
            spec, destination, repair=repair, timeout=timeout
        )
    return paths


def print_sources() -> None:
    for spec in SPECS.values():
        print(f"{spec.name}: {spec.expected_rows} records")
        print(f"  {spec.description}")
        print(f"  homepage: {spec.homepage}")
        for url in spec.urls:
            print(f"  download: {url}")
        if spec.hf_dataset is not None:
            print(
                f"  download: hf://datasets/{spec.hf_dataset}/{spec.hf_split}"
            )


def configure_download_environment(args: argparse.Namespace) -> None:
    """Configure mirrors and caches before urllib or datasets is imported."""
    hf_endpoint = str(args.hf_endpoint).rstrip("/")
    github_mirror = str(args.github_mirror).rstrip("/")
    os.environ["GRADALIGN_HF_DOWNLOAD_ENDPOINT"] = hf_endpoint
    if endpoint_is_enabled(hf_endpoint):
        os.environ["HF_ENDPOINT"] = hf_endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)
    os.environ["GRADALIGN_GITHUB_MIRROR"] = github_mirror
    os.environ.setdefault(
        "HF_HOME", str(DEFAULT_STORAGE_ROOT / "cache" / "huggingface")
    )
    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        str(Path(os.environ["HF_HOME"]) / "datasets"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent, default=str)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                count += 1
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def extract_last_boxed_content(text: str) -> str:
    """Extract the final balanced ``\\boxed{...}`` value from reference text."""
    marker = r"\boxed"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError("reference solution contains no \\boxed answer")
    opening = text.find("{", start + len(marker))
    if opening < 0:
        raise ValueError("final \\boxed marker has no opening brace")
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    raise ValueError("final \\boxed answer has unbalanced braces")


def format_scalar_answer(value: Any) -> str:
    """Avoid turning integer-valued AMC answers into strings such as '27.0'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def format_category(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(values) or None
    text = str(value or "").strip()
    return text or None


def benchmark_question_and_answer(
    name: str, row: dict[str, Any]
) -> tuple[str, str, str | None, str]:
    """Return question, ground truth, category, and stable source identifier."""
    if name == "gsm8k":
        question = str(row["question"])
        raw_answer = str(row["answer"])
        if "####" not in raw_answer:
            raise ValueError("GSM8K answer is missing the #### final-answer delimiter")
        answer = raw_answer.rsplit("####", 1)[1].strip()
        category = None
        source_id = str(row.get("id", ""))
    elif name == "math500":
        question = str(row["problem"])
        answer = str(row["answer"]).strip()
        category = str(row.get("subject") or "") or None
        source_id = str(row.get("unique_id") or row.get("id") or "")
    elif name == "minerva_math":
        question = str(row["problem"])
        answer = extract_last_boxed_content(str(row["solution"]))
        category = str(row.get("type") or "") or None
        source_id = str(row.get("idx", ""))
    elif name == "olympiadbench":
        question = str(row["question"])
        final_answers = row["final_answer"]
        if not isinstance(final_answers, list) or not final_answers:
            raise ValueError("OlympiadBench final_answer must be a non-empty list")
        answer = ", ".join(str(value).strip() for value in final_answers)
        category = str(row.get("subfield") or "") or None
        source_id = str(row.get("id", ""))
    elif name in {"aime24", "aime25", "aime26", "hmmt26", "amc23"}:
        question = str(row["problem"])
        answer = format_scalar_answer(row["answer"])
        category = format_category(
            row.get("problem_type") or row.get("subject") or row.get("type")
        )
        source_id = str(row.get("problem_idx", row.get("id", "")))
    else:
        raise ValueError(f"unsupported evaluation benchmark: {name}")

    if not question.strip() or not answer.strip():
        raise ValueError(f"{name}: empty question or ground truth")
    return question, answer, category, source_id


def build_eval_input(
    benchmark_paths: Mapping[str, Path],
    names: Sequence[str],
    destination_root: Path,
    system_prompt: str,
) -> tuple[Path, dict[str, Any]]:
    """Create one combined GRPO-format prompt file so each model loads once."""
    benchmark_metadata = {
        name: {
            "path": str(benchmark_paths[name]),
            "sha256": file_sha256(benchmark_paths[name]),
            "records": SPECS[name].expected_rows,
        }
        for name in names
    }
    input_identity = {
        "benchmarks": list(names),
        "benchmark_metadata": benchmark_metadata,
        "system_prompt": system_prompt,
        # v2 makes every nested extra_info field schema-stable for Ray/Arrow.
        "schema_version": 2,
    }
    input_sha256 = json_sha256(input_identity)
    input_dir = destination_root / "_inputs" / input_sha256[:16]
    prompts_path = input_dir / "prompts.jsonl"
    manifest_path = input_dir / "manifest.json"
    expected_rows = sum(SPECS[name].expected_rows for name in names)

    if prompts_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_rows = sum(1 for _ in iter_jsonl(prompts_path))
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
            actual_rows = -1
        if (
            manifest.get("input_sha256") == input_sha256
            and actual_rows == expected_rows
        ):
            return prompts_path, manifest

    def prepared_rows() -> Iterator[dict[str, Any]]:
        group_id = 0
        for name in names:
            for benchmark_index, row in enumerate(iter_jsonl(benchmark_paths[name])):
                try:
                    question, answer, category, source_id = (
                        benchmark_question_and_answer(name, row)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{name} record {benchmark_index}: {exc}"
                    ) from exc
                extra_info: dict[str, Any] = {
                    "index": group_id,
                    "benchmark": name,
                    "benchmark_index": benchmark_index,
                    "source_id": source_id,
                    # Ray 2.44/PyArrow cannot concatenate nested structs when
                    # one block infers this field as null/missing and another
                    # infers it as string. Keep the key and type identical for
                    # every benchmark record.
                    "category": category or "",
                }
                yield {
                    "data_source": name,
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    "ability": "math",
                    "reward_model": {"ground_truth": answer, "style": "rule"},
                    "extra_info": extra_info,
                }
                group_id += 1

    written_rows = atomic_write_jsonl(prompts_path, prepared_rows())
    if written_rows != expected_rows:
        raise RuntimeError(
            f"prepared {written_rows} prompts, expected {expected_rows}"
        )
    manifest = {
        **input_identity,
        "input_sha256": input_sha256,
        "prompt_count": written_rows,
        "prompts_path": str(prompts_path),
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Prepared evaluation prompts: {written_rows} -> {prompts_path}")
    return prompts_path, manifest


def actor_checkpoint_world_size(actor_dir: Path) -> int | None:
    """Return FSDP shard world size only when every expected rank is present."""
    if not (actor_dir / "huggingface" / "config.json").is_file():
        return None
    world_sizes: set[int] = set()
    ranks: set[int] = set()
    for path in actor_dir.glob("model_world_size_*_rank_*.pt"):
        match = ACTOR_SHARD_RE.fullmatch(path.name)
        if match is None or path.stat().st_size == 0:
            continue
        world_sizes.add(int(match.group(1)))
        ranks.add(int(match.group(2)))
    if len(world_sizes) != 1:
        return None
    world_size = next(iter(world_sizes))
    if ranks != set(range(world_size)):
        return None
    return world_size


def discover_complete_steps(experiment_dir: Path) -> list[int]:
    if not experiment_dir.is_dir():
        return []
    steps: list[int] = []
    for step_dir in experiment_dir.iterdir():
        if not step_dir.is_dir():
            continue
        match = STEP_DIR_RE.fullmatch(step_dir.name)
        if match and actor_checkpoint_world_size(step_dir / "actor") is not None:
            steps.append(int(match.group(1)))
    return sorted(steps)


def resolve_experiment(
    checkpoint_root: Path,
    explicit_name: str | None,
    default_name: str,
    fallback_prefix: str,
) -> str:
    """Use the exact script-default experiment, or require an unambiguous match."""
    if explicit_name:
        candidate = Path(explicit_name).expanduser()
        if candidate.is_absolute():
            if candidate.parent.resolve() != checkpoint_root.resolve():
                raise ValueError(
                    "an absolute experiment path must be directly under its "
                    f"checkpoint root {checkpoint_root}"
                )
            name = candidate.name
        else:
            name = explicit_name
        if not (checkpoint_root / name).is_dir():
            raise FileNotFoundError(
                f"experiment directory does not exist: {checkpoint_root / name}"
            )
        return name

    if (checkpoint_root / default_name).is_dir():
        return default_name

    matches = sorted(
        path.name
        for path in checkpoint_root.glob(f"{fallback_prefix}*")
        if path.is_dir() and "qwen3-0.6b-base" in path.name
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no Qwen3-0.6B experiment found under {checkpoint_root}"
        )
    formatted = "\n  - ".join(matches)
    raise ValueError(
        f"multiple experiments found under {checkpoint_root}; pass the explicit "
        f"experiment option:\n  - {formatted}"
    )


def parse_requested_steps(value: str, available: Sequence[int]) -> list[int]:
    if not available:
        raise ValueError("no complete actor checkpoints are available")
    normalized = value.strip().lower()
    if normalized == "latest":
        return [max(available)]
    if normalized == "all":
        return list(available)
    try:
        requested = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as exc:
        raise ValueError("--steps must be latest, all, or comma-separated integers") from exc
    if not requested or any(step <= 0 for step in requested):
        raise ValueError("requested checkpoint steps must be positive")
    missing = [step for step in requested if step not in available]
    if missing:
        raise ValueError(
            f"requested incomplete/missing steps {missing}; available: {list(available)}"
        )
    return requested


def resolve_checkpoint_targets(args: argparse.Namespace) -> list[CheckpointTarget]:
    selections = [args.selection] if args.selection != "both" else ["svd", "random"]
    targets: list[CheckpointTarget] = []
    for selection in selections:
        if selection == "svd":
            root = args.svd_root.expanduser().resolve()
            experiment = resolve_experiment(
                root,
                args.svd_experiment,
                DEFAULT_SVD_EXPERIMENT,
                "svd_erank_qwen3_0_6b_",
            )
        else:
            root = args.random_root.expanduser().resolve()
            experiment = resolve_experiment(
                root,
                args.random_experiment,
                DEFAULT_RANDOM_EXPERIMENT,
                "random_qwen3_0_6b_",
            )
        available = discover_complete_steps(root / experiment)
        for step in parse_requested_steps(args.steps, available):
            targets.append(
                CheckpointTarget(
                    selection=selection,
                    checkpoint_root=root,
                    experiment=experiment,
                    step=step,
                )
            )
    return targets


def is_huggingface_model_dir(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    if not (path / "tokenizer_config.json").is_file():
        return False
    weight_markers = (
        list(path.glob("*.safetensors"))
        + list(path.glob("*.safetensors.index.json"))
        + list(path.glob("pytorch_model*.bin"))
    )
    return any(marker.is_file() and marker.stat().st_size > 0 for marker in weight_markers)


def ensure_merged_model(
    target: CheckpointTarget,
    *,
    env: Mapping[str, str],
    dry_run: bool,
) -> Path:
    """Reuse an existing HF model or merge complete VERL FSDP shards."""
    for name in ("merged_for_eval", "merged"):
        candidate = target.step_dir / name
        if is_huggingface_model_dir(candidate):
            print(f"[reuse]  {target.selection} step {target.step}: {candidate}")
            return candidate

    merge_target = target.step_dir / "merged_for_eval"
    if merge_target.exists() and not is_huggingface_model_dir(merge_target):
        raise RuntimeError(
            f"incomplete merge directory already exists: {merge_target}. "
            "Move it aside before retrying; it will not be overwritten automatically."
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "automated" / "merge_model.py"),
        "--experiment_name",
        target.experiment,
        "--step",
        str(target.step),
        "--output_model_name",
        target.experiment,
        "--backend",
        "fsdp",
        "--ckpt_root",
        str(target.checkpoint_root),
        "--dest_root",
        str(target.checkpoint_root / "merged_models"),
        "--target_dir",
        str(merge_target),
    ]
    print("[merge] ", " ".join(command))
    if dry_run:
        return merge_target
    subprocess.run(command, check=True, env=dict(env))
    if not is_huggingface_model_dir(merge_target):
        raise RuntimeError(f"merger completed without a usable HF model: {merge_target}")
    return merge_target


def response_extra_info(row: Mapping[str, Any]) -> dict[str, Any]:
    original_data = row.get("original_data")
    if not isinstance(original_data, dict):
        return {}
    original_entry = original_data.get("original_entry")
    if not isinstance(original_entry, dict):
        return {}
    extra_info = original_entry.get("extra_info")
    return extra_info if isinstance(extra_info, dict) else {}


def response_prediction(row: Mapping[str, Any]) -> str:
    reward_result = row.get("reward_result")
    if not isinstance(reward_result, Mapping):
        return "[INVALID]"
    prediction = reward_result.get("pred", "[INVALID]")
    return str(prediction)


def majority_vote_correct(group_rows: Sequence[dict[str, Any]]) -> bool:
    """Match VERL's maj@N: exact normalized-pred vote, first-seen tie break."""
    ordered = sorted(group_rows, key=lambda row: int(row.get("sample_id", -1)))
    vote_counts: dict[str, int] = {}
    for row in ordered:
        prediction = response_prediction(row)
        vote_counts[prediction] = vote_counts.get(prediction, 0) + 1
    winning_prediction = max(vote_counts, key=vote_counts.get)
    winning_row = next(
        row for row in ordered if response_prediction(row) == winning_prediction
    )
    return bool(winning_row.get("passed", False))


def calculate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("responses file is empty")
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("group_id")].append(row)

    sample_counts = {len(group_rows) for group_rows in groups.values()}
    if len(sample_counts) != 1:
        raise ValueError(
            f"inconsistent samples per problem: {sorted(sample_counts)}"
        )
    samples_per_problem = next(iter(sample_counts))

    correct_samples = sum(bool(row.get("passed", False)) for row in rows)
    pass_groups = sum(
        any(bool(row.get("passed", False)) for row in group_rows)
        for group_rows in groups.values()
    )
    majority_groups = sum(
        majority_vote_correct(group_rows) for group_rows in groups.values()
    )
    response_tokens = [
        int(row.get("response_token_count", 0))
        for row in rows
        if row.get("response_token_count") is not None
    ]
    return {
        "problem_count": len(groups),
        "sample_count": len(rows),
        "samples_per_problem": samples_per_problem,
        "correct_samples": correct_samples,
        "avg_at_n": correct_samples / len(rows),
        "p_at_n": pass_groups / len(groups),
        "maj_at_n": majority_groups / len(groups),
        # Backward-compatible aliases used by earlier summaries.
        "sample_accuracy": correct_samples / len(rows),
        "pass_at_n": pass_groups / len(groups),
        "invalid_samples": sum(
            bool((row.get("reward_result") or {}).get("invalid", False))
            for row in rows
        ),
        "verification_error_samples": sum(
            bool((row.get("reward_result") or {}).get("verification_error", False))
            for row in rows
        ),
        "hit_max_tokens_samples": sum(
            bool(row.get("generation_hit_max_tokens", False)) for row in rows
        ),
        "mean_response_tokens": mean(response_tokens) if response_tokens else None,
    }


def build_evaluation_summary(
    rows: list[dict[str, Any]],
    target: CheckpointTarget,
    model_path: Path,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        benchmark = str(row.get("data_source") or "unknown")
        by_benchmark[benchmark].append(row)
        category = response_extra_info(row).get("category")
        if category:
            by_category[benchmark][str(category)].append(row)

    benchmark_metrics = {
        name: calculate_metrics(benchmark_rows)
        for name, benchmark_rows in sorted(by_benchmark.items())
    }
    category_metrics = {
        name: {
            category: calculate_metrics(category_rows)
            for category, category_rows in sorted(categories.items())
        }
        for name, categories in sorted(by_category.items())
    }
    overall = calculate_metrics(rows)
    overall["macro_benchmark_avg_at_n"] = mean(
        metrics["avg_at_n"] for metrics in benchmark_metrics.values()
    )
    overall["macro_benchmark_p_at_n"] = mean(
        metrics["p_at_n"] for metrics in benchmark_metrics.values()
    )
    overall["macro_benchmark_maj_at_n"] = mean(
        metrics["maj_at_n"] for metrics in benchmark_metrics.values()
    )
    overall["macro_benchmark_pass_at_n"] = mean(
        metrics["pass_at_n"] for metrics in benchmark_metrics.values()
    )
    return {
        "status": "complete",
        "selection": target.selection,
        "experiment": target.experiment,
        "checkpoint_root": str(target.checkpoint_root),
        "checkpoint_step": target.step,
        "model_path": str(model_path),
        "run_config": run_config,
        "overall": overall,
        "benchmarks": benchmark_metrics,
        "categories": category_metrics,
    }


def build_subprocess_env(cuda_visible_devices: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONNOUSERSITE"] = "1"
    verl_path = str(REPO_ROOT / "verl")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{verl_path}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else verl_path
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    storage_root = Path(
        env.get("GRADALIGN_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    )
    env.setdefault("HF_HOME", str(storage_root / "cache" / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(Path(env["HF_HOME"]) / "datasets"))
    env.setdefault("XDG_CACHE_HOME", str(storage_root / "cache"))
    env.setdefault("TMPDIR", str(storage_root / "runtime" / "tmp"))
    env.setdefault("RAY_TMPDIR", str(storage_root / "r"))
    return env


def validate_gpu_layout(args: argparse.Namespace) -> None:
    devices = [
        device.strip()
        for device in args.cuda_visible_devices.split(",")
        if device.strip()
    ]
    required = args.concurrency * args.tensor_parallel_size * args.pipeline_parallel_size
    if not devices:
        raise ValueError("--cuda-visible-devices must contain at least one GPU")
    if required > len(devices):
        raise ValueError(
            "inference layout requires "
            f"{required} GPUs (concurrency*TP*PP), but only {len(devices)} "
            f"are visible: {devices}"
        )
    if args.max_tokens >= args.max_model_len:
        raise ValueError("--max-tokens must be smaller than --max-model-len")
    if args.n_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--n-samples and --batch-size must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")


def evaluation_output_dir(
    results_root: Path,
    target: CheckpointTarget,
    benchmark_names: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    suite = "-".join(benchmark_names)
    temperature = str(args.temperature).replace(".", "p")
    run_name = (
        f"{suite}__n{args.n_samples}_t{temperature}_"
        f"max{args.max_tokens}_ctx{args.max_model_len}"
    )
    return (
        results_root
        / target.selection
        / target.experiment
        / f"global_step_{target.step}"
        / run_name
    )


def sort_response_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (int(row.get("group_id", -1)), int(row.get("sample_id", -1))),
    )


def metric_value(metrics: Mapping[str, Any], name: str) -> float:
    fallback = {
        "avg_at_n": "sample_accuracy",
        "p_at_n": "pass_at_n",
        "maj_at_n": "sample_accuracy",
    }[name]
    return float(metrics.get(name, metrics[fallback]))


def render_leaderboard_at_8(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Render every default benchmark with equal-weight macro statistics."""
    eligible = [
        entry
        for entry in entries
        if int((entry.get("run_config") or {}).get("n_samples") or 0) == 8
        and all(
            name in (entry.get("benchmarks") or {})
            for name in LEADERBOARD_BENCHMARKS
        )
    ]

    first_header = [
        '<th rowspan="2">Checkpoint</th>',
        *(
            f'<th colspan="3">{html.escape(name.upper())}</th>'
            for name in LEADERBOARD_BENCHMARKS
        ),
        '<th colspan="3">统计值（全部 benchmark 宏平均）</th>',
    ]
    second_header = [
        *(
            cell
            for _ in (*LEADERBOARD_BENCHMARKS, "macro")
            for cell in ("<th>AVG@8</th>", "<th>P@8</th>", "<th>maj@8</th>")
        )
    ]
    markdown_lines = [
        "# Math benchmark leaderboard (@8)",
        "",
        "数值为百分比；统计值是全部 benchmark 的等权宏平均。",
        "",
        "<table>",
        "  <thead>",
        "    <tr>" + "".join(first_header) + "</tr>",
        "    <tr>" + "".join(second_header) + "</tr>",
        "  </thead>",
        "  <tbody>",
    ]

    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.writer(csv_buffer)
    csv_header = ["checkpoint"]
    for name in (*LEADERBOARD_BENCHMARKS, "macro"):
        csv_header.extend(
            (f"{name}_AVG@8", f"{name}_P@8", f"{name}_maj@8")
        )
    csv_writer.writerow(csv_header)

    for entry in eligible:
        label = (
            f"{entry.get('selection')} / global_step_"
            f"{entry.get('checkpoint_step')} / {entry.get('experiment')}"
        )
        benchmark_values: list[tuple[float, float, float]] = []
        for name in LEADERBOARD_BENCHMARKS:
            metrics = entry["benchmarks"][name]
            benchmark_values.append(
                (
                    metric_value(metrics, "avg_at_n"),
                    metric_value(metrics, "p_at_n"),
                    metric_value(metrics, "maj_at_n"),
                )
            )
        macro_values = tuple(
            mean(values[index] for values in benchmark_values)
            for index in range(3)
        )
        all_values = [value for values in benchmark_values for value in values]
        all_values.extend(macro_values)
        markdown_lines.append(
            "    <tr><td>"
            + html.escape(label)
            + "</td>"
            + "".join(f"<td>{value:.2%}</td>" for value in all_values)
            + "</tr>"
        )
        csv_writer.writerow([label, *(f"{100.0 * value:.6f}" for value in all_values)])

    markdown_lines.extend(("  </tbody>", "</table>", ""))
    if not eligible:
        markdown_lines.extend(
            (
                "尚无同时包含全部 benchmark 且 n_samples=8 的完整结果。",
                "",
            )
        )
    return "\n".join(markdown_lines), csv_buffer.getvalue()


def update_scoreboard(results_root: Path) -> dict[str, Path]:
    entries: list[dict[str, Any]] = []
    for summary_path in results_root.glob("*/*/global_step_*/*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "complete":
            continue
        entries.append(
            {
                "selection": summary.get("selection"),
                "experiment": summary.get("experiment"),
                "checkpoint_step": summary.get("checkpoint_step"),
                "run_config": summary.get("run_config"),
                "overall": summary.get("overall"),
                "benchmarks": summary.get("benchmarks"),
                "summary_path": str(summary_path),
            }
        )
    entries.sort(
        key=lambda entry: (
            str(entry.get("selection")),
            str(entry.get("experiment")),
            int(entry.get("checkpoint_step") or -1),
            str(entry.get("summary_path")),
        )
    )
    scoreboard_path = results_root / "scoreboard.json"
    leaderboard_path = results_root / "leaderboard_at_8.md"
    leaderboard_csv_path = results_root / "leaderboard_at_8.csv"
    atomic_write_json(
        scoreboard_path,
        {
            "runs": entries,
            "leaderboard_benchmarks": list(LEADERBOARD_BENCHMARKS),
        },
    )
    leaderboard_markdown, leaderboard_csv = render_leaderboard_at_8(entries)
    atomic_write_text(leaderboard_path, leaderboard_markdown)
    atomic_write_text(leaderboard_csv_path, leaderboard_csv)
    return {
        "json": scoreboard_path,
        "markdown": leaderboard_path,
        "csv": leaderboard_csv_path,
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print(
        f"\n{summary['selection']} step {summary['checkpoint_step']} "
        f"({summary['experiment']})"
    )
    for name, metrics in summary["benchmarks"].items():
        print(
            f"  {name:16s} AVG@{summary['run_config']['n_samples']}="
            f"{metrics.get('avg_at_n', metrics['sample_accuracy']):.2%} "
            f"P@{summary['run_config']['n_samples']}="
            f"{metrics.get('p_at_n', metrics['pass_at_n']):.2%} "
            f"maj@{summary['run_config']['n_samples']}="
            f"{metrics.get('maj_at_n', metrics['sample_accuracy']):.2%} "
            f"({metrics['problem_count']} problems)"
        )
    print(
        f"  macro AVG@{summary['run_config']['n_samples']}="
        f"{summary['overall'].get('macro_benchmark_avg_at_n', summary['overall']['sample_accuracy']):.2%} "
        f"P@{summary['run_config']['n_samples']}="
        f"{summary['overall'].get('macro_benchmark_p_at_n', summary['overall']['macro_benchmark_pass_at_n']):.2%} "
        f"maj@{summary['run_config']['n_samples']}="
        f"{summary['overall'].get('macro_benchmark_maj_at_n', summary['overall']['sample_accuracy']):.2%}"
    )


def run_evaluation_target(
    target: CheckpointTarget,
    benchmark_names: Sequence[str],
    prompts_path: Path,
    input_manifest: Mapping[str, Any],
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> dict[str, Any] | None:
    model_path = ensure_merged_model(target, env=env, dry_run=args.dry_run)
    output_dir = evaluation_output_dir(
        args.results_root.expanduser().resolve(), target, benchmark_names, args
    )
    response_path = output_dir / "responses.json"
    sorted_response_path = output_dir / "responses_sorted.json"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    reward_manifest_path = output_dir / "reward_scoring.json"

    reward_path = args.reward_path.expanduser().resolve()
    run_config = {
        "selection": target.selection,
        "experiment": target.experiment,
        "checkpoint_step": target.step,
        "model_path": str(model_path),
        "benchmarks": list(benchmark_names),
        "evaluation_input_sha256": input_manifest.get("input_sha256"),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reward_path": str(reward_path),
        "reward_sha256": file_sha256(reward_path),
        "cuda_visible_devices": args.cuda_visible_devices,
    }
    run_config_sha256 = json_sha256(run_config)
    run_config["run_config_sha256"] = run_config_sha256
    expected_samples = int(input_manifest["prompt_count"]) * args.n_samples

    if not args.overwrite and summary_path.is_file() and response_path.is_file():
        try:
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_config = previous_summary.get("run_config", {})
            with response_path.open("r", encoding="utf-8") as handle:
                previous_rows = json.load(handle)
        except (OSError, json.JSONDecodeError):
            previous_summary = {}
            previous_config = {}
            previous_rows = []
        if (
            previous_summary.get("status") == "complete"
            and previous_config.get("run_config_sha256") == run_config_sha256
            and isinstance(previous_rows, list)
            and len(previous_rows) == expected_samples
        ):
            print(
                f"[skip]   {target.selection} step {target.step}: "
                f"completed result at {summary_path}"
            )
            print_summary(previous_summary)
            return previous_summary

    command = [
        sys.executable,
        str(REPO_ROOT / "select" / "inference_ray_batch.py"),
        "--model_path",
        str(model_path),
        "--prompts_file",
        str(prompts_path),
        "--output_dir",
        str(output_dir),
        "--n_samples",
        str(args.n_samples),
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--pipeline_parallel_size",
        str(args.pipeline_parallel_size),
        "--concurrency",
        str(args.concurrency),
        "--batch_size",
        str(args.batch_size),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--reward_path",
        str(reward_path),
        "--reward_name",
        "compute_score",
        "--reward_manifest_path",
        str(reward_manifest_path),
    ]
    print(f"\n[evaluate] {target.selection} step {target.step}")
    print("Executing:", " ".join(command))
    if args.dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, {**run_config, "status": "running"})
    try:
        subprocess.run(command, check=True, env=dict(env))
        with response_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise TypeError(f"expected a JSON list of responses: {response_path}")
        if len(rows) != expected_samples:
            raise RuntimeError(
                f"expected {expected_samples} responses, found {len(rows)}"
            )
        rows = sort_response_rows(rows)
        atomic_write_json(sorted_response_path, rows)
        summary = build_evaluation_summary(rows, target, model_path, run_config)
        atomic_write_json(summary_path, summary)
        atomic_write_json(config_path, {**run_config, "status": "complete"})
    except BaseException as exc:
        atomic_write_json(
            config_path,
            {**run_config, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    print_summary(summary)
    print(f"  summary: {summary_path}")
    return summary


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--selection",
        choices=("svd", "random", "both"),
        default="both",
        help="which training-script checkpoints to inspect/evaluate (default: both)",
    )
    parser.add_argument(
        "--steps",
        default="latest",
        help="latest, all, or comma-separated global steps (default: latest)",
    )
    parser.add_argument("--svd-root", type=Path, default=DEFAULT_SVD_ROOT)
    parser.add_argument("--random-root", type=Path, default=DEFAULT_RANDOM_ROOT)
    parser.add_argument(
        "--svd-experiment",
        default=None,
        help="override the experiment directory produced by the SVD script",
    )
    parser.add_argument(
        "--random-experiment",
        default=None,
        help="override the experiment directory produced by the random script",
    )


def add_mirror_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help=(
            "Hugging Face endpoint used by direct files and datasets "
            f"(default: {DEFAULT_HF_ENDPOINT}; use https://alpha.hf-mirror.com "
            "as an alternative)"
        ),
    )
    parser.add_argument(
        "--github-mirror",
        default=DEFAULT_GITHUB_MIRROR,
        help=(
            "GitHub/raw mirror prefix "
            f"(default: {DEFAULT_GITHUB_MIRROR})"
        ),
    )


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "benchmarks",
        nargs="*",
        metavar="NAME",
        help="benchmarks to ensure; default: all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--repair",
        action="store_true",
        help="replace an existing benchmark only when it fails validation",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="print counts and sources without downloading",
    )
    add_mirror_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="download missing benchmark JSONL files"
    )
    add_download_arguments(download_parser)

    subparsers.add_parser("sources", help="print benchmark counts and sources")

    checkpoint_parser = subparsers.add_parser(
        "checkpoints", help="list complete FSDP checkpoints without evaluating"
    )
    add_checkpoint_arguments(checkpoint_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="merge and evaluate GradAlign checkpoints"
    )
    add_checkpoint_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_EVAL_BENCHMARKS),
        help=(
            "evaluation suite (default, all requested math benchmarks: "
            + " ".join(DEFAULT_EVAL_BENCHMARKS)
            + ")"
        ),
    )
    evaluate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    evaluate_parser.add_argument("--repair", action="store_true")
    evaluate_parser.add_argument("--timeout", type=float, default=60.0)
    evaluate_parser.add_argument("--reward-path", type=Path, default=DEFAULT_REWARD_PATH)
    evaluate_parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    evaluate_parser.add_argument("--n-samples", type=int, default=8)
    evaluate_parser.add_argument("--temperature", type=float, default=0.6)
    evaluate_parser.add_argument("--max-tokens", type=int, default=4096)
    evaluate_parser.add_argument("--max-model-len", type=int, default=5120)
    evaluate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    evaluate_parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    evaluate_parser.add_argument("--concurrency", type=int, default=4)
    evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    evaluate_parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help="comma-separated physical GPU IDs (default: current env or 4,5,6,7)",
    )
    evaluate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun an otherwise complete matching evaluation",
    )
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print checkpoint merge/inference commands without changing files",
    )
    add_mirror_arguments(evaluate_parser)
    return parser


def normalize_cli_argv(argv: Sequence[str]) -> list[str]:
    """Preserve the original downloader CLI while adding explicit subcommands."""
    if not argv:
        return ["download"]
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return list(argv)
    return ["download", *argv]


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(normalize_cli_argv(raw_argv))
    try:
        if args.command == "sources":
            print_sources()
            return 0

        if args.command in {"download", "evaluate"}:
            configure_download_environment(args)

        if args.command == "download":
            if args.timeout <= 0:
                raise ValueError("--timeout must be greater than zero")
            if args.show_sources:
                print_sources()
                return 0
            paths = ensure_benchmarks(
                args.benchmarks,
                output_dir=args.output_dir,
                repair=args.repair,
                timeout=args.timeout,
            )
            print("\nReady:")
            for name, path in paths.items():
                print(f"  {name}: {path}")
            return 0

        targets = resolve_checkpoint_targets(args)
        if args.command == "checkpoints":
            for target in targets:
                world_size = actor_checkpoint_world_size(target.actor_dir)
                merged = next(
                    (
                        target.step_dir / name
                        for name in ("merged_for_eval", "merged")
                        if is_huggingface_model_dir(target.step_dir / name)
                    ),
                    None,
                )
                print(
                    f"{target.selection:6s} step={target.step:<5d} "
                    f"world_size={world_size} merged={merged or '-'}\n"
                    f"       experiment={target.experiment}"
                )
            return 0

        if args.timeout <= 0:
            raise ValueError("--timeout must be greater than zero")
        validate_gpu_layout(args)
        reward_path = args.reward_path.expanduser().resolve()
        if not reward_path.is_file():
            raise FileNotFoundError(f"reward file does not exist: {reward_path}")
        benchmark_names = normalize_names(
            args.benchmarks, default=DEFAULT_EVAL_BENCHMARKS
        )
        if args.dry_run:
            print("Dry run: benchmark downloads and prompt generation are skipped.")
            prompts_path = Path("<prepared-evaluation-prompts>")
            input_manifest: Mapping[str, Any] = {
                "input_sha256": "<dry-run>",
                "prompt_count": sum(SPECS[name].expected_rows for name in benchmark_names),
            }
        else:
            benchmark_paths = ensure_benchmarks(
                benchmark_names,
                output_dir=args.output_dir,
                repair=args.repair,
                timeout=args.timeout,
                default=DEFAULT_EVAL_BENCHMARKS,
            )
            prompts_path, input_manifest = build_eval_input(
                benchmark_paths,
                benchmark_names,
                args.results_root.expanduser().resolve(),
                args.system_prompt,
            )

        env = build_subprocess_env(args.cuda_visible_devices)
        for target in targets:
            run_evaluation_target(
                target,
                benchmark_names,
                prompts_path,
                input_manifest,
                args,
                env,
            )
        if not args.dry_run:
            result_paths = update_scoreboard(args.results_root.expanduser().resolve())
            print(f"\nScoreboard JSON: {result_paths['json']}")
            print(f"Leaderboard @8: {result_paths['markdown']}")
            print(f"Leaderboard CSV: {result_paths['csv']}")
        return 0
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
