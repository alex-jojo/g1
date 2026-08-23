#!/usr/bin/env bash
set -euo pipefail

# Unified evaluation entry point for checkpoints produced by:
#   run_svd_selection_qwen3_0_6b.sh
#   run_random_selection_qwen3_0_6b.sh
#
# Examples:
#   bash scripts/run_math_benchmark_eval_qwen3_0_6b.sh
#   STEPS=all bash scripts/run_math_benchmark_eval_qwen3_0_6b.sh
#   SELECTION=svd STEPS=10,50,100 bash scripts/run_math_benchmark_eval_qwen3_0_6b.sh
#   HF_ENDPOINT=https://alpha.hf-mirror.com bash scripts/run_math_benchmark_eval_qwen3_0_6b.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenzhipeng/anaconda3/envs/cll/bin/python}"
GRADALIGN_STORAGE_ROOT="${GRADALIGN_STORAGE_ROOT:-/media/chenzhipeng/cll}"
GRADALIGN_BENCHMARK_ROOT="${GRADALIGN_BENCHMARK_ROOT:-${GRADALIGN_STORAGE_ROOT}/datasets/math_benchmarks}"
RESULTS_ROOT="${RESULTS_ROOT:-${GRADALIGN_STORAGE_ROOT}/runtime/evaluations/math_benchmarks}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GRADALIGN_GITHUB_MIRROR="${GRADALIGN_GITHUB_MIRROR:-https://ghfast.top}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

SELECTION="${SELECTION:-both}"
STEPS="${STEPS:-latest}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5120}"
CONCURRENCY="${CONCURRENCY:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

# Default: the original four benchmarks plus the five newly requested ones.
# Override with, for example:
#   BENCHMARKS="aime24 aime25 aime26 hmmt26 amc23 math500" bash ...
BENCHMARKS="${BENCHMARKS:-gsm8k math500 minerva_math olympiadbench aime24 aime25 aime26 hmmt26 amc23}"
read -r -a BENCHMARK_ARGS <<< "${BENCHMARKS}"

export GRADALIGN_STORAGE_ROOT
export GRADALIGN_BENCHMARK_ROOT
export HF_ENDPOINT
export GRADALIGN_GITHUB_MIRROR
export CUDA_VISIBLE_DEVICES

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/download_math_benchmarks.py" evaluate \
  --selection "${SELECTION}" \
  --steps "${STEPS}" \
  --benchmarks "${BENCHMARK_ARGS[@]}" \
  --output-dir "${GRADALIGN_BENCHMARK_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --n-samples "${N_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --concurrency "${CONCURRENCY}" \
  --batch-size "${BATCH_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --github-mirror "${GRADALIGN_GITHUB_MIRROR}" \
  "$@"
