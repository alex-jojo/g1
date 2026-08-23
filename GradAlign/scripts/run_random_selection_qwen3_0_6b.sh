#!/usr/bin/env bash
set -euo pipefail

# Qwen3-0.6B: full-pool random selection + GRPO training.
# Training/model/reward defaults match run_svd_selection_qwen3_0_6b.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRADALIGN_STORAGE_ROOT="${GRADALIGN_STORAGE_ROOT:-/media/chenzhipeng/cll}"
PYTHON_BIN="${PYTHON_BIN:-${GRADALIGN_PYTHON:-/home/chenzhipeng/anaconda3/envs/cll/bin/python}}"
MODEL_KEY="${MODEL_KEY:-qwen3-0.6b-base}"
MODEL_PATH="${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-0.6B-Base}"
SOURCE_TRAIN_DIR="${SOURCE_TRAIN_DIR:-${GRADALIGN_STORAGE_ROOT}/data/gsm_math_dsr_test}"
SOURCE_TRAIN_PARQUET="${SOURCE_TRAIN_PARQUET:-train_qwen_qwen3-1.7b-base_n8_p1024_r1024_t1.0_seed1_rejudge_mathverify1_pass0.125_0.875_nontrivial.parquet}"
RANDOM_POOL_DIR="${RANDOM_POOL_DIR:-${GRADALIGN_STORAGE_ROOT}/runtime/random_pools/qwen3_0_6b_pass_0.125_0.875}"
RANDOM_POOL_JSONL="${RANDOM_POOL_DIR}/train.jsonl"
CKPT_ROOT="${CKPT_ROOT:-${GRADALIGN_STORAGE_ROOT}/runtime/checkpoints/random}"
REWARD_PATH="${REWARD_PATH:-${REPO_ROOT}/rewards/grpo_math_verify_reward.py}"

TRAIN_DATASET="${TRAIN_DATASET:-1to8}"
PREFIX="${PREFIX:-random_qwen3_0_6b_unified_reward_v1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS="${N_GPUS:-4}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
INFERENCE_PIPELINE_PARALLEL_SIZE="${INFERENCE_PIPELINE_PARALLEL_SIZE:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
STEPS_PER_SELECTION="${STEPS_PER_SELECTION:-10}"
CANDIDATES_PER_SELECTION="${CANDIDATES_PER_SELECTION:-2560}"
SELECTION_DENOMINATOR="${SELECTION_DENOMINATOR:-2}"
export WANDB_API_KEY="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"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5120}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
SEED="${SEED:-1}"

SOURCE_PARQUET_PATH="${SOURCE_TRAIN_DIR}/${SOURCE_TRAIN_PARQUET}"

if ! "${PYTHON_BIN}" -c 'import datasets, torch' >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is not the GradAlign training environment (torch/datasets missing)." >&2
  echo "Activate the correct Conda environment or set PYTHON_BIN=/path/to/env/bin/python." >&2
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_PARQUET_PATH}" ]]; then
  echo "Source training parquet does not exist: ${SOURCE_PARQUET_PATH}" >&2
  exit 2
fi
if [[ ! -f "${REWARD_PATH}" ]]; then
  echo "Reward function does not exist: ${REWARD_PATH}" >&2
  exit 2
fi
if (( TOTAL_STEPS % STEPS_PER_SELECTION != 0 )); then
  echo "TOTAL_STEPS must be divisible by STEPS_PER_SELECTION" >&2
  exit 2
fi
if (( CANDIDATES_PER_SELECTION % SELECTION_DENOMINATOR != 0 )); then
  echo "CANDIDATES_PER_SELECTION must be divisible by SELECTION_DENOMINATOR" >&2
  exit 2
fi

SELECTED_PER_ROUND=$((CANDIDATES_PER_SELECTION / SELECTION_DENOMINATOR))
EXPECTED_TRAIN_EXAMPLES=$((TRAIN_BATCH_SIZE * STEPS_PER_SELECTION))
if (( SELECTED_PER_ROUND != EXPECTED_TRAIN_EXAMPLES )); then
  echo "Selected prompts (${SELECTED_PER_ROUND}) must equal batch_size * steps (${EXPECTED_TRAIN_EXAMPLES})" >&2
  exit 2
fi
if (( PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "PPO_MINI_BATCH_SIZE must be divisible by PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
  exit 2
fi

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="${#CUDA_DEVICE_LIST[@]}"
if (( VISIBLE_GPU_COUNT != N_GPUS )); then
  echo "N_GPUS (${N_GPUS}) must match CUDA_VISIBLE_DEVICES count (${VISIBLE_GPU_COUNT})" >&2
  exit 2
fi
if (( INFERENCE_TENSOR_PARALLEL_SIZE <= 0 || INFERENCE_PIPELINE_PARALLEL_SIZE <= 0 )); then
  echo "Inference tensor/pipeline parallel sizes must be > 0" >&2
  exit 2
fi

GPUS_PER_INFERENCE_REPLICA=$((INFERENCE_TENSOR_PARALLEL_SIZE * INFERENCE_PIPELINE_PARALLEL_SIZE))
if (( N_GPUS % GPUS_PER_INFERENCE_REPLICA != 0 )); then
  echo "N_GPUS must be divisible by inference TP * PP" >&2
  exit 2
fi
MAX_INFERENCE_REPLICAS=$((N_GPUS / GPUS_PER_INFERENCE_REPLICA))
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-${MAX_INFERENCE_REPLICAS}}"
if (( INFERENCE_CONCURRENCY <= 0 || INFERENCE_CONCURRENCY > MAX_INFERENCE_REPLICAS )); then
  echo "INFERENCE_CONCURRENCY must be between 1 and ${MAX_INFERENCE_REPLICAS}" >&2
  exit 2
fi

NUM_SELECTIONS=$((TOTAL_STEPS / STEPS_PER_SELECTION))
HF_HOME="${HF_HOME:-${GRADALIGN_STORAGE_ROOT}/cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${GRADALIGN_STORAGE_ROOT}/cache}"
TMPDIR="${TMPDIR:-${GRADALIGN_STORAGE_ROOT}/runtime/tmp}"
RAY_TMPDIR="${RAY_TMPDIR:-${GRADALIGN_STORAGE_ROOT}/r}"
RAY_OBJECT_SPILL_ROOT="${RAY_OBJECT_SPILL_ROOT:-${GRADALIGN_STORAGE_ROOT}/runtime/ray_spill}"

export CUDA_VISIBLE_DEVICES
export GRADALIGN_ROOT="${REPO_ROOT}"
export GRADALIGN_STORAGE_ROOT
export GRADALIGN_PYTHON="${PYTHON_BIN}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/verl"
export HF_HOME
export HF_DATASETS_CACHE
export XDG_CACHE_HOME
export TMPDIR
export RAY_TMPDIR
export RAY_object_spilling_directory="${RAY_OBJECT_SPILL_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/automated/dynamic_selection.py"
  --prefix "${PREFIX}"
  --model "${MODEL_KEY}"
  --model_path "${MODEL_PATH}"
  --train_dataset "${TRAIN_DATASET}"
  --train_dir "${RANDOM_POOL_DIR}"
  --ckpt_root "${CKPT_ROOT}"
  --mode rand
  --chunk_size "${CANDIDATES_PER_SELECTION}"
  --k "${SELECTION_DENOMINATOR}"
  --num_selections "${NUM_SELECTIONS}"
  --iters_per_select "${STEPS_PER_SELECTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --inference_num_gpus "${N_GPUS}"
  --concurrency "${INFERENCE_CONCURRENCY}"
  --tensor_parallel_size "${INFERENCE_TENSOR_PARALLEL_SIZE}"
  --pipeline_parallel_size "${INFERENCE_PIPELINE_PARALLEL_SIZE}"
  --max_tokens "${MAX_RESPONSE_LENGTH}"
  --max_model_len "${MAX_MODEL_LEN}"
  --training_backend fsdp
  --merge_backend fsdp
  --n_gpus_per_node "${N_GPUS}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}"
  --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  --rollout_n "${ROLLOUT_N}"
  --ref_log_prob_micro_batch_size_per_gpu "${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  --rollout_log_prob_micro_batch_size_per_gpu "${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  --lr "${LEARNING_RATE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --save_freq "${SAVE_FREQ}"
  --test_freq "${TEST_FREQ}"
  --seed "${SEED}"
  --no-use-kl-loss
  --kl_loss_coef 0.0
  --reward_path "${REWARD_PATH}"
)

printf 'Random selection: full candidate pool -> %d prompts every %d GRPO steps\n' \
  "${SELECTED_PER_ROUND}" "${STEPS_PER_SELECTION}"
printf 'Training: total_steps=%d, selections=%d, batch=%d, rollout_n=%d, GPUs=%d\n' \
  "${TOTAL_STEPS}" "${NUM_SELECTIONS}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_N}" "${N_GPUS}"
printf 'Source candidate parquet: %s\n' "${SOURCE_PARQUET_PATH}"
printf 'Cached full-pool JSONL: %s\n' "${RANDOM_POOL_JSONL}"
printf 'Checkpoints: %s\n' "${CKPT_ROOT}"
printf 'Reward function: %s\n' "${REWARD_PATH}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${CKPT_ROOT}" \
  "${RANDOM_POOL_DIR}" \
  "${HF_DATASETS_CACHE}" \
  "${TMPDIR}" \
  "${RAY_TMPDIR}" \
  "${RAY_OBJECT_SPILL_ROOT}"

# select_data.py's original random path expects contiguous extra_info.index
# values in train.jsonl. Build that full-pool JSONL from the exact parquet used
# by the SVD experiment, while preserving the former ID as original_index.
if [[ ! -s "${RANDOM_POOL_JSONL}" || "${SOURCE_PARQUET_PATH}" -nt "${RANDOM_POOL_JSONL}" ]]; then
  RANDOM_POOL_TMP="${RANDOM_POOL_JSONL}.tmp.$$"
  "${PYTHON_BIN}" - "${SOURCE_PARQUET_PATH}" "${RANDOM_POOL_TMP}" <<'PY'
import os
import sys

from datasets import load_dataset

source_path, output_path = sys.argv[1:]
dataset = load_dataset("parquet", data_files=source_path, split="train")
if len(dataset) == 0:
    raise SystemExit(f"Source candidate parquet is empty: {source_path}")


def reindex(record, index):
    extra_info = dict(record.get("extra_info") or {})
    extra_info["original_index"] = extra_info.get("index", index)
    extra_info["index"] = index
    return {"extra_info": extra_info}


dataset = dataset.map(reindex, with_indices=True)
dataset.to_json(output_path, force_ascii=False)
print(f"Prepared full random pool: {len(dataset)} rows -> {output_path}")
PY
  mv "${RANDOM_POOL_TMP}" "${RANDOM_POOL_JSONL}"
fi

FULL_POOL_SIZE=$(wc -l < "${RANDOM_POOL_JSONL}")
if (( SELECTED_PER_ROUND > FULL_POOL_SIZE )); then
  echo "Cannot select ${SELECTED_PER_ROUND} prompts from a pool of ${FULL_POOL_SIZE}." >&2
  exit 2
fi
printf 'Full random pool size: %d prompts\n' "${FULL_POOL_SIZE}"

cd "${REPO_ROOT}"
exec "${CMD[@]}"
