#!/usr/bin/env bash
set -euo pipefail

# Qwen3-0.6B AdamW/backbone singular-subspace selection + GRPO training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export WANDB_API_KEY="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"
GRADALIGN_STORAGE_ROOT="${GRADALIGN_STORAGE_ROOT:-/media/chenzhipeng/cll}"
PYTHON_BIN="${PYTHON_BIN:-${GRADALIGN_PYTHON:-/home/chenzhipeng/anaconda3/envs/cll/bin/python}}"
MODEL_KEY="${MODEL_KEY:-qwen3-0.6b-base}"
MODEL_PATH="${MODEL_PATH:-/media/public/models/huggingface/Qwen/Qwen3-0.6B-Base}"
TRAIN_DIR="${TRAIN_DIR:-${GRADALIGN_STORAGE_ROOT}/data/gsm_math_dsr_test}"
CKPT_ROOT="${CKPT_ROOT:-${GRADALIGN_STORAGE_ROOT}/runtime/checkpoints/gradalign_svd}"
REWARD_PATH="${REWARD_PATH:-${REPO_ROOT}/rewards/grpo_math_verify_reward.py}"

TRAIN_DATASET="${TRAIN_DATASET:-1to8}"
TRAIN_PARQUET="${TRAIN_PARQUET:-train_qwen_qwen3-1.7b-base_n8_p1024_r1024_t1.0_seed1_rejudge_mathverify1_pass0.125_0.875_nontrivial.parquet}"
PREFIX="${PREFIX:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
N_GPUS="${N_GPUS:-4}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
INFERENCE_PIPELINE_PARALLEL_SIZE="${INFERENCE_PIPELINE_PARALLEL_SIZE:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-100}"
STEPS_PER_SELECTION="${STEPS_PER_SELECTION:-10}"
CANDIDATES_PER_SELECTION="${CANDIDATES_PER_SELECTION:-2560}"
# Subspace mode always keeps the highest-scoring 50% of the original pool.
SELECTION_DENOMINATOR=2

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

SVD_RANK="${SVD_RANK:-128}"
SVD_SCORE_SCOPE="${SVD_SCORE_SCOPE:-${SVD_PARAMETER_SCOPE:-transformer_2d}}"
SVD_GRADIENT_SOURCE="adamw"
# Choose the AdamW tensor used for subspace analysis. marginal_data removes the
# shared zero-gradient momentum baseline; full also includes weight decay.
ADAMW_UPDATE_TARGET="${ADAMW_UPDATE_TARGET:-marginal_data}"
ADAMW_GRAD_CLIP="${ADAMW_GRAD_CLIP:-1.0}"
SUBSPACE_SCORE_SIDE="${SUBSPACE_SCORE_SIDE:-mean}"
ANALYSIS_BACKEND="${ANALYSIS_BACKEND:-independent}"
ANALYSIS_MINIBATCH_SIZE="${ANALYSIS_MINIBATCH_SIZE:-1}"
ANALYSIS_PREPARE_WORKERS="${ANALYSIS_PREPARE_WORKERS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
SEED="${SEED:-1}"

if [[ -z "${PREFIX}" ]]; then
  PREFIX="subspace_qwen3_0_6b_adamw_${ADAMW_UPDATE_TARGET}_phi${SUBSPACE_SCORE_SIDE}_random_warmup${STEPS_PER_SELECTION}"
fi

if ! "${PYTHON_BIN}" -c 'import datasets, torch' >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is not the GradAlign training environment (torch/datasets missing)." >&2
  echo "Activate the correct Conda environment or set PYTHON_BIN=/path/to/env/bin/python." >&2
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_DIR}/${TRAIN_PARQUET}" ]]; then
  echo "Training parquet does not exist: ${TRAIN_DIR}/${TRAIN_PARQUET}" >&2
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
if [[ "${ANALYSIS_BACKEND}" == "independent" && "${ANALYSIS_MINIBATCH_SIZE}" != "1" ]]; then
  echo "Independent SVD requires ANALYSIS_MINIBATCH_SIZE=1" >&2
  exit 2
fi
if [[ "${SVD_SCORE_SCOPE}" != "qkvo_only" && "${SVD_SCORE_SCOPE}" != "ffn_only" && "${SVD_SCORE_SCOPE}" != "transformer_2d" ]]; then
  echo "SVD_SCORE_SCOPE must be qkvo_only, ffn_only, or transformer_2d" >&2
  exit 2
fi
if [[ "${ADAMW_UPDATE_TARGET}" != "actual_data" && "${ADAMW_UPDATE_TARGET}" != "marginal_data" && "${ADAMW_UPDATE_TARGET}" != "full" ]]; then
  echo "ADAMW_UPDATE_TARGET must be actual_data, marginal_data, or full" >&2
  exit 2
fi
if [[ "${SVD_GRADIENT_SOURCE}" == "adamw" && "${ANALYSIS_BACKEND}" != "independent" ]]; then
  echo "AdamW SVD requires ANALYSIS_BACKEND=independent" >&2
  exit 2
fi
if [[ "${SUBSPACE_SCORE_SIDE}" != "u" && "${SUBSPACE_SCORE_SIDE}" != "v" && "${SUBSPACE_SCORE_SIDE}" != "mean" ]]; then
  echo "SUBSPACE_SCORE_SIDE must be u, v, or mean" >&2
  exit 2
fi
if [[ "${SVD_SCORE_SCOPE}" != "qkvo_only" && "${ANALYSIS_BACKEND}" != "independent" ]]; then
  echo "Non-QKVO SVD_SCORE_SCOPE requires ANALYSIS_BACKEND=independent" >&2
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

mkdir -p \
  "${CKPT_ROOT}" \
  "${HF_DATASETS_CACHE}" \
  "${TMPDIR}" \
  "${RAY_TMPDIR}" \
  "${RAY_OBJECT_SPILL_ROOT}"

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
  --train_dir "${TRAIN_DIR}"
  --train_parquet "${TRAIN_PARQUET}"
  --ckpt_root "${CKPT_ROOT}"
  --mode subspace
  --chunk_size "${CANDIDATES_PER_SELECTION}"
  --k "${SELECTION_DENOMINATOR}"
  --num_selections "${NUM_SELECTIONS}"
  --iters_per_select "${STEPS_PER_SELECTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --n_samples_train "${ROLLOUT_N}"
  --minibatch_size "${ANALYSIS_MINIBATCH_SIZE}"
  --analysis_backend "${ANALYSIS_BACKEND}"
  --analysis_prepare_workers "${ANALYSIS_PREPARE_WORKERS}"
  --inference_num_gpus "${N_GPUS}"
  --concurrency "${INFERENCE_CONCURRENCY}"
  --tensor_parallel_size "${INFERENCE_TENSOR_PARALLEL_SIZE}"
  --pipeline_parallel_size "${INFERENCE_PIPELINE_PARALLEL_SIZE}"
  --max_tokens "${MAX_RESPONSE_LENGTH}"
  --max_model_len "${MAX_MODEL_LEN}"
  --analysis_num_gpus "${N_GPUS}"
  --svd_rank "${SVD_RANK}"
  --svd_score_scope "${SVD_SCORE_SCOPE}"
  --svd_gradient_source "${SVD_GRADIENT_SOURCE}"
  --adamw_update_target "${ADAMW_UPDATE_TARGET}"
  --adamw_grad_clip "${ADAMW_GRAD_CLIP}"
  --subspace_score_side "${SUBSPACE_SCORE_SIDE}"
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
  --inference_reward_path "${REWARD_PATH}"
  --reward_path "${REWARD_PATH}"
)

printf 'Subspace selection: %d candidates -> top %d prompts every %d GRPO steps\n' \
  "${CANDIDATES_PER_SELECTION}" "${SELECTED_PER_ROUND}" "${STEPS_PER_SELECTION}"
printf 'Training: total_steps=%d, selections=%d, batch=%d, rollout_n=%d, GPUs=%d\n' \
  "${TOTAL_STEPS}" "${NUM_SELECTIONS}" "${TRAIN_BATCH_SIZE}" "${ROLLOUT_N}" "${N_GPUS}"
printf 'Inference: replicas=%d, TP=%d, PP=%d, GPUs/replica=%d\n' \
  "${INFERENCE_CONCURRENCY}" "${INFERENCE_TENSOR_PARALLEL_SIZE}" \
  "${INFERENCE_PIPELINE_PARALLEL_SIZE}" "${GPUS_PER_INFERENCE_REPLICA}"
printf 'Analysis: backend=%s, source=%s, adamw_target=%s, score_scope=%s, recorded_scope=transformer_2d, workers=%d, micro_batch=%d, prepare_workers=%d\n' \
  "${ANALYSIS_BACKEND}" "${SVD_GRADIENT_SOURCE}" "${ADAMW_UPDATE_TARGET}" "${SVD_SCORE_SCOPE}" "${N_GPUS}" \
  "${ANALYSIS_MINIBATCH_SIZE}" "${ANALYSIS_PREPARE_WORKERS}"
printf 'Subspace score: phi_%s, descending; U0/V0 use the fixed initial backbone.\n' \
  "${SUBSPACE_SCORE_SIDE}"
printf 'Warm start: round 0 uses random data for %d GRPO steps; optimizer-aware selection starts at step %d.\n' \
  "${STEPS_PER_SELECTION}" "${STEPS_PER_SELECTION}"
printf 'Checkpoints: %s\n' "${CKPT_ROOT}"
printf 'Reward function: %s\n' "${REWARD_PATH}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"
