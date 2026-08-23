#!/usr/bin/env python3

import os
import json
import argparse
import shutil
import subprocess
import sys
from typing import List, Tuple

import datasets as hfds  # type: ignore[import-not-found]

from config import get_dataset_dir, get_response_dir
from launch_verl_training import build_exp_name


def _load_train_dataset(train_dir: str, train_parquet: str):
    parquet_path = os.path.join(train_dir, train_parquet)
    jsonl_path = os.path.join(train_dir, "train.jsonl")
    if os.path.isfile(parquet_path):
        return hfds.load_dataset("parquet", data_files=parquet_path, split="train")
    if os.path.isfile(jsonl_path):
        records: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
        return hfds.Dataset.from_list(records)
    raise SystemExit(f"No train parquet or jsonl found in {train_dir}")


def write_jsonl(records: List[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def chunk_dataset(train_dir: str, exp_root: str, train_parquet: str, chunk_size: int, seed: int) -> Tuple[int, List[str]]:
    ds = _load_train_dataset(train_dir, train_parquet)
    total = len(ds)
    if total == 0:
        raise SystemExit("Empty training dataset; nothing to chunk")
    ds = ds.shuffle(seed=seed)

    if chunk_size <= 0:
        raise SystemExit("--chunk_size must be > 0")
    num_chunks = total // chunk_size
    if num_chunks <= 0:
        num_chunks = 1
        effective_chunk_size = total
    else:
        effective_chunk_size = chunk_size

    leftover = total - (num_chunks * effective_chunk_size)
    if leftover > 0:
        print(f"Info: dropping leftover {leftover} records to match floor(total/chunk_size)")

    chunk_dirs: List[str] = []
    for i in range(num_chunks):
        start = i * effective_chunk_size
        end = start + effective_chunk_size
        indices = list(range(start, min(end, total)))
        if not indices:
            continue
        chunk = ds.select(indices)
        chunk_dir = os.path.join(exp_root, f"chunk_{i}")
        os.makedirs(chunk_dir, exist_ok=True)
        out_parquet = os.path.join(chunk_dir, "train.parquet")
        chunk.to_parquet(out_parquet)
        # Also write JSONL for compatibility with inference scripts
        records = chunk.to_list()
        out_jsonl = os.path.join(chunk_dir, "train.jsonl")
        write_jsonl(records, out_jsonl)
        print(f"Wrote chunk {i}: {len(chunk)} rows -> {out_parquet} and {out_jsonl}")
        chunk_dirs.append(chunk_dir)

    # Write/update manifest
    manifest = {
        "experiment_name": os.path.basename(exp_root.rstrip("/")),
        "train_dir": train_dir,
        "total_rows": total,
        "chunk_size": effective_chunk_size,
        "num_chunks": num_chunks,
        "dropped_rows": leftover if leftover > 0 else 0,
        "seed": seed,
        "train_parquet": train_parquet,
    }
    with open(os.path.join(exp_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return num_chunks, chunk_dirs


def _run_stage(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run one stage synchronously without sending signals to any process."""
    return subprocess.run(cmd, check=True, **kwargs)


def _configure_wandb(exp_ckpt_root: str, project_name: str, trainer_logger: str) -> None:
    """Persist one W&B run ID for all sequential training launches."""
    loggers = [logger.strip() for logger in trainer_logger.split(",") if logger.strip()]
    if "wandb" not in loggers:
        return

    wandb_dir = os.path.join(exp_ckpt_root, "wandb")
    run_id_path = os.path.join(exp_ckpt_root, "wandb_run_id")
    os.makedirs(wandb_dir, exist_ok=True)

    run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if not run_id and os.path.isfile(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as handle:
            run_id = handle.read().strip()
    if not run_id:
        import wandb

        run_id = wandb.util.generate_id()
    if not run_id:
        raise SystemExit("Failed to create a non-empty W&B run ID")

    with open(run_id_path, "w", encoding="utf-8") as handle:
        handle.write(run_id + "\n")

    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_DIR", wandb_dir)
    print(
        "W&B continuous run: "
        f"project={project_name}, run_id={run_id}, run_id_file={run_id_path}"
    )


def _load_json_object(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_jsonl_object(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _response_cache_status(path: str) -> Tuple[int, bool]:
    """Return response count and whether every row has original token IDs."""
    if not os.path.isfile(path):
        return 0, False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0, False
    if not isinstance(rows, list):
        return 0, False
    token_compatible = all(
        isinstance(row, dict)
        and row.get("token_schema_version") == 1
        and row.get("token_ids_encoding") == "zlib+base64+uint32-le"
        and ("prompt_token_ids_b64" in row or "prompt_token_ids" in row)
        and ("response_token_ids_b64" in row or "response_token_ids" in row)
        and isinstance(row.get("prompt_token_count"), int)
        and isinstance(row.get("response_token_count"), int)
        and isinstance(row.get("requested_max_tokens"), int)
        for row in rows
    )
    return len(rows), token_compatible


def _independent_analysis_is_compatible(
    analysis_path: str,
    responses_dir: str,
    model_path: str,
    world_size: int,
    rollout_n: int,
    svd_rank: int,
    svd_score_scope: str,
    max_length: int,
    expected_groups: int,
) -> bool:
    """Prevent a complete legacy/stale JSONL from bypassing independent SVD."""
    manifest = _load_json_object(
        os.path.join(responses_dir, "data_independent", "partition_manifest.json")
    )
    analysis = manifest.get("analysis")
    signature = manifest.get("analysis_signature")
    if not isinstance(analysis, dict) or not isinstance(signature, str):
        return False
    expected = {
        "record_schema_version": 5,
        "analysis_backend": "independent",
        "advantage_estimator": "grpo",
        "norm_adv_by_std_in_grpo": True,
        "advantage_epsilon": 1e-6,
        "advantage_std_correction": 1,
        "token_input_source": "inference_token_ids",
        "world_size": world_size,
        "rollout_n": rollout_n,
        "analysis_minibatch_size": 1,
        "gradient_parameter_scope": "transformer_2d",
        "svd_score_scope": svd_score_scope,
        "svd_rank": svd_rank,
        "svd_q": svd_rank + 8,
        "svd_niter": 2,
        "svd_seed": 0,
        "max_length": max_length,
        "expected_groups": expected_groups,
        "expected_responses": expected_groups * rollout_n,
        "model_path": os.path.abspath(os.path.expanduser(model_path)),
    }
    if any(analysis.get(key) != value for key, value in expected.items()):
        return False
    first_row = _first_jsonl_object(analysis_path)
    group_stats_path = os.path.join(responses_dir, "group_stats.jsonl")
    try:
        with open(group_stats_path, "r", encoding="utf-8") as handle:
            group_stats_rows = sum(1 for line in handle if line.strip())
    except OSError:
        return False
    first_group_stats = _first_jsonl_object(group_stats_path)
    return (
        first_row.get("analysis_backend") == "independent"
        and first_row.get("analysis_signature") == signature
        and first_row.get("record_schema_version") == 5
        and first_row.get("gradient_parameter_scope") == "transformer_2d"
        and first_row.get("svd_score_scope") == svd_score_scope
        and group_stats_rows == expected_groups
        and first_group_stats.get("analysis_signature") == signature
    )


def _set_random_seed(seed: int) -> None:
    # os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    # try:
    #     import torch as _torch  # type: ignore
    #     _torch.manual_seed(seed)
    #     if _torch.cuda.is_available():
    #         _torch.cuda.manual_seed_all(seed)
    #     if hasattr(_torch.backends, "cudnn"):
    #         _torch.backends.cudnn.deterministic = True
    #         _torch.backends.cudnn.benchmark = False
    # except Exception:
    #     pass


def main():
    parser = argparse.ArgumentParser(description="Dynamic selection orchestrator: chunk → infer → analyze → aggregate → select → train")
    parser.add_argument("--prefix", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--model_path", default=None, type=str,
                        help="Explicit base model path (avoids relying on config.MODELS)")
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--val_dataset", default=None, type=str,
                        help="Optional for SVD mode; required by selection modes that use validation gradients")

    # Optional explicit overrides (to perfectly match experiment naming)
    parser.add_argument("--train_dir", default=None, type=str,
                        help="Override directory containing the training parquet/jsonl")
    parser.add_argument("--val_dir", default=None, type=str,
                        help="Override directory containing the validation parquet/jsonl")
    parser.add_argument("--train_parquet", default="train.parquet", type=str)

    # Chunking controls
    parser.add_argument("--chunk_size", default=2560, type=int,
                        help="Candidate prompts scored per selection round (default: 2560)")
    parser.add_argument("--k", default=2, type=int,
                        help="Selection denominator: keep chunk_size / k prompts (default: 2, i.e. top 50%%)")
    parser.add_argument("--seed", default=42, type=int)

    # Selection controls
    parser.add_argument("--mode", choices=["sim", "svd", "simacc", "acc", "rand", "accgreedy", 'align', 'negsim', 'norm', 'dot'], required=True)
    parser.add_argument("--acc_low", type=float, default=0.2)
    parser.add_argument("--acc_high", type=float, default=0.8)
    parser.add_argument("--epochs_per_select", type=int, default=1,
                        help="Train epochs per selection iteration (we run 1 epoch per launch; this controls total iterations with total_epochs)")
    parser.add_argument("--num_selections", type=int, required=True,
                        help="Number of selection iterations to run")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="GRPO train batch size (default: 128)")
    parser.add_argument("--iters_per_select", type=int, default=10,
                        help="Number of GRPO global steps between selection rounds (default: 10)")

    # GRPO training controls forwarded to launch_verl_training.py
    parser.add_argument("--training_backend", choices=["megatron", "fsdp"], default="megatron")
    parser.add_argument(
        "--project_name", default=os.environ.get("WANDB_PROJECT", "GradAlign-SVD"), type=str
    )
    parser.add_argument(
        "--trainer_logger", default="console,tensorboard,wandb", type=str,
        help="Comma-separated VERL logging backends forwarded to the trainer",
    )
    parser.add_argument("--n_gpus_per_node", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=32)
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--rollout_n", type=int, default=32)
    parser.add_argument("--ref_log_prob_micro_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--rollout_log_prob_micro_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--test_freq", type=int, default=5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--kl_loss_coef", type=float, default=0.001)
    kl_group = parser.add_mutually_exclusive_group()
    kl_group.add_argument("--use-kl-loss", dest="use_kl_loss", action="store_true")
    kl_group.add_argument("--no-use-kl-loss", dest="use_kl_loss", action="store_false")
    parser.set_defaults(use_kl_loss=True)
    parser.add_argument("--training_config_path", default=None, type=str)
    parser.add_argument("--training_config_name", default=None, type=str)

    # Inference controls
    parser.add_argument("--minibatch_size", type=int, default=8)
    parser.add_argument("--n_samples_train", type=int, default=32)
    parser.add_argument("--n_samples_val", type=int, default=128)
    parser.add_argument("--min_problems", type=int, default=1)
    parser.add_argument("--max_problems", type=int, default=40000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--inference_num_gpus", type=int, default=None,
                        help="GPU budget for local vLLM inference; defaults to n_gpus_per_node")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of vLLM replicas; defaults to inference_num_gpus / (TP * PP)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--train_infer_batch_size", type=int, default=512)
    parser.add_argument("--val_infer_batch_size", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--analysis_num_gpus", type=int, default=4,
                        help="Number of GPUs used by distributed gradient/SVD analysis")
    parser.add_argument(
        "--analysis_backend",
        choices=["fsdp", "independent"],
        default="fsdp",
        help="Gradient-analysis backend; independent is currently SVD-only",
    )
    parser.add_argument("--analysis_prepare_workers", type=int, default=32)
    parser.add_argument("--svd_rank", type=int, default=128,
                        help="Number of leading singular values used by SVD scoring")
    parser.add_argument(
        "--svd_score_scope",
        "--svd_parameter_scope",
        dest="svd_score_scope",
        choices=["qkvo_only", "ffn_only", "transformer_2d"],
        default=None,
        help="Matrix families included in S; independent still records all seven",
    )
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--reward_manager", type=str, default=None)
    parser.add_argument("--reward_path", type=str, default=None)
    parser.add_argument("--reward_name", type=str, default="compute_score")
    parser.add_argument(
        "--inference_reward_path",
        type=str,
        default=None,
        help="Custom reward used to label generated responses before gradient analysis",
    )
    
    # Checkpoint/merge controls
    parser.add_argument("--ckpt_root", type=str, default="",
                        help="Root directory where VERL checkpoints are saved")
    parser.add_argument("--merge_backend", type=str, default="megatron")
    parser.add_argument("--verl_val_set", type=str, default=None,
                        help="Optional VERL validation dataset; omit to train without validation")

    args = parser.parse_args()
    _set_random_seed(args.seed)

    if args.chunk_size <= 0:
        parser.error("--chunk_size must be > 0")
    if args.k <= 0:
        parser.error("--k must be > 0")
    if args.chunk_size % args.k != 0:
        parser.error("--chunk_size must be divisible by --k so the selection size is exact")
    if args.train_batch_size <= 0:
        parser.error("--train_batch_size must be > 0")
    if args.iters_per_select <= 0:
        parser.error("--iters_per_select must be > 0")
    if args.ppo_mini_batch_size <= 0 or args.ppo_micro_batch_size_per_gpu <= 0:
        parser.error("PPO mini/micro batch sizes must be > 0")
    if args.ppo_mini_batch_size % args.ppo_micro_batch_size_per_gpu != 0:
        parser.error("--ppo_mini_batch_size must be divisible by --ppo_micro_batch_size_per_gpu")
    if args.n_gpus_per_node <= 0:
        parser.error("--n_gpus_per_node must be > 0")
    if args.analysis_num_gpus <= 0 or args.analysis_prepare_workers <= 0:
        parser.error("--analysis_num_gpus and --analysis_prepare_workers must be > 0")
    if args.analysis_backend == "independent" and args.mode != "svd":
        parser.error("--analysis_backend independent is only supported with --mode svd")
    if args.analysis_backend == "independent" and args.minibatch_size != 1:
        parser.error("Independent SVD requires --minibatch_size 1")
    if args.svd_score_scope is None:
        args.svd_score_scope = (
            "transformer_2d"
            if args.analysis_backend == "independent"
            else "qkvo_only"
        )
    if args.analysis_backend != "independent" and args.svd_score_scope != "qkvo_only":
        parser.error("Non-QKVO SVD score scopes require --analysis_backend independent")
    if args.tensor_parallel_size <= 0 or args.pipeline_parallel_size <= 0:
        parser.error("--tensor_parallel_size and --pipeline_parallel_size must be > 0")

    inference_num_gpus = (
        args.inference_num_gpus
        if args.inference_num_gpus is not None
        else args.n_gpus_per_node
    )
    if inference_num_gpus <= 0:
        parser.error("--inference_num_gpus must be > 0")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        visible_gpu_count = len([
            device for device in visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ])
        if inference_num_gpus > visible_gpu_count:
            parser.error(
                f"--inference_num_gpus={inference_num_gpus} exceeds the "
                f"{visible_gpu_count} devices in CUDA_VISIBLE_DEVICES"
            )

    gpus_per_inference_replica = (
        args.tensor_parallel_size * args.pipeline_parallel_size
    )
    max_inference_replicas = inference_num_gpus // gpus_per_inference_replica
    if max_inference_replicas <= 0:
        parser.error(
            "Inference TP * PP exceeds the available inference GPU budget"
        )
    if args.concurrency is None:
        args.concurrency = max_inference_replicas
    elif args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    elif args.concurrency > max_inference_replicas:
        parser.error(
            f"--concurrency={args.concurrency} exceeds capacity "
            f"{max_inference_replicas} for {inference_num_gpus} GPUs with "
            f"TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}"
        )

    print(
        "Inference topology: "
        f"{args.concurrency} replica(s), TP={args.tensor_parallel_size}, "
        f"PP={args.pipeline_parallel_size}, GPU budget={inference_num_gpus}."
    )
    requires_validation_gradient = args.mode in {"sim", "simacc", "negsim", "dot"}
    if requires_validation_gradient and not args.val_dataset:
        parser.error(f"--mode {args.mode} requires --val_dataset")

    select_n = args.chunk_size // args.k
    train_examples_per_round = args.train_batch_size * args.iters_per_select
    if train_examples_per_round % select_n != 0:
        parser.error(
            "train_batch_size * iters_per_select must be divisible by the selected "
            "prompt count (chunk_size / k) so it maps to a whole number of epochs"
        )
    train_epochs = train_examples_per_round // select_n
    print(
        "Selection schedule: score "
        f"{args.chunk_size} prompts, keep top {select_n} "
        f"({100.0 / args.k:.1f}%), then train {args.iters_per_select} global steps "
        f"({train_epochs} epoch(s) at batch size {args.train_batch_size})."
    )

    # Resolve dataset dirs when not explicitly provided
    train_dir = args.train_dir or get_dataset_dir(args.train_dataset, args.model)
    val_dir = args.val_dir or (get_dataset_dir(args.val_dataset, args.model) if args.val_dataset else None)

    # Use a fixed experiment name (prefix + model + datasets) to align with VERL checkpoints
    exp_name = build_exp_name(args.prefix, args.model, args.train_dataset, train_dir, val_dir)
    exp_root = os.path.join(train_dir, exp_name)
    os.makedirs(exp_root, exist_ok=True)

    # Prepare experiment checkpoint root and chunk storage under it
    # ckpt_root/{exp_name}/chunks/chunk_{i}
    exp_ckpt_root = os.path.join(args.ckpt_root, exp_name)
    os.makedirs(exp_ckpt_root, exist_ok=True)
    _configure_wandb(exp_ckpt_root, args.project_name, args.trainer_logger)
    chunks_root = os.path.join(exp_ckpt_root, "chunks")
    os.makedirs(chunks_root, exist_ok=True)

    # 1) Chunk (idempotent with auto-resume): if chunks already exist, skip splitting
    existing_chunks = [d for d in os.listdir(chunks_root) if d.startswith("chunk_") and os.path.isdir(os.path.join(chunks_root, d))]
    if existing_chunks:
        existing_chunks.sort(key=lambda name: int(name.split("_")[-1]))
        num_chunks = len(existing_chunks)
        chunk_dirs = [os.path.join(chunks_root, d) for d in existing_chunks]
        manifest_path = os.path.join(chunks_root, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_chunk_size = int(manifest.get("chunk_size", -1))
            manifest_total_rows = int(manifest.get("total_rows", -1))
            expected_chunk_size = min(args.chunk_size, manifest_total_rows) if manifest_total_rows > 0 else args.chunk_size
            if manifest_chunk_size != expected_chunk_size:
                raise SystemExit(
                    f"Existing chunks under {chunks_root} contain {manifest_chunk_size} prompts each, "
                    f"but --chunk_size requests {args.chunk_size}. Use a new --prefix/--ckpt_root, "
                    "or intentionally remove the old chunks before restarting."
                )
        print(f"Found existing {num_chunks} chunks under {chunks_root}; skipping chunking")
    else:
        num_chunks, chunk_dirs = chunk_dataset(train_dir, chunks_root, args.train_parquet, args.chunk_size, args.seed)
        print(f"Manifest written at {os.path.join(chunks_root, 'manifest.json')}")

    # Precompute chunk sizes (dataset rows per chunk) for resume logic
    def _count_lines(path: str) -> int:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    chunk_sizes = []
    for d in chunk_dirs:
        chunk_sizes.append(_count_lines(os.path.join(d, "train.jsonl")))

    # Auto-resume: find earliest incomplete iteration
    i_start = 0
    val_jsonl = os.path.join(val_dir, "train.jsonl") if val_dir else None
    val_lines = _count_lines(val_jsonl) if val_jsonl and os.path.isfile(val_jsonl) else 0
    for i_probe in range(1, args.num_selections + 10):
        step_root_probe = os.path.join(exp_ckpt_root, f"global_step_{i_probe * args.iters_per_select}")
        if os.path.isdir(step_root_probe):
            i_start = i_probe
            # break

    if i_start > 0:
        print(f"Auto-resume: resuming from iteration {i_start}")

    for i in range(i_start, args.num_selections):
        chunk_idx = i % num_chunks
        chunk_dir = os.path.join(chunks_root, f"chunk_{chunk_idx}")
        prompts_file = os.path.join(chunk_dir, "train.jsonl")
        # Fixed global step directory naming to align with VERL
        gstep = i * args.iters_per_select
        step_root = os.path.join(exp_ckpt_root, f"global_step_{gstep}")
        train_parts_root = os.path.join(step_root, "train_split")
        val_dir_override = os.path.join(step_root, "val_responses") if requires_validation_gradient else None
        train_resp_dir = os.path.join(train_parts_root, "part_0")
        os.makedirs(train_resp_dir, exist_ok=True)
        reward_manifest_path = os.path.join(train_resp_dir, "reward_scoring.json")
        previous_reward_manifest = (
            _load_json_object(reward_manifest_path)
            if args.inference_reward_path
            else {}
        )
        force_analysis_recompute = False
        if val_dir_override:
            os.makedirs(val_dir_override, exist_ok=True)

        # 2) If i>0: merge previous iteration checkpoint and override model path
        merged_model_path = None
        if i > 0:
            # Expect VERL to have produced checkpoints under ckpt_root/{prefix}_sel{iter-1}_{model}_{train_dataset}_{train_name}_{val_name}/global_step_{i}
            # We'll merge that step to dest_root and use it for inference/analysis
            # Use fixed experiment name rather than per-iter suffix
            exp_prev = exp_name
            # Write merged model directly into exp ckpt dir at this global step
            merged_target = os.path.join(exp_ckpt_root, f"global_step_{gstep}", "merged")
            os.makedirs(merged_target, exist_ok=True)
            merge_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "merge_model.py"),
                "--experiment_name", exp_prev,
                "--step", str(i * args.iters_per_select),
                "--output_model_name", f"{exp_prev}",
                "--backend", args.merge_backend,
                "--ckpt_root", args.ckpt_root,
                "--dest_root", os.path.join(args.ckpt_root, "merged_models"),
                "--target_dir", merged_target,
            ]
            print("Executing:", " ".join(merge_cmd))
            _run_stage(merge_cmd)
            merged_model_path = merged_target
            if args.use_optimizer:
                optimizer_state_path = os.path.join(step_root, "opt_converted")
                convert_cmd = [
                    sys.executable, os.path.join(os.path.dirname(__file__), "convert_megatron_optimizer_to_hf.py"),
                    "--checkpoint_path", os.path.join(step_root, "actor"),
                    "--hf-config", merged_model_path,
                    "--output", optimizer_state_path,
                ]
                print("Executing:", " ".join(convert_cmd))
                _run_stage(convert_cmd)
                if not os.path.isfile(optimizer_state_path):
                    raise SystemExit(f"Optimizer state not found: {optimizer_state_path}")

        current_model_path = merged_model_path or args.model_path

        # 3) Local training inference → responses.json in step-root train_split/part_0
        infer_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "run_inference_local.py"),
            "--model", args.model,
            "--resp_dataset", args.train_dataset,
            "--prompts_file", prompts_file,
            "--output_part", "0",
            "--parts_root", train_parts_root,
            *( ["--model_path", current_model_path] if current_model_path else [] ),
            "--n_samples", str(args.n_samples_train),
            "--min_problems", str(args.min_problems),
            "--max_problems", str(args.max_problems),
            "--temperature", str(args.temperature),
            "--max_tokens", str(args.max_tokens),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--pipeline_parallel_size", str(args.pipeline_parallel_size),
            "--batch_size", str(args.train_infer_batch_size),
            "--max_model_len", str(args.max_model_len),
            "--concurrency", str(args.concurrency),
        ]
        if args.inference_reward_path:
            infer_cmd.extend([
                "--reward_path", args.inference_reward_path,
                "--reward_name", args.reward_name,
                "--reward_manifest_path", reward_manifest_path,
            ])

        # Skip train inference if responses already complete
        expected_train = chunk_sizes[chunk_idx] * args.n_samples_train
        train_part_dir = os.path.join(train_parts_root, "part_0")
        resp_json = os.path.join(train_part_dir, "responses.json")
        resp_sorted = os.path.join(train_part_dir, "responses_sorted.json")
        actual_train = 0
        train_tokens_available = False
        if os.path.isfile(resp_json):
            actual_train, train_tokens_available = _response_cache_status(resp_json)
        elif os.path.isfile(resp_sorted):
            actual_train, train_tokens_available = _response_cache_status(resp_sorted)
        if args.mode == "rand":
            print(
                f"Skip train inference for step {i}: "
                "rand selection does not use rollout responses"
            )
        elif (
            actual_train >= expected_train
            and expected_train > 0
            and train_tokens_available
        ):
            print(f"Skip train inference for step {i}: found {actual_train}/{expected_train} responses")
        else:
            if actual_train >= expected_train and not train_tokens_available:
                print(
                    "Cached train responses are text-only; regenerating them to "
                    "capture the original inference token IDs"
                )
            print("Executing:", " ".join(infer_cmd))
            _run_stage(infer_cmd)
            force_analysis_recompute = True

        # Use the same custom reward function as VERL training. Keeping this as
        # a separate stage lets a complete rollout cache be rescored without
        # running vLLM again.
        if args.inference_reward_path and args.mode != "rand":
            scoring_input = resp_json if os.path.isfile(resp_json) else resp_sorted
            score_cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "score_responses.py"),
                "--responses_file", scoring_input,
                "--reward_path", args.inference_reward_path,
                "--reward_name", args.reward_name,
                "--manifest_path", reward_manifest_path,
            ]
            print("Executing:", " ".join(score_cmd))
            _run_stage(score_cmd)

            current_reward_manifest = _load_json_object(reward_manifest_path)
            manifest_keys = (
                "response_count",
                "response_content_sha256",
                "reward_path",
                "reward_name",
                "reward_sha256",
            )
            force_analysis_recompute = force_analysis_recompute or any(
                previous_reward_manifest.get(key) != current_reward_manifest.get(key)
                for key in manifest_keys
            )
            response_content_changed = (
                previous_reward_manifest.get("response_content_sha256")
                != current_reward_manifest.get("response_content_sha256")
            )

            # Gradient analysis reads responses_sorted.json, so rebuild it after
            # adding the numeric reward fields to responses.json.
            if scoring_input == resp_json:
                sort_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "sort_responses.py"),
                    "--model_name", args.model,
                    "--dataset", args.train_dataset,
                    "--responses_dir", train_part_dir,
                ]
                print("Executing:", " ".join(sort_cmd))
                _run_stage(sort_cmd)

            # Prepared shards contain prompt/response token IDs. Reward-only
            # changes can reuse them, while changed response content cannot.
            prepared_data_dir = os.path.join(train_part_dir, "data")
            if response_content_changed and os.path.isdir(prepared_data_dir):
                print(f"Removing stale prepared response shards: {prepared_data_dir}")
                shutil.rmtree(prepared_data_dir)

        # 3b) Local validation inference → responses.json in step-root val_responses (flat)
        if args.mode in ['sim', 'simacc', 'negsim', 'dot']:
            val_prompts_file = os.path.join(val_dir, "train.jsonl")
            if not os.path.isfile(val_prompts_file):
                raise SystemExit(f"Validation prompts not found: {val_prompts_file}")
            infer_val_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_inference_local.py"),
                "--model", args.model,
                "--resp_dataset", args.val_dataset,
                "--prompts_file", val_prompts_file,
                "--output_part", "0",
                "--output_dir", val_dir_override,
                *( ["--model_path", current_model_path] if current_model_path else [] ),
                "--n_samples", str(args.n_samples_val),
                "--min_problems", str(args.min_problems),
                "--max_problems", str(args.max_problems),
                "--temperature", str(args.temperature),
                "--max_tokens", str(args.max_tokens),
                "--tensor_parallel_size", str(args.tensor_parallel_size),
                "--pipeline_parallel_size", str(args.pipeline_parallel_size),
                "--batch_size", str(args.val_infer_batch_size),
                "--max_model_len", str(args.max_model_len),
                "--concurrency", str(args.concurrency),
            ]
            val_reward_manifest_path = os.path.join(
                val_dir_override, "reward_scoring.json"
            )
            if args.inference_reward_path:
                infer_val_cmd.extend([
                    "--reward_path", args.inference_reward_path,
                    "--reward_name", args.reward_name,
                    "--reward_manifest_path", val_reward_manifest_path,
                ])
            # Skip val inference if responses already complete
            expected_val = val_lines * args.n_samples_val if val_lines > 0 else 0
            val_resp_json = os.path.join(val_dir_override, "responses.json")
            val_resp_sorted = os.path.join(val_dir_override, "responses_sorted.json")
            actual_val = 0
            if os.path.isfile(val_resp_sorted):
                try:
                    with open(val_resp_sorted, "r", encoding="utf-8") as f:
                        v = json.load(f)
                        if isinstance(v, list):
                            actual_val = len(v)
                except json.JSONDecodeError:
                    actual_val = 0
            elif os.path.isfile(val_resp_sorted):
                try:
                    with open(val_resp_sorted, "r", encoding="utf-8") as f:
                        v = json.load(f)
                        if isinstance(v, list):
                            actual_val = len(v)
                except json.JSONDecodeError:
                    actual_val = 0
            if expected_val > 0 and actual_val >= expected_val:
                print(f"Skip val inference for step {i}: found {actual_val}/{expected_val} responses")
            else:
                print("Executing:", " ".join(infer_val_cmd))
                _run_stage(infer_val_cmd)

            # Rescore cached validation rollouts with the same reward as VERL.
            if args.inference_reward_path:
                val_scoring_input = (
                    val_resp_json if os.path.isfile(val_resp_json) else val_resp_sorted
                )
                val_score_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "score_responses.py"),
                    "--responses_file", val_scoring_input,
                    "--reward_path", args.inference_reward_path,
                    "--reward_name", args.reward_name,
                    "--manifest_path", val_reward_manifest_path,
                ]
                print("Executing:", " ".join(val_score_cmd))
                _run_stage(val_score_cmd)
                if val_scoring_input == val_resp_json:
                    val_sort_cmd = [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "sort_responses.py"),
                        "--model_name", args.model,
                        "--dataset", args.val_dataset,
                        "--responses_dir", val_dir_override,
                    ]
                    print("Executing:", " ".join(val_sort_cmd))
                    _run_stage(val_sort_cmd)

        # 3) Gradient analysis (GradAlign modes or per-prompt SVD score)
        if args.mode in {"sim", "svd", "simacc", "align", "negsim", 'norm', 'dot'}:
            if args.mode == 'align' or args.mode == 'norm':
                args.val_dataset = args.train_dataset
                val_dir_override = train_resp_dir
            expected_analysis = chunk_sizes[chunk_idx]
            ana_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "run_parallel_analysis.py"),
                "--size", "1",
                "--idx", "0",
                "--model", args.model,
                "--resp_dataset", args.train_dataset,
                "--problem_dataset", args.train_dataset,
                "--problem_set_path", prompts_file,
                "--parts_root", train_parts_root,
                "--num_gpus", str(args.analysis_num_gpus),
                "--mini_batch_size", str(args.minibatch_size),
                "--analysis_backend", args.analysis_backend,
                "--rollout_n", str(args.n_samples_train),
                "--expected_groups", str(expected_analysis),
                "--prepare_workers", str(args.analysis_prepare_workers),
                "--mixed_precision", "bf16",
                "--max_length", str(args.max_model_len),
            ]
            if args.mode != 'svd':
                ana_cmd.extend([
                    "--val_dataset", args.val_dataset,
                    "--val_responses_dir", val_dir_override,
                ])
            if current_model_path:
                ana_cmd.extend(["--model_path", current_model_path])
            if args.mode in {'svd', 'norm', 'dot'}:
                ana_cmd.extend(["--mode", args.mode])
            if args.mode == 'svd':
                ana_cmd.extend([
                    "--svd_rank", str(args.svd_rank),
                    "--svd_score_scope", args.svd_score_scope,
                ])
                analysis_path = os.path.join(train_part_dir, f"svd_results_top{args.svd_rank}.jsonl")
            else:
                analysis_path = os.path.join(train_part_dir, "similarity_results_cosine_real.jsonl")

            if force_analysis_recompute and os.path.isfile(analysis_path):
                print(f"Removing stale analysis produced with different reward inputs: {analysis_path}")
                os.remove(analysis_path)
            if args.analysis_backend == "independent":
                existing_partition = _load_json_object(
                    os.path.join(
                        train_part_dir,
                        "data_independent",
                        "partition_manifest.json",
                    )
                )
                stale_partition = bool(existing_partition) and (
                    existing_partition.get("partition_schema_version") != 3
                    or existing_partition.get("token_input_source")
                    != "inference_token_ids"
                )
                if force_analysis_recompute or stale_partition:
                    ana_cmd.append("--force_independent_prepare")

            # Analysis writes one row per prompt/group_id, not per response.
            analysis_lines = _count_lines(analysis_path) if os.path.isfile(analysis_path) else 0
            analysis_compatible = True
            if (
                args.analysis_backend == "independent"
                and analysis_lines >= expected_analysis
                and expected_analysis > 0
            ):
                analysis_compatible = _independent_analysis_is_compatible(
                    analysis_path=analysis_path,
                    responses_dir=train_part_dir,
                    model_path=current_model_path,
                    world_size=args.analysis_num_gpus,
                    rollout_n=args.n_samples_train,
                    svd_rank=args.svd_rank,
                    svd_score_scope=args.svd_score_scope,
                    max_length=args.max_model_len,
                    expected_groups=expected_analysis,
                )
                if not analysis_compatible:
                    print(
                        "Existing analysis row count is complete but its independent "
                        "manifest/signature is incompatible; recomputing."
                    )
                    if "--force_independent_prepare" not in ana_cmd:
                        ana_cmd.append("--force_independent_prepare")
            if (
                analysis_lines >= expected_analysis
                and expected_analysis > 0
                and analysis_compatible
            ):
                print(f"Skip analysis for step {i}: found {analysis_lines}/{expected_analysis} prompt rows")
            else:
                print("Executing:", " ".join(ana_cmd))
                _run_stage(ana_cmd)

        # Aggregate accuracy plus the score file used by selection.
        if args.mode in {"sim", "svd", "simacc", "acc", "accgreedy", 'align', 'negsim', 'norm', 'dot'}:
            agg_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "aggregate.py"),
                "--model_name", args.model,
                "--dataset", args.train_dataset,
                "--parts_root", train_parts_root,
            ]
            if args.mode == 'svd':
                agg_cmd.extend(["--score_mode", "svd", "--svd_rank", str(args.svd_rank)])
            print("Executing:", " ".join(agg_cmd))
            _run_stage(agg_cmd)

        # 4) Select top-N according to mode
        iter_out = os.path.join(exp_ckpt_root, "selected", f"iter_{i}_{args.mode}_{select_n}")
        os.makedirs(iter_out, exist_ok=True)
        select_mode = args.mode
        if select_mode == 'dot' or select_mode == 'norm':
            select_mode = 'sim'
        # Preserve chunk-local candidates for scored selection modes, while the
        # random baseline samples from the complete training dataset.
        selection_dataset_dir = train_dir if args.mode == "rand" else chunk_dir
        sel_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "select_data.py"),
            "--mode", select_mode,
            "--dataset", args.train_dataset,
            "--dataset_dir", selection_dataset_dir,
            "--model", args.model,
            "--n", str(select_n),
            "--output_dir", iter_out,
            "--seed", str(args.seed),
            "--iteration", str(i),
            "--global_step", str(gstep),
        ]
        if args.mode in {"sim", "svd", "simacc", "acc", "accgreedy", "align", "negsim", 'norm', 'dot'}:
            sel_cmd.extend(["--parts_root", train_parts_root])
        if args.mode == "svd":
            sel_cmd.extend(["--svd_rank", str(args.svd_rank)])
        if args.mode == "acc":
            sel_cmd.extend(["--acc_low", str(args.acc_low), "--acc_high", str(args.acc_high)])
        print("Executing:", " ".join(sel_cmd))
        _run_stage(sel_cmd)

        # 5) Launch training with selected set, fixed experiment name
        os.system(f"rm {step_root}/data.pt")
        # print('remove data.pt')
        train_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "launch_verl_training.py"),
            "--prefix", args.prefix,
            "--model", args.model,
            "--train_dataset", args.train_dataset,
            "--train_dir", iter_out,
            "--ckpts_root", args.ckpt_root,
            "--project_name", args.project_name,
            "--trainer_logger", args.trainer_logger,
            "--backend", args.training_backend,
            "--total_epochs", f'{(i+1) * train_epochs}',
            "--exp_name", exp_name,
            '--train_batch_size', str(args.train_batch_size),
            '--max_prompt_length', str(args.max_prompt_length),
            '--max_response_length', str(args.max_tokens),
            '--ppo_mini_batch_size', str(args.ppo_mini_batch_size),
            '--ppo_micro_batch_size_per_gpu', str(args.ppo_micro_batch_size_per_gpu),
            '--lr', str(args.lr),
            '--rollout_n', str(args.rollout_n),
            '--ref_log_prob_micro_batch_size_per_gpu', str(args.ref_log_prob_micro_batch_size_per_gpu),
            '--rollout_log_prob_micro_batch_size_per_gpu', str(args.rollout_log_prob_micro_batch_size_per_gpu),
            '--n_gpus_per_node', str(args.n_gpus_per_node),
            '--save_freq', str(args.save_freq),
            '--test_freq', str(args.test_freq),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
            '--kl_loss_coef', str(args.kl_loss_coef),
        ]
        if val_dir and args.verl_val_set:
            train_cmd.extend(["--val_dataset", args.verl_val_set, "--val_dir", val_dir])
        train_cmd.append("--use-kl-loss" if args.use_kl_loss else "--no-use-kl-loss")
        if args.model_path:
            train_cmd.extend(["--model_path", args.model_path])
        if args.training_config_path:
            train_cmd.extend(["--config_path", args.training_config_path])
        if args.training_config_name:
            train_cmd.extend(["--config_name", args.training_config_name])
        if args.reward_manager:
            train_cmd.extend(["--reward_manager", args.reward_manager])
        if args.reward_path:
            train_cmd.extend([
                "--reward_path", args.reward_path,
                "--reward_name", args.reward_name,
            ])
            
        print("Executing:", " ".join(train_cmd))
        _run_stage(train_cmd)


if __name__ == "__main__":
    main()
