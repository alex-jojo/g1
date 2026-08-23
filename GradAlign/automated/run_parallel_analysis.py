#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

from config import get_dataset_dir, get_response_dir, get_model_path


def part_dir_exists(parts_root: str, part_index: int) -> bool:
    return os.path.isdir(os.path.join(parts_root, f"part_{part_index}"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate part indices as size*k+idx and run parallel analysis for each existing part."
        )
    )
    parser.add_argument("--size", required=True, type=int, help="Number of machines")
    parser.add_argument("--idx", required=True, type=int, help="This machine index [0..size-1]")

    # Response split location (consistent with split_responses.py convention)
    parser.add_argument("--model", required=True, type=str, help="Model key (used for responses dir and dataset resolution)")
    parser.add_argument("--resp_dataset", required=True, type=str, help="Dataset name used for responses")
    parser.add_argument("--parts_root", default=None, type=str, help="Override responses split root (use parts_root/part_*)")

    # Problem set used in analysis
    parser.add_argument("--problem_dataset", required=True, type=str, help="Dataset name for prompts")
    parser.add_argument(
        "--problem_set_path",
        default=None,
        type=str,
        help="Explicit train.jsonl containing the prompts and ground truth for this response shard",
    )

    # Validation responses directory used by analysis
    parser.add_argument("--val_dataset", default=None, type=str,
                        help="Validation responses dataset name (not used in svd mode)")
    parser.add_argument("--val_responses_dir", default=None, type=str, help="Override validation responses directory")

    # Other analysis hyperparameters
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument("--mini_batch_size", default=2, type=int)
    parser.add_argument(
        "--analysis_backend",
        choices=["fsdp", "independent"],
        default="fsdp",
        help="FSDP collective analysis or isolated single-GPU SVD workers",
    )
    parser.add_argument("--rollout_n", default=8, type=int)
    parser.add_argument("--expected_groups", default=None, type=int)
    parser.add_argument("--prepare_workers", default=32, type=int)
    parser.add_argument("--force_independent_prepare", action="store_true")
    parser.add_argument("--mixed_precision", default="bf16", type=str)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--cpu_offload", action="store_true", default=True)
    parser.add_argument("--max_num_samples", default=10240000, type=int)
    # Model path resolved from config
    # (no explicit --model_path; derived from --model)
    parser.add_argument("--start_k", default=0, type=int, help="Start enumeration from this k (default 0)")
    parser.add_argument("--max_k", default=None, type=int, help="Optional max k to try (exclusive)")
    parser.add_argument("--model_path", default=None, type=str, help="Override model path for analysis")
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--optimizer_state_path", default=None, type=str)
    parser.add_argument("--mode", default='sim', type=str)
    parser.add_argument("--svd_rank", default=128, type=int,
                        help="Number of leading singular values saved per 2-D layer gradient in svd mode")
    parser.add_argument(
        "--svd_score_scope",
        "--svd_parameter_scope",
        dest="svd_score_scope",
        choices=["qkvo_only", "ffn_only", "transformer_2d"],
        default=None,
        help=(
            "Matrix families included in S. Independent always records all seven; "
            "the old --svd_parameter_scope name remains as an alias."
        ),
    )

    args = parser.parse_args()

    # assert 0 <= args.idx < args.size, "idx must be in [0, size)"

    parts_root = args.parts_root or (get_response_dir(args.resp_dataset, args.model) + "_split")
    problem_set = (
        os.path.abspath(os.path.expanduser(args.problem_set_path))
        if args.problem_set_path
        else os.path.join(get_dataset_dir(args.problem_dataset, args.model), "train.jsonl")
    )
    if args.mode != 'svd' and not (args.val_responses_dir or args.val_dataset):
        parser.error("--val_dataset or --val_responses_dir is required unless --mode svd is used")
    if args.mode == 'svd' and args.use_optimizer:
        parser.error("--use_optimizer is incompatible with --mode svd; SVD uses raw GRPO gradients")
    if args.analysis_backend == "independent" and args.mode != "svd":
        parser.error("--analysis_backend independent is only supported with --mode svd")
    if args.analysis_backend == "independent" and args.mini_batch_size != 1:
        parser.error("Independent SVD currently requires --mini_batch_size 1")
    if args.svd_score_scope is None:
        args.svd_score_scope = (
            "transformer_2d"
            if args.analysis_backend == "independent"
            else "qkvo_only"
        )
    if args.analysis_backend != "independent" and args.svd_score_scope != "qkvo_only":
        parser.error(
            "Non-QKVO SVD score scopes require "
            "--analysis_backend independent"
        )
    if args.num_gpus <= 0 or args.rollout_n <= 0 or args.prepare_workers <= 0:
        parser.error("num_gpus, rollout_n, and prepare_workers must be positive")
    val_responses_dir = None
    if args.mode != 'svd':
        val_responses_dir = args.val_responses_dir or get_response_dir(args.val_dataset, args.model)

    if not os.path.isfile(problem_set):
        raise SystemExit(f"Problem set not found: {problem_set}")
    if not os.path.isdir(parts_root):
        raise SystemExit(f"Parts root not found: {parts_root}")
    if args.mode != 'svd' and not os.path.isdir(val_responses_dir):
        raise SystemExit(f"Validation responses dir not found: {val_responses_dir}")

    parallel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "parallel"))
    independent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "independent"))
    fsdp_launcher = os.path.join(parallel_dir, "launch_parallel_analysis.py")
    independent_launcher = os.path.join(independent_dir, "launch_independent_svd.py")

    # Resolve model path from CLI override, env, or config
    model_path = args.model_path or os.environ.get("MODEL_PATH_OVERRIDE", get_model_path(args.model))

    k = args.start_k
    processed_any = False
    while True:
        if args.max_k is not None and k >= args.max_k:
            break
        part_index = args.size * k + args.idx
        if not part_dir_exists(parts_root, part_index):
            # Stop if the next assigned part is missing
            if not processed_any and k == args.start_k:
                print(f"No part found for part_index={part_index} at {parts_root}")
            break

        part_dir = os.path.join(parts_root, f"part_{part_index}")
        output_filename = (
            f"svd_results_top{args.svd_rank}.jsonl"
            if args.mode == 'svd'
            else "similarity_results_cosine_real.jsonl"
        )
        output_path = os.path.join(part_dir, output_filename)

        if args.analysis_backend == "independent":
            cmd = [
                sys.executable, independent_launcher,
                "--model_path", model_path,
                "--responses_dir", part_dir,
                "--output_path", output_path,
                "--num_workers", str(args.num_gpus),
                "--rollout_n", str(args.rollout_n),
                "--max_length", str(args.max_length),
                "--prepare_workers", str(args.prepare_workers),
                "--svd_rank", str(args.svd_rank),
                "--gradient_parameter_scope", "transformer_2d",
                "--svd_score_scope", args.svd_score_scope,
            ]
            if args.expected_groups is not None:
                cmd.extend(["--expected_groups", str(args.expected_groups)])
            if args.force_independent_prepare:
                cmd.append("--force_prepare")
            command_cwd = independent_dir
        else:
            cmd = [
                sys.executable, fsdp_launcher,
                "--model_path", model_path,
                "--train_responses_dir", part_dir,
                "--output_path", output_path,
                "--problem_set_path", problem_set,
                "--num_gpus", str(args.num_gpus),
                "--mini_batch_size", str(args.mini_batch_size),
                "--mixed_precision", args.mixed_precision,
                "--max_length", str(args.max_length),
                "--mode", args.mode,
                "--svd_rank", str(args.svd_rank),
            ]
            if args.mode != 'svd':
                cmd.extend(["--val_responses_dir", val_responses_dir])
            if args.cpu_offload:
                cmd.append("--cpu_offload")
            if args.max_num_samples is not None:
                cmd.extend(["--max_num_samples", str(args.max_num_samples)])
            if args.use_optimizer:
                cmd.append("--use_optimizer")
                cmd.extend(["--optimizer_state_path", args.optimizer_state_path])
            command_cwd = parallel_dir

        env = os.environ.copy()
        env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

        print(f"Executing (backend={args.analysis_backend}):", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=command_cwd, env=env)
        processed_any = True
        k += 1

    print("Done.")


if __name__ == "__main__":
    main()
