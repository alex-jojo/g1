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
    parser.add_argument(
        "--val_dataset",
        default=None,
        type=str,
        help="Validation responses dataset name (not used in svd/subspace modes)",
    )
    parser.add_argument("--val_responses_dir", default=None, type=str, help="Override validation responses directory")

    # Other analysis hyperparameters
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument("--mini_batch_size", default=2, type=int)
    parser.add_argument(
        "--analysis_backend",
        choices=["fsdp", "independent"],
        default="fsdp",
        help="FSDP collective analysis or isolated single-GPU spectral workers",
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
    parser.add_argument("--reference_model_path", default=None, type=str)
    parser.add_argument("--reference_basis_path", default=None, type=str)
    parser.add_argument("--mode", default='sim', type=str)
    parser.add_argument(
        "--svd_rank",
        default=128,
        type=int,
        help="Number of leading singular directions used in svd/subspace modes",
    )
    parser.add_argument(
        "--svd_gradient_source",
        choices=["raw", "adamw"],
        default="raw",
        help="Analyze raw GRPO gradients or counterfactual AdamW parameter deltas",
    )
    parser.add_argument(
        "--adamw_update_target",
        choices=["actual_data", "marginal_data", "full"],
        default="full",
    )
    parser.add_argument("--adamw_lr", type=float, default=1e-6)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.999)
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--subspace_score_side",
        choices=["u", "v", "mean"],
        default="u",
    )
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
    spectral_modes = {"svd", "subspace"}
    if args.mode not in spectral_modes and not (
        args.val_responses_dir or args.val_dataset
    ):
        parser.error(
            "--val_dataset or --val_responses_dir is required unless "
            "--mode is svd or subspace"
        )
    if args.mode in spectral_modes and args.use_optimizer:
        parser.error(
            "Legacy --use_optimizer is incompatible with spectral analysis; use "
            "--svd_gradient_source adamw instead"
        )
    if args.svd_gradient_source == "adamw" and (
        args.mode not in spectral_modes or args.analysis_backend != "independent"
    ):
        parser.error(
            "AdamW spectral analysis requires --mode svd or --mode subspace "
            "with --analysis_backend independent"
        )
    if (
        args.mode in spectral_modes
        and args.optimizer_state_path
        and args.svd_gradient_source != "adamw"
    ):
        parser.error(
            "--optimizer_state_path in spectral analysis requires "
            "--svd_gradient_source adamw"
        )
    if args.analysis_backend == "independent" and args.mode not in spectral_modes:
        parser.error(
            "--analysis_backend independent is only supported with "
            "--mode svd/subspace"
        )
    if args.mode == "subspace" and args.svd_gradient_source != "adamw":
        parser.error("--mode subspace requires --svd_gradient_source adamw")
    if args.mode == "subspace" and (
        not args.reference_model_path or not args.reference_basis_path
    ):
        parser.error(
            "--mode subspace requires --reference_model_path and "
            "--reference_basis_path"
        )
    if args.analysis_backend == "independent" and args.mini_batch_size != 1:
        parser.error(
            "Independent spectral analysis currently requires --mini_batch_size 1"
        )
    if args.svd_score_scope is None:
        args.svd_score_scope = (
            "transformer_2d"
            if args.analysis_backend == "independent"
            else "qkvo_only"
        )
    if args.analysis_backend != "independent" and args.svd_score_scope != "qkvo_only":
        parser.error(
            "Non-QKVO spectral score scopes require "
            "--analysis_backend independent"
        )
    if args.num_gpus <= 0 or args.rollout_n <= 0 or args.prepare_workers <= 0:
        parser.error("num_gpus, rollout_n, and prepare_workers must be positive")
    val_responses_dir = None
    if args.mode not in spectral_modes:
        val_responses_dir = args.val_responses_dir or get_response_dir(args.val_dataset, args.model)

    if not os.path.isfile(problem_set):
        raise SystemExit(f"Problem set not found: {problem_set}")
    if not os.path.isdir(parts_root):
        raise SystemExit(f"Parts root not found: {parts_root}")
    if args.mode not in spectral_modes and not os.path.isdir(val_responses_dir):
        raise SystemExit(f"Validation responses dir not found: {val_responses_dir}")

    parallel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "parallel"))
    independent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "select", "independent"))
    fsdp_launcher = os.path.join(parallel_dir, "launch_parallel_analysis.py")
    independent_launchers = {
        "svd": os.path.join(independent_dir, "launch_independent_svd.py"),
        "subspace": os.path.join(
            independent_dir, "launch_independent_subspace.py"
        ),
    }

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
        if args.mode == "svd":
            output_filename = f"svd_results_top{args.svd_rank}.jsonl"
        elif args.mode == "subspace":
            output_filename = f"subspace_results_top{args.svd_rank}.jsonl"
        else:
            output_filename = "similarity_results_cosine_real.jsonl"
        output_path = os.path.join(part_dir, output_filename)

        if args.analysis_backend == "independent":
            independent_launcher = independent_launchers[args.mode]
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
                "--adamw_update_target", args.adamw_update_target,
                "--adamw_lr", str(args.adamw_lr),
                "--adamw_beta1", str(args.adamw_beta1),
                "--adamw_beta2", str(args.adamw_beta2),
                "--adamw_eps", str(args.adamw_eps),
                "--adamw_weight_decay", str(args.adamw_weight_decay),
                "--adamw_grad_clip", str(args.adamw_grad_clip),
            ]
            if args.optimizer_state_path:
                cmd.extend(["--optimizer_state_path", args.optimizer_state_path])
            if args.mode == "subspace":
                cmd.extend(
                    [
                        "--reference_model_path",
                        args.reference_model_path,
                        "--reference_basis_path",
                        args.reference_basis_path,
                        "--subspace_score_side",
                        args.subspace_score_side,
                    ]
                )
            else:
                cmd.extend(
                    ["--svd_gradient_source", args.svd_gradient_source]
                )
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
            if args.mode not in spectral_modes:
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
