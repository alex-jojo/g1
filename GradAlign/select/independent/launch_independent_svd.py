#!/usr/bin/env python3
"""Prepare, launch, and merge isolated single-GPU SVD workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


HERE = Path(__file__).resolve().parent


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def parse_visible_devices(num_workers: int) -> List[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        devices = [value.strip() for value in configured.split(",") if value.strip()]
    else:
        devices = [str(index) for index in range(num_workers)]
    if len(devices) < num_workers:
        raise ValueError(
            f"Requested {num_workers} workers but CUDA_VISIBLE_DEVICES provides {devices}"
        )
    return devices[:num_workers]


def analysis_signature(config: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def output_matches_signature(path: str, signature: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line).get("analysis_signature") == signature
    except (OSError, json.JSONDecodeError):
        return False
    return False


def terminate_processes(processes: List[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 15.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch independent single-GPU SVD")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--responses_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=5120)
    parser.add_argument("--expected_groups", type=int, default=None)
    parser.add_argument("--prepare_workers", type=int, default=32)
    parser.add_argument("--svd_rank", type=int, default=128)
    parser.add_argument("--svd_oversample", type=int, default=8)
    parser.add_argument("--svd_niter", type=int, default=2)
    parser.add_argument("--svd_seed", type=int, default=0)
    parser.add_argument(
        "--gradient_parameter_scope",
        choices=["qkvo_only", "transformer_2d"],
        default="transformer_2d",
    )
    parser.add_argument(
        "--svd_score_scope",
        choices=["qkvo_only", "ffn_only", "transformer_2d"],
        default="transformer_2d",
    )
    parser.add_argument("--force_prepare", action="store_true")
    args = parser.parse_args()

    if args.gradient_parameter_scope != "transformer_2d":
        parser.error("Independent SVD always records QKVO and FFN matrices")

    if args.num_workers <= 0 or args.rollout_n <= 0:
        parser.error("num_workers and rollout_n must be positive")
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    responses_dir = os.path.abspath(os.path.expanduser(args.responses_dir))
    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    if args.force_prepare:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for rank in range(args.num_workers):
            worker_output = os.path.join(responses_dir, f"svd_rank_{rank}.jsonl")
            if os.path.isfile(worker_output):
                stale_path = f"{worker_output}.stale_{timestamp}"
                os.replace(worker_output, stale_path)
                print(f"Archived stale worker output to {stale_path}", flush=True)
    prepare_command = [
        sys.executable,
        str(HERE / "prepare.py"),
        "--responses_dir",
        responses_dir,
        "--model_path",
        model_path,
        "--world_size",
        str(args.num_workers),
        "--rollout_n",
        str(args.rollout_n),
        "--max_length",
        str(args.max_length),
        "--num_workers",
        str(args.prepare_workers),
    ]
    if args.expected_groups is not None:
        prepare_command.extend(["--expected_groups", str(args.expected_groups)])
    if args.force_prepare:
        prepare_command.append("--force")
    print("Preparing independent shards:", " ".join(prepare_command), flush=True)
    subprocess.run(prepare_command, check=True)

    data_dir = os.path.join(responses_dir, "data_independent")
    manifest_path = os.path.join(data_dir, "partition_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    analysis_config = {
        "record_schema_version": 5,
        "analysis_backend": "independent",
        "advantage_estimator": "grpo",
        "norm_adv_by_std_in_grpo": True,
        "advantage_epsilon": 1e-6,
        "advantage_std_correction": 1,
        "partition_strategy": "group_token_greedy",
        "partition_schema_version": manifest["partition_schema_version"],
        "partition_signature": manifest["partition_signature"],
        "token_input_source": manifest["token_input_source"],
        "token_schema_version": manifest["token_schema_version"],
        "token_ids_encoding": manifest["token_ids_encoding"],
        "world_size": args.num_workers,
        "rollout_n": args.rollout_n,
        "analysis_minibatch_size": 1,
        "gradient_parameter_scope": args.gradient_parameter_scope,
        "svd_score_scope": args.svd_score_scope,
        "matrix_statistics": [
            "frobenius_norm",
            "spectral_norm",
            "stable_rank",
            "topk_energy_ratio",
        ],
        "spectral_statistics_source": "randomized_top_singular_values",
        "group_statistics_output": "group_stats.jsonl",
        "svd_rank": args.svd_rank,
        "svd_q": args.svd_rank + args.svd_oversample,
        "svd_niter": args.svd_niter,
        "svd_seed": args.svd_seed,
        "max_length": args.max_length,
        "expected_groups": manifest["expected_groups"],
        "expected_responses": manifest["expected_responses"],
        "model_path": model_path,
        "responses_sha256": manifest["responses_sha256"],
    }
    signature = analysis_signature(analysis_config)
    manifest["analysis"] = analysis_config
    manifest["analysis_signature"] = signature
    manifest["gradient_parameter_scope"] = args.gradient_parameter_scope
    manifest["svd_score_scope"] = args.svd_score_scope
    atomic_write_json(manifest_path, manifest)
    print(f"Independent analysis signature: {signature}", flush=True)

    devices = parse_visible_devices(args.num_workers)
    stale_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for rank in range(args.num_workers):
        worker_output = os.path.join(responses_dir, f"svd_rank_{rank}.jsonl")
        if os.path.isfile(worker_output) and not output_matches_signature(
            worker_output, signature
        ):
            stale_path = (
                f"{worker_output}.stale_{stale_timestamp}_{signature[:12]}"
            )
            os.replace(worker_output, stale_path)
            print(
                f"Archived incompatible worker output to {stale_path}",
                flush=True,
            )
    processes: List[subprocess.Popen] = []
    for rank, device in enumerate(devices):
        worker_output = os.path.join(responses_dir, f"svd_rank_{rank}.jsonl")
        worker_command = [
            sys.executable,
            str(HERE / "svd_worker.py"),
            "--rank",
            str(rank),
            "--world_size",
            str(args.num_workers),
            "--model_path",
            model_path,
            "--shard_path",
            os.path.join(data_dir, f"data_rank_{rank}.npz"),
            "--manifest_path",
            manifest_path,
            "--output_path",
            worker_output,
            "--analysis_signature",
            signature,
            "--rollout_n",
            str(args.rollout_n),
            "--svd_rank",
            str(args.svd_rank),
            "--svd_oversample",
            str(args.svd_oversample),
            "--svd_niter",
            str(args.svd_niter),
            "--svd_seed",
            str(args.svd_seed),
            "--gradient_parameter_scope",
            args.gradient_parameter_scope,
            "--svd_score_scope",
            args.svd_score_scope,
        ]
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = device
        worker_env["PYTHONUNBUFFERED"] = "1"
        for variable in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            worker_env.pop(variable, None)
        print(
            f"Launching worker {rank} on CUDA_VISIBLE_DEVICES={device}: "
            + " ".join(worker_command),
            flush=True,
        )
        processes.append(subprocess.Popen(worker_command, env=worker_env))

    try:
        while True:
            failed = [process for process in processes if process.poll() not in (None, 0)]
            if failed:
                terminate_processes(processes)
                return_codes = [process.poll() for process in processes]
                raise subprocess.CalledProcessError(
                    next(code for code in return_codes if code not in (None, 0)),
                    "independent SVD workers",
                )
            if all(process.poll() == 0 for process in processes):
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        terminate_processes(processes)
        raise

    merge_command = [
        sys.executable,
        str(HERE / "merge_svd_results.py"),
        "--responses_dir",
        responses_dir,
        "--manifest_path",
        manifest_path,
        "--output_path",
        output_path,
        "--world_size",
        str(args.num_workers),
        "--analysis_signature",
        signature,
    ]
    print("Merging independent SVD results:", " ".join(merge_command), flush=True)
    subprocess.run(merge_command, check=True)


if __name__ == "__main__":
    main()
