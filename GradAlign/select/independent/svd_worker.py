#!/usr/bin/env python3
"""Analyze complete response groups on one isolated GPU without collectives."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any, Dict, List, Set

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from adamw import (  # noqa: E402
    NamedAdamWSnapshot,
    UPDATE_TARGETS,
    global_grad_norm,
    grad_clip_coefficient,
    simulate_adamw_delta,
)
from reference_basis import ReferenceBasis  # noqa: E402


SELECT_DIR = Path(__file__).resolve().parents[1]
if str(SELECT_DIR) not in sys.path:
    sys.path.insert(0, str(SELECT_DIR))

from svd_utils import (  # noqa: E402
    PARAMETER_SCOPES,
    SCORE_SCOPES,
    aggregate_effective_rank,
    is_matrix_parameter,
    truncated_svd,
    truncated_svd_with_subspace,
    aggregate_subspace_similarity,
    zero_svd_result,
)


GRPO_ADVANTAGE_EPSILON = 1e-6
GRPO_ADVANTAGE_STD_CORRECTION = 1


def compute_grpo_advantages(rewards: torch.Tensor) -> tuple:
    """Match VERL's outcome-only GRPO normalization for one response group."""
    if rewards.numel() == 1:
        reward_mean = torch.tensor(0.0, dtype=rewards.dtype)
        reward_std = torch.tensor(1.0, dtype=rewards.dtype)
    else:
        reward_mean = rewards.mean()
        # torch.std defaults to correction=1, matching VERL GRPO.
        reward_std = rewards.std()
    advantages = (rewards - reward_mean) / (
        reward_std + GRPO_ADVANTAGE_EPSILON
    )
    return advantages, reward_mean, reward_std


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_shard(path: str) -> List[Dict[str, Any]]:
    required = {
        "input_ids",
        "attention_mask",
        "response_mask",
        "original_input_length",
        "was_truncated",
        "prompt_length",
        "generated_response_length",
        "generation_hit_max_tokens",
        "generation_finish_reason",
        "generation_stop_reason",
        "requested_max_tokens",
        "group_id",
        "global_index",
        "reward_score",
    }
    # NpzFile loads members lazily. Looking up arrays["field"] inside the row
    # loop would decompress the complete member once per row, turning a small
    # shard into gigabytes of repeated CPU work before the model reaches CUDA.
    # Materialize each required member once, then close the archive.
    with np.load(path, allow_pickle=True) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Shard {path} is missing arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    entries = []
    for index in range(len(arrays["group_id"])):
        entries.append(
            {
                "input_ids": np.asarray(arrays["input_ids"][index], dtype=np.int64),
                "attention_mask": np.asarray(
                    arrays["attention_mask"][index], dtype=np.int64
                ),
                "response_mask": np.asarray(
                    arrays["response_mask"][index], dtype=np.uint8
                ),
                "original_input_length": int(arrays["original_input_length"][index]),
                "was_truncated": bool(arrays["was_truncated"][index]),
                "prompt_length": int(arrays["prompt_length"][index]),
                "generated_response_length": int(
                    arrays["generated_response_length"][index]
                ),
                "generation_hit_max_tokens": bool(
                    arrays["generation_hit_max_tokens"][index]
                ),
                "generation_finish_reason": str(
                    arrays["generation_finish_reason"][index]
                ),
                "generation_stop_reason": str(
                    arrays["generation_stop_reason"][index]
                ),
                "requested_max_tokens": int(
                    arrays["requested_max_tokens"][index]
                ),
                "group_id": int(arrays["group_id"][index]),
                "global_index": int(arrays["global_index"][index]),
                "reward_score": float(arrays["reward_score"][index]),
            }
        )
    return entries


def load_processed_groups(
    output_path: str,
    analysis_signature: str,
    assigned_groups: Set[int],
) -> Set[int]:
    processed: Set[int] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {output_path} at line {line_number}; "
                    "repair or archive the partial worker output before resuming"
                ) from error
            if row.get("analysis_signature") != analysis_signature:
                raise ValueError(
                    f"Worker output {output_path} was produced by a different analysis configuration"
                )
            group_id = int(row["group_id"])
            if group_id not in assigned_groups:
                raise ValueError(
                    f"Worker output contains unassigned group_id={group_id}: {output_path}"
                )
            if group_id in processed:
                raise ValueError(f"Duplicate group_id={group_id} in {output_path}")
            processed.add(group_id)
    return processed


def matrix_parameter_specs(
    model: torch.nn.Module, parameter_scope: str
) -> List[tuple]:
    specs = []
    for name, parameter in model.named_parameters():
        if is_matrix_parameter(name, parameter.shape, parameter_scope):
            specs.append((name, parameter))
    specs.sort(key=lambda item: item[0])
    if not specs:
        raise RuntimeError(
            f"No matrix weights were found for parameter scope {parameter_scope}"
        )
    return specs


def configure_matrix_gradients(
    model: torch.nn.Module, parameter_scope: str, *, require_all_gradients: bool
) -> List[tuple]:
    specs = matrix_parameter_specs(model, parameter_scope)
    if require_all_gradients:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for _, parameter in specs:
            parameter.requires_grad_(True)
    return specs


def compute_group_gradients(
    model: torch.nn.Module,
    group_entries: List[Dict[str, Any]],
    device: torch.device,
) -> tuple:
    rewards = torch.tensor(
        [entry["reward_score"] for entry in group_entries], dtype=torch.float32
    )
    advantages, reward_mean, reward_std = compute_grpo_advantages(rewards)
    response_tokens = [int(entry["response_mask"].sum()) for entry in group_entries]
    total_response_tokens = sum(response_tokens)
    if total_response_tokens <= 0:
        raise ValueError(
            f"group_id={group_entries[0]['group_id']} has no valid response tokens"
        )
    reward_values = [float(value) for value in rewards.tolist()]
    advantage_values = [float(value) for value in advantages.tolist()]
    zero_advantage = bool(torch.all(torch.abs(advantages) < 1e-6).item())
    group_stats = {
        "global_indices": [int(entry["global_index"]) for entry in group_entries],
        "total_response_tokens": total_response_tokens,
        "response_tokens": response_tokens,
        "prompt_tokens": [int(entry["prompt_length"]) for entry in group_entries],
        "generated_response_tokens": [
            int(entry["generated_response_length"]) for entry in group_entries
        ],
        "input_tokens": [int(len(entry["input_ids"])) for entry in group_entries],
        "original_input_tokens": [
            int(entry["original_input_length"]) for entry in group_entries
        ],
        "analysis_truncated_flags": [
            bool(entry["was_truncated"]) for entry in group_entries
        ],
        "analysis_truncated_responses": sum(
            bool(entry["was_truncated"]) for entry in group_entries
        ),
        "generation_hit_max_tokens_flags": [
            bool(entry["generation_hit_max_tokens"]) for entry in group_entries
        ],
        "generation_hit_max_tokens_responses": sum(
            bool(entry["generation_hit_max_tokens"]) for entry in group_entries
        ),
        "generation_finish_reasons": [
            entry["generation_finish_reason"] for entry in group_entries
        ],
        "generation_stop_reasons": [
            entry["generation_stop_reason"] for entry in group_entries
        ],
        "requested_max_tokens": [
            int(entry["requested_max_tokens"]) for entry in group_entries
        ],
        "token_input_source": "inference_token_ids",
        "advantage_estimator": "grpo",
        "norm_adv_by_std_in_grpo": True,
        "advantage_epsilon": GRPO_ADVANTAGE_EPSILON,
        "advantage_std_correction": GRPO_ADVANTAGE_STD_CORRECTION,
        "reward_mean": float(reward_mean.item()),
        "reward_std": float(reward_std.item()),
        "rewards": reward_values,
        "advantages": advantage_values,
    }
    model.zero_grad(set_to_none=True)
    if zero_advantage:
        return 0.0, True, group_stats

    accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
    for entry, advantage, token_count in zip(
        group_entries, advantage_values, response_tokens
    ):
        if token_count == 0 or advantage == 0.0:
            continue
        input_ids = torch.from_numpy(entry["input_ids"]).unsqueeze(0).to(device)
        attention_mask = (
            torch.from_numpy(entry["attention_mask"]).unsqueeze(0).to(device)
        )
        response_mask = (
            torch.from_numpy(entry["response_mask"]).unsqueeze(0).to(device).bool()
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        if response_mask.shape != shift_labels.shape:
            raise ValueError(
                f"response_mask shape {tuple(response_mask.shape)} does not match "
                f"labels shape {tuple(shift_labels.shape)}"
            )
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape_as(shift_labels)
        weighted_loss = (
            token_ce.masked_select(response_mask).sum()
            * float(advantage)
            / float(total_response_tokens)
        )
        accumulated_loss.add_(weighted_loss.detach().float())
        weighted_loss.backward()
        del (
            outputs,
            shift_logits,
            shift_labels,
            token_ce,
            weighted_loss,
            input_ids,
            attention_mask,
            response_mask,
        )
    group_loss = float(accumulated_loss.item())
    del accumulated_loss
    return group_loss, False, group_stats


def compute_group_svd(
    parameter_specs: List[tuple],
    zero_advantage: bool,
    gradient_source: str,
    optimizer_snapshot: NamedAdamWSnapshot | None,
    cold_start_group: Dict[str, Any],
    adamw_update_target: str,
    gradient_scale: float,
    analysis_method: str,
    reference_basis: ReferenceBasis | None,
    svd_rank: int,
    oversample: int,
    niter: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    zero_marginal_subspace = (
        analysis_method == "subspace"
        and gradient_source == "adamw"
        and adamw_update_target == "marginal_data"
        and zero_advantage
    )
    for name, parameter in parameter_specs:
        if (gradient_source == "raw" and zero_advantage) or zero_marginal_subspace:
            result = zero_svd_result(parameter.shape, svd_rank)
            if analysis_method == "subspace":
                result.update(
                    {
                        "subspace_rank": min(svd_rank, min(parameter.shape)),
                        "subspace_phi_u": 0.0,
                        "subspace_phi_v": 0.0,
                    }
                )
        else:
            if parameter.grad is None and not zero_advantage:
                raise RuntimeError(f"Missing matrix gradient for {name}")
            gradient = (
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach().mul(gradient_scale)
            )
            analysis_tensor = gradient
            if gradient_source == "adamw":
                if optimizer_snapshot is None:
                    state, group = {}, cold_start_group
                else:
                    state, group = optimizer_snapshot.state_and_group(name)
                delta = simulate_adamw_delta(parameter, gradient, state, group)
                analysis_tensor = getattr(delta, adamw_update_target)
            if analysis_method == "subspace":
                if reference_basis is None:
                    raise RuntimeError("Subspace analysis requires an initial basis")
                reference_u, reference_v = reference_basis.factors(name)
                result = truncated_svd_with_subspace(
                    analysis_tensor,
                    reference_u=reference_u,
                    reference_v=reference_v,
                    svd_rank=svd_rank,
                    oversample=oversample,
                    niter=niter,
                    seed=seed,
                )
            else:
                result = truncated_svd(
                    analysis_tensor,
                    svd_rank=svd_rank,
                    oversample=oversample,
                    niter=niter,
                    seed=seed,
                )
        results[name] = result
    return results


def append_result(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent single-GPU SVD worker")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--shard_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--analysis_signature", required=True)
    parser.add_argument(
        "--analysis_method",
        choices=["effective_rank", "subspace"],
        default="effective_rank",
    )
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--svd_rank", type=int, default=128)
    parser.add_argument("--svd_oversample", type=int, default=8)
    parser.add_argument("--svd_niter", type=int, default=2)
    parser.add_argument("--svd_seed", type=int, default=0)
    parser.add_argument(
        "--gradient_parameter_scope",
        choices=PARAMETER_SCOPES,
        default="transformer_2d",
    )
    parser.add_argument(
        "--svd_score_scope",
        choices=SCORE_SCOPES,
        default="transformer_2d",
    )
    parser.add_argument(
        "--svd_gradient_source",
        choices=["raw", "adamw"],
        default="raw",
    )
    parser.add_argument("--optimizer_state_path", default=None)
    parser.add_argument("--reference_basis_path", default=None)
    parser.add_argument(
        "--subspace_score_side",
        choices=["u", "v", "mean"],
        default="mean",
    )
    parser.add_argument(
        "--adamw_update_target",
        choices=UPDATE_TARGETS,
        default="full",
    )
    parser.add_argument("--adamw_lr", type=float, default=1e-6)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.999)
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    if args.gradient_parameter_scope != "transformer_2d":
        parser.error("Independent SVD must record QKVO and FFN matrices")
    if args.svd_gradient_source == "raw" and args.optimizer_state_path:
        parser.error("--optimizer_state_path requires --svd_gradient_source adamw")
    if args.optimizer_state_path and not os.path.isfile(args.optimizer_state_path):
        parser.error(f"Optimizer state not found: {args.optimizer_state_path}")
    if args.analysis_method == "subspace" and args.svd_gradient_source != "adamw":
        parser.error("Subspace analysis requires --svd_gradient_source adamw")
    if args.analysis_method == "subspace" and not args.reference_basis_path:
        parser.error("Subspace analysis requires --reference_basis_path")
    if args.reference_basis_path and not os.path.isfile(args.reference_basis_path):
        parser.error(f"Reference basis not found: {args.reference_basis_path}")

    optimizer_snapshot = (
        NamedAdamWSnapshot.load(args.optimizer_state_path)
        if args.optimizer_state_path
        else None
    )
    reference_basis = (
        ReferenceBasis.load(args.reference_basis_path)
        if args.reference_basis_path
        else None
    )
    if reference_basis is not None and reference_basis.rank < args.svd_rank:
        parser.error(
            f"Reference basis rank {reference_basis.rank} is below requested "
            f"rank {args.svd_rank}"
        )
    cold_start_group = {
        "lr": args.adamw_lr,
        "betas": (args.adamw_beta1, args.adamw_beta2),
        "eps": args.adamw_eps,
        "weight_decay": args.adamw_weight_decay,
        "amsgrad": False,
        "maximize": False,
    }

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Independent worker must see exactly one GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    manifest = load_manifest(args.manifest_path)
    worker_manifest = manifest["workers"][str(args.rank)]
    assigned_group_order = [int(value) for value in worker_manifest["group_ids"]]
    assigned_groups = set(assigned_group_order)
    entries = load_shard(args.shard_path)
    grouped_entries: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry["group_id"]].append(entry)
    if set(grouped_entries) != assigned_groups:
        raise ValueError(f"rank {args.rank} shard does not match its manifest assignment")
    for group_id, group_entries in grouped_entries.items():
        group_entries.sort(key=lambda entry: entry["global_index"])
        if len(group_entries) != args.rollout_n:
            raise ValueError(
                f"rank {args.rank} group_id={group_id} has {len(group_entries)} "
                f"responses, expected {args.rollout_n}"
            )
    processed = load_processed_groups(
        args.output_path, args.analysis_signature, assigned_groups
    )
    remaining_groups = [
        group_id for group_id in assigned_group_order if group_id not in processed
    ]
    print(
        f"[worker {args.rank}] assigned={len(assigned_groups)}, "
        f"processed={len(processed)}, remaining={len(remaining_groups)}",
        flush=True,
    )
    if not remaining_groups:
        return

    model = AutoModelForCausalLM.from_pretrained(
        os.path.abspath(os.path.expanduser(args.model_path)),
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    parameter_specs = configure_matrix_gradients(
        model,
        args.gradient_parameter_scope,
        require_all_gradients=args.svd_gradient_source == "adamw",
    )
    model.to(device)
    model.train()
    print(
        f"[worker {args.rank}] loaded model; scope={args.gradient_parameter_scope} "
        f"method={args.analysis_method} score_scope={args.svd_score_scope} "
        f"source={args.svd_gradient_source} "
        f"adamw_target={args.adamw_update_target} matrices={len(parameter_specs)}",
        flush=True,
    )

    for completed_index, group_id in enumerate(remaining_groups, 1):
        started = time()
        group_loss, zero_advantage, group_stats = compute_group_gradients(
            model, grouped_entries[group_id], device
        )
        gradient_norm = (
            global_grad_norm(model.parameters())
            if args.svd_gradient_source == "adamw"
            else None
        )
        gradient_scale = (
            grad_clip_coefficient(gradient_norm, args.adamw_grad_clip)
            if gradient_norm is not None
            else 1.0
        )
        matrix_results = compute_group_svd(
            parameter_specs,
            zero_advantage=zero_advantage,
            gradient_source=args.svd_gradient_source,
            optimizer_snapshot=optimizer_snapshot,
            cold_start_group=cold_start_group,
            adamw_update_target=args.adamw_update_target,
            gradient_scale=gradient_scale,
            analysis_method=args.analysis_method,
            reference_basis=reference_basis,
            svd_rank=args.svd_rank,
            oversample=args.svd_oversample,
            niter=args.svd_niter,
            seed=args.svd_seed,
        )
        marginal_delta_all_zero = (
            args.analysis_method == "subspace"
            and args.adamw_update_target == "marginal_data"
            and all(
                float(matrix_result["frobenius_norm"]) == 0.0
                for matrix_result in matrix_results.values()
            )
        )
        selection_filter_reason = None
        if args.analysis_method == "subspace":
            if zero_advantage:
                selection_filter_reason = "zero_advantage"
            elif marginal_delta_all_zero:
                selection_filter_reason = "zero_marginal_delta"
        selection_eligible = selection_filter_reason is None
        effective_rank_score = aggregate_effective_rank(
            matrix_results,
            args.svd_rank,
            args.svd_score_scope,
        )
        subspace_score = (
            aggregate_subspace_similarity(
                matrix_results,
                args.svd_rank,
                args.svd_score_scope,
                args.subspace_score_side,
            )
            if args.analysis_method == "subspace"
            else None
        )
        selection_score = (
            float(subspace_score["s"])
            if subspace_score is not None
            else float(effective_rank_score["s"])
        )
        group_stats.update(
            {
                "group_id": group_id,
                "worker_rank": args.rank,
                "loss": group_loss,
                "zero_advantage": zero_advantage,
                "marginal_delta_all_zero": marginal_delta_all_zero,
                "selection_eligible": selection_eligible,
                "selection_filter_reason": selection_filter_reason,
                "analysis_method": args.analysis_method,
                "svd_gradient_source": args.svd_gradient_source,
                "adamw_update_target": (
                    args.adamw_update_target
                    if args.svd_gradient_source == "adamw"
                    else None
                ),
                "gradient_global_norm": gradient_norm,
                "gradient_clip_coefficient": gradient_scale,
                "svd_score_scope": args.svd_score_scope,
                "svd_score": float(effective_rank_score["s"]),
                "subspace_score_side": (
                    args.subspace_score_side
                    if args.analysis_method == "subspace"
                    else None
                ),
                "subspace_phi_u": (
                    float(subspace_score["phi_u"])
                    if subspace_score is not None
                    else None
                ),
                "subspace_phi_v": (
                    float(subspace_score["phi_v"])
                    if subspace_score is not None
                    else None
                ),
                "subspace_score": (
                    float(subspace_score["s"])
                    if subspace_score is not None
                    else None
                ),
            }
        )
        row = {
            "record_schema_version": 6 if args.analysis_method == "subspace" else 5,
            "analysis_backend": "independent",
            "analysis_signature": args.analysis_signature,
            "worker_rank": args.rank,
            "group_id": group_id,
            "loss": group_loss,
            "zero_advantage": zero_advantage,
            "marginal_delta_all_zero": marginal_delta_all_zero,
            "selection_eligible": selection_eligible,
            "selection_filter_reason": selection_filter_reason,
            "analysis_method": args.analysis_method,
            "svd_gradient_source": args.svd_gradient_source,
            "adamw_update_target": (
                args.adamw_update_target
                if args.svd_gradient_source == "adamw"
                else None
            ),
            "optimizer_state_source": (
                optimizer_snapshot.source_path
                if optimizer_snapshot is not None
                else ("cold_start" if args.svd_gradient_source == "adamw" else None)
            ),
            "gradient_parameter_scope": args.gradient_parameter_scope,
            "svd_score_scope": args.svd_score_scope,
            "svd_rank": args.svd_rank,
            "svd_method": "torch.svd_lowrank",
            "svd_q": args.svd_rank + args.svd_oversample,
            "svd_niter": args.svd_niter,
            "svd_seed": args.svd_seed,
            "matrices": matrix_results,
            "effective_rank_topk": effective_rank_score,
            "subspace_similarity": subspace_score,
            "group_stats": group_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_result(args.output_path, row)
        model.zero_grad(set_to_none=True)
        print(
            f"[worker {args.rank}] {completed_index}/{len(remaining_groups)} "
            f"group_id={group_id} zero={zero_advantage} "
            f"eligible={selection_eligible} "
            f"filter_reason={selection_filter_reason} "
            f"method={args.analysis_method} score={selection_score:.6f} "
            f"elapsed={time() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
