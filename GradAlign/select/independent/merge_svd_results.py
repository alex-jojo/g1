#!/usr/bin/env python3
"""Validate and merge per-worker independent SVD JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List


MATRIX_STAT_KEYS = (
    "frobenius_norm",
    "spectral_norm",
    "stable_rank",
    "topk_energy_ratio",
    "topk_effective_rank",
)
GROUP_LIST_KEYS = (
    "global_indices",
    "response_tokens",
    "prompt_tokens",
    "generated_response_tokens",
    "input_tokens",
    "original_input_tokens",
    "analysis_truncated_flags",
    "generation_hit_max_tokens_flags",
    "generation_finish_reasons",
    "generation_stop_reasons",
    "requested_max_tokens",
    "rewards",
    "advantages",
)

SCORE_SCOPES = ("qkvo_only", "ffn_only", "transformer_2d")
QKVO_FAMILIES = ("Q", "K", "V", "O")
FFN_FAMILIES = ("GATE", "UP", "DOWN")


def read_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def existing_matches_signature(path: str, signature: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        rows = read_rows(path)
    except ValueError:
        return False
    return bool(rows) and all(row.get("analysis_signature") == signature for row in rows)


def validate_record(row: Dict[str, Any], rank: int, rollout_n: int) -> None:
    group_id = int(row["group_id"])
    if row.get("record_schema_version") != 5:
        raise ValueError(f"group_id={group_id} has an unsupported record schema")
    if int(row.get("worker_rank", -1)) != rank:
        raise ValueError(f"group_id={group_id} has the wrong worker_rank")
    matrices = row.get("matrices")
    if not isinstance(matrices, dict) or not matrices:
        raise ValueError(f"group_id={group_id} has no matrix results")
    for matrix_name, matrix in matrices.items():
        for key in MATRIX_STAT_KEYS:
            try:
                value = float(matrix[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"group_id={group_id} matrix={matrix_name} has invalid {key}"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"group_id={group_id} matrix={matrix_name} has invalid {key}={value}"
                )
        ratio = float(matrix["topk_energy_ratio"])
        if ratio > 1.0:
            raise ValueError(
                f"group_id={group_id} matrix={matrix_name} has energy ratio {ratio}"
            )
    if row.get("gradient_parameter_scope") != "transformer_2d":
        raise ValueError(f"group_id={group_id} did not record QKVO and FFN")
    score_scope = row.get("svd_score_scope")
    if score_scope not in SCORE_SCOPES:
        raise ValueError(f"group_id={group_id} has invalid SVD score scope")
    score_record = row.get("effective_rank_topk")
    if not isinstance(score_record, dict):
        raise ValueError(f"group_id={group_id} has no effective-rank record")
    if score_record.get("score_scope") != score_scope:
        raise ValueError(f"group_id={group_id} has inconsistent score scope")
    family_sums = score_record.get("per_family_layer_sum")
    if not isinstance(family_sums, dict):
        raise ValueError(f"group_id={group_id} has no family score sums")
    expected_families = set(QKVO_FAMILIES + FFN_FAMILIES)
    if set(family_sums) != expected_families:
        raise ValueError(f"group_id={group_id} did not record all seven families")
    scores_by_scope = score_record.get("scores_by_scope")
    if not isinstance(scores_by_scope, dict):
        raise ValueError(f"group_id={group_id} has no per-scope scores")
    expected_scores = {
        "qkvo_only": math.fsum(float(family_sums[key]) for key in QKVO_FAMILIES),
        "ffn_only": math.fsum(float(family_sums[key]) for key in FFN_FAMILIES),
        "transformer_2d": math.fsum(
            float(family_sums[key]) for key in QKVO_FAMILIES + FFN_FAMILIES
        ),
    }
    for scope, expected_score in expected_scores.items():
        try:
            actual_score = float(scores_by_scope[scope])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"group_id={group_id} has invalid {scope} score") from error
        if not math.isfinite(actual_score) or not math.isclose(
            actual_score, expected_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"group_id={group_id} has inconsistent {scope} score")
    selected_score = float(score_record["s"])
    if not math.isfinite(selected_score) or not math.isclose(
        selected_score, expected_scores[score_scope], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"group_id={group_id} selected the wrong SVD score")
    stats = row.get("group_stats")
    if not isinstance(stats, dict):
        raise ValueError(f"group_id={group_id} has no group_stats")
    if stats.get("svd_score_scope") != score_scope:
        raise ValueError(f"group_id={group_id} has inconsistent group score scope")
    for key in GROUP_LIST_KEYS:
        if not isinstance(stats.get(key), list) or len(stats[key]) != rollout_n:
            raise ValueError(
                f"group_id={group_id} group_stats.{key} must contain {rollout_n} values"
            )
    legacy_total = stats.pop("question_tokens", None)
    if "total_response_tokens" not in stats and legacy_total is not None:
        stats["total_response_tokens"] = legacy_total
    elif legacy_total is not None and int(legacy_total) != int(
        stats["total_response_tokens"]
    ):
        raise ValueError(f"group_id={group_id} has conflicting response token totals")
    if int(stats.get("total_response_tokens", -1)) != sum(
        int(value) for value in stats["response_tokens"]
    ):
        raise ValueError(
            f"group_id={group_id} has inconsistent total_response_tokens"
        )
    if stats.get("advantage_estimator") != "grpo":
        raise ValueError(f"group_id={group_id} did not use GRPO advantages")
    if stats.get("norm_adv_by_std_in_grpo") is not True:
        raise ValueError(f"group_id={group_id} did not normalize GRPO advantages")
    if not math.isclose(
        float(stats.get("advantage_epsilon", math.nan)),
        1e-6,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"group_id={group_id} has the wrong advantage epsilon")
    if int(stats.get("advantage_std_correction", -1)) != 1:
        raise ValueError(f"group_id={group_id} has the wrong std correction")
    rewards = [float(value) for value in stats["rewards"]]
    if len(rewards) == 1:
        expected_mean = 0.0
        expected_std = 1.0
    else:
        expected_mean = sum(rewards) / len(rewards)
        expected_std = math.sqrt(
            sum((value - expected_mean) ** 2 for value in rewards)
            / (len(rewards) - 1)
        )
    reward_mean = float(stats.get("reward_mean", math.nan))
    reward_std = float(stats.get("reward_std", math.nan))
    if not math.isclose(reward_mean, expected_mean, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"group_id={group_id} has an inconsistent reward mean")
    if not math.isclose(reward_std, expected_std, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"group_id={group_id} has an inconsistent reward std")
    for reward, advantage in zip(rewards, stats["advantages"]):
        expected_advantage = (reward - reward_mean) / (reward_std + 1e-6)
        if not math.isclose(
            float(advantage), expected_advantage, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"group_id={group_id} has inconsistent advantages")
    for index in range(rollout_n):
        prompt_tokens = int(stats["prompt_tokens"][index])
        generated_tokens = int(stats["generated_response_tokens"][index])
        original_tokens = int(stats["original_input_tokens"][index])
        input_tokens = int(stats["input_tokens"][index])
        response_tokens = int(stats["response_tokens"][index])
        if original_tokens != prompt_tokens + generated_tokens:
            raise ValueError(
                f"group_id={group_id} response={index} has inconsistent token totals"
            )
        if input_tokens > original_tokens or response_tokens > generated_tokens:
            raise ValueError(
                f"group_id={group_id} response={index} has impossible token counts"
            )
    if int(stats.get("analysis_truncated_responses", -1)) != sum(
        bool(value) for value in stats["analysis_truncated_flags"]
    ):
        raise ValueError(
            f"group_id={group_id} has inconsistent analysis truncation counts"
        )
    if int(stats.get("generation_hit_max_tokens_responses", -1)) != sum(
        bool(value) for value in stats["generation_hit_max_tokens_flags"]
    ):
        raise ValueError(
            f"group_id={group_id} has inconsistent generation length counts"
        )
    if stats.get("token_input_source") != "inference_token_ids":
        raise ValueError(f"group_id={group_id} did not use inference token IDs")
    if int(stats.get("group_id", -1)) != group_id or int(
        stats.get("worker_rank", -1)
    ) != rank:
        raise ValueError(f"group_id={group_id} has inconsistent group_stats identity")
    score = float(row["effective_rank_topk"]["s"])
    if not math.isclose(float(stats.get("svd_score", math.nan)), score, rel_tol=1e-12):
        raise ValueError(f"group_id={group_id} has inconsistent SVD scores")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent SVD worker outputs")
    parser.add_argument("--responses_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--analysis_signature", required=True)
    parser.add_argument("--group_stats_path", default=None)
    args = parser.parse_args()

    with open(args.manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_groups = {int(value) for value in manifest["all_group_ids"]}
    all_rows: Dict[int, Dict[str, Any]] = {}
    for rank in range(args.world_size):
        worker_path = os.path.join(args.responses_dir, f"svd_rank_{rank}.jsonl")
        if not os.path.isfile(worker_path):
            raise FileNotFoundError(f"Missing worker output: {worker_path}")
        assigned_groups = {
            int(value) for value in manifest["workers"][str(rank)]["group_ids"]
        }
        worker_groups = set()
        for row in read_rows(worker_path):
            if row.get("analysis_backend") != "independent":
                raise ValueError(f"Non-independent row found in {worker_path}")
            if row.get("analysis_signature") != args.analysis_signature:
                raise ValueError(f"Configuration mismatch in {worker_path}")
            validate_record(row, rank, int(manifest["rollout_n"]))
            group_id = int(row["group_id"])
            if group_id not in assigned_groups:
                raise ValueError(f"rank {rank} contains unassigned group_id={group_id}")
            if group_id in worker_groups:
                raise ValueError(f"Duplicate group_id={group_id} in {worker_path}")
            if group_id in all_rows:
                raise ValueError(f"Duplicate group_id={group_id} across workers")
            worker_groups.add(group_id)
            all_rows[group_id] = row
        missing_worker_groups = assigned_groups.difference(worker_groups)
        if missing_worker_groups:
            raise ValueError(
                f"rank {rank} is missing {len(missing_worker_groups)} assigned groups"
            )

    actual_groups = set(all_rows)
    missing = expected_groups.difference(actual_groups)
    unexpected = actual_groups.difference(expected_groups)
    if missing or unexpected or len(all_rows) != len(expected_groups):
        raise ValueError(
            f"Merge validation failed: expected={len(expected_groups)}, "
            f"actual={len(all_rows)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.isfile(output_path) and not existing_matches_signature(
        output_path, args.analysis_signature
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_path}.pre_independent_{timestamp}.bak"
        shutil.copy2(output_path, backup_path)
        print(f"Backed up previous final output to {backup_path}")
    temp_path = f"{output_path}.independent.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for group_id in sorted(all_rows):
            handle.write(
                json.dumps(all_rows[group_id], ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)
    group_stats_path = os.path.abspath(
        os.path.expanduser(
            args.group_stats_path
            or os.path.join(args.responses_dir, "group_stats.jsonl")
        )
    )
    os.makedirs(os.path.dirname(group_stats_path), exist_ok=True)
    group_stats_temp = f"{group_stats_path}.independent.tmp"
    with open(group_stats_temp, "w", encoding="utf-8") as handle:
        for group_id in sorted(all_rows):
            stats = dict(all_rows[group_id]["group_stats"])
            stats["analysis_signature"] = args.analysis_signature
            handle.write(json.dumps(stats, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(group_stats_temp, group_stats_path)
    print(
        f"Merged {len(all_rows)} unique groups from {args.world_size} workers "
        f"into {output_path}"
    )
    print(f"Wrote {len(all_rows)} per-group summaries to {group_stats_path}")


if __name__ == "__main__":
    main()
