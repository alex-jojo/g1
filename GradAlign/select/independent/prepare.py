#!/usr/bin/env python3
"""Load exact rollout token IDs and assign complete groups to SVD workers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import heapq
import json
import math
import os
import zlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


TOKEN_SCHEMA_VERSION = 1
TOKEN_IDS_ENCODING = "zlib+base64+uint32-le"
PARTITION_SCHEMA_VERSION = 3


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def find_responses_file(responses_dir: str) -> str:
    for filename in (
        "responses_sorted.json",
        "response_sorted.jsonl",
        "responses_sorted.jsonl",
    ):
        path = os.path.join(responses_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No sorted responses file found under {responses_dir}")


def read_responses(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def reward_value(record: Dict[str, Any]) -> float:
    value = record.get("reward_score")
    if value is None:
        passed = record.get("passed")
        if passed is None:
            passed = record.get("is_correct", False)
        value = 1.0 if bool(passed) else 0.0
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite reward for group_id={record.get('group_id')}: {value}")
    return value


def _decode_token_ids(record: Dict[str, Any], prefix: str) -> np.ndarray:
    """Decode stored IDs; never reconstruct them from prompt/response text."""
    record_identity = (
        f"group_id={record.get('group_id')}, sample_id={record.get('sample_id')}"
    )
    schema_version = record.get("token_schema_version")
    if schema_version != TOKEN_SCHEMA_VERSION:
        raise ValueError(
            f"{record_identity} has token_schema_version={schema_version!r}; "
            "this rollout must be regenerated with the token-preserving inference code"
        )

    encoded_key = f"{prefix}_token_ids_b64"
    list_key = f"{prefix}_token_ids"
    if encoded_key in record:
        if record.get("token_ids_encoding") != TOKEN_IDS_ENCODING:
            raise ValueError(
                f"{record_identity} uses unsupported token encoding "
                f"{record.get('token_ids_encoding')!r}"
            )
        try:
            compressed = base64.b64decode(record[encoded_key], validate=True)
            raw = zlib.decompress(compressed)
        except (TypeError, ValueError, zlib.error) as error:
            raise ValueError(
                f"{record_identity} has invalid {encoded_key}"
            ) from error
        if len(raw) % np.dtype("<u4").itemsize:
            raise ValueError(
                f"{record_identity} has a misaligned uint32 token payload"
            )
        token_ids_u32 = np.frombuffer(raw, dtype="<u4")
        if token_ids_u32.size and int(token_ids_u32.max()) > np.iinfo(np.int32).max:
            raise ValueError(f"{record_identity} has a token ID outside int32 range")
        token_ids = token_ids_u32.astype(np.int32, copy=True)
    elif list_key in record:
        token_ids_i64 = np.asarray(record[list_key], dtype=np.int64).reshape(-1)
        if token_ids_i64.size and (
            int(token_ids_i64.min()) < 0
            or int(token_ids_i64.max()) > np.iinfo(np.int32).max
        ):
            raise ValueError(f"{record_identity} has a token ID outside int32 range")
        token_ids = token_ids_i64.astype(np.int32, copy=False)
    else:
        raise ValueError(
            f"{record_identity} has no stored {prefix} token IDs; "
            "old text-only rollouts must be regenerated"
        )

    expected_count = record.get(f"{prefix}_token_count")
    if expected_count is None or int(expected_count) != int(token_ids.size):
        raise ValueError(
            f"{record_identity} has inconsistent {prefix} token count: "
            f"metadata={expected_count}, decoded={token_ids.size}"
        )
    return token_ids


def _prepare_record(
    index_and_record: Tuple[int, Dict[str, Any]], max_length: int
) -> Dict[str, Any]:
    global_index, record = index_and_record
    prompt_input_ids = _decode_token_ids(record, "prompt")
    response_input_ids = _decode_token_ids(record, "response")
    requested_max_tokens = int(record.get("requested_max_tokens", -1))
    generation_hit_max_tokens = bool(
        record.get("generation_hit_max_tokens", False)
    )
    if requested_max_tokens <= 0:
        raise ValueError(
            f"group_id={record.get('group_id')}, sample_id={record.get('sample_id')} "
            "has no positive requested_max_tokens"
        )
    if int(response_input_ids.size) > requested_max_tokens:
        raise ValueError(
            f"group_id={record.get('group_id')}, sample_id={record.get('sample_id')} "
            "contains more generated IDs than requested_max_tokens"
        )
    if generation_hit_max_tokens != (
        int(response_input_ids.size) == requested_max_tokens
    ):
        raise ValueError(
            f"group_id={record.get('group_id')}, sample_id={record.get('sample_id')} "
            "has an inconsistent generation_hit_max_tokens flag"
        )
    full_input_ids = np.concatenate((prompt_input_ids, response_input_ids))
    original_input_length = int(full_input_ids.shape[0])
    was_truncated = original_input_length > int(max_length)
    input_ids = full_input_ids[:max_length]
    attention_mask = np.ones(int(input_ids.shape[0]), dtype=np.uint8)
    prompt_length = int(prompt_input_ids.shape[0])
    prompt_length = min(prompt_length, int(input_ids.shape[0]))
    full_response_mask = np.zeros(int(input_ids.shape[0]), dtype=np.uint8)
    full_response_mask[prompt_length:] = 1
    response_mask = full_response_mask[1:]
    finish_reason = record.get("generation_finish_reason")
    stop_reason = record.get("generation_stop_reason")
    return {
        "input_ids": input_ids.astype(np.int32, copy=False),
        "attention_mask": attention_mask.astype(np.uint8, copy=False),
        "response_mask": response_mask,
        "original_input_length": original_input_length,
        "was_truncated": was_truncated,
        "prompt_length": int(prompt_input_ids.shape[0]),
        "generated_response_length": int(response_input_ids.shape[0]),
        "generation_hit_max_tokens": generation_hit_max_tokens,
        "generation_finish_reason": "" if finish_reason is None else str(finish_reason),
        "generation_stop_reason": "" if stop_reason is None else str(stop_reason),
        "requested_max_tokens": requested_max_tokens,
        "group_id": int(record["group_id"]),
        "global_index": int(global_index),
        "reward_score": reward_value(record),
    }


def greedy_assign_groups(
    grouped_entries: Dict[int, List[Dict[str, Any]]],
    world_size: int,
) -> Tuple[List[List[int]], List[int]]:
    """Longest-processing-time greedy assignment using post-truncation tokens."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    group_costs = {
        group_id: sum(len(entry["input_ids"]) for entry in entries)
        for group_id, entries in grouped_entries.items()
    }
    worker_groups: List[List[int]] = [[] for _ in range(world_size)]
    worker_costs = [0] * world_size
    heap = [(0, rank) for rank in range(world_size)]
    heapq.heapify(heap)
    for group_id, cost in sorted(
        group_costs.items(), key=lambda item: (-item[1], item[0])
    ):
        current_cost, rank = heapq.heappop(heap)
        worker_groups[rank].append(group_id)
        worker_costs[rank] = current_cost + cost
        heapq.heappush(heap, (worker_costs[rank], rank))
    return worker_groups, worker_costs


def _object_array(values: Iterable[np.ndarray]) -> np.ndarray:
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def save_shard(path: str, entries: List[Dict[str, Any]]) -> None:
    temp_path = f"{path}.tmp.npz"
    np.savez_compressed(
        temp_path,
        input_ids=_object_array(entry["input_ids"] for entry in entries),
        attention_mask=_object_array(entry["attention_mask"] for entry in entries),
        response_mask=_object_array(entry["response_mask"] for entry in entries),
        original_input_length=np.asarray(
            [entry["original_input_length"] for entry in entries], dtype=np.int32
        ),
        was_truncated=np.asarray(
            [entry["was_truncated"] for entry in entries], dtype=np.bool_
        ),
        prompt_length=np.asarray(
            [entry["prompt_length"] for entry in entries], dtype=np.int32
        ),
        generated_response_length=np.asarray(
            [entry["generated_response_length"] for entry in entries], dtype=np.int32
        ),
        generation_hit_max_tokens=np.asarray(
            [entry["generation_hit_max_tokens"] for entry in entries], dtype=np.bool_
        ),
        generation_finish_reason=np.asarray(
            [entry["generation_finish_reason"] for entry in entries], dtype=object
        ),
        generation_stop_reason=np.asarray(
            [entry["generation_stop_reason"] for entry in entries], dtype=object
        ),
        requested_max_tokens=np.asarray(
            [entry["requested_max_tokens"] for entry in entries], dtype=np.int32
        ),
        group_id=np.asarray([entry["group_id"] for entry in entries], dtype=np.int64),
        global_index=np.asarray(
            [entry["global_index"] for entry in entries], dtype=np.int64
        ),
        reward_score=np.asarray(
            [entry["reward_score"] for entry in entries], dtype=np.float32
        ),
    )
    os.replace(temp_path, path)


def _manifest_is_reusable(
    manifest_path: str,
    expected: Dict[str, Any],
    world_size: int,
) -> bool:
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    output_dir = os.path.dirname(manifest_path)
    return all(
        os.path.isfile(os.path.join(output_dir, f"data_rank_{rank}.npz"))
        for rank in range(world_size)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare whole-group shards for independent single-GPU SVD workers"
    )
    parser.add_argument("--responses_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=5120)
    parser.add_argument("--expected_groups", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Deprecated compatibility option; stored-ID decoding is local and fast",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.world_size <= 0 or args.rollout_n <= 0 or args.max_length <= 1:
        parser.error("world_size/rollout_n must be positive and max_length must exceed 1")
    if args.num_workers <= 0:
        parser.error("num_workers must be positive")

    responses_dir = os.path.abspath(os.path.expanduser(args.responses_dir))
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    responses_file = find_responses_file(responses_dir)
    responses_sha256 = sha256_file(responses_file)
    output_dir = os.path.join(responses_dir, "data_independent")
    manifest_path = os.path.join(output_dir, "partition_manifest.json")
    expected_manifest_fields = {
        "partition_schema_version": PARTITION_SCHEMA_VERSION,
        "analysis_backend": "independent",
        "partition_strategy": "group_token_greedy",
        "token_input_source": "inference_token_ids",
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_ids_encoding": TOKEN_IDS_ENCODING,
        "world_size": args.world_size,
        "rollout_n": args.rollout_n,
        "max_length": args.max_length,
        "responses_file": responses_file,
        "responses_sha256": responses_sha256,
        "model_path": model_path,
    }
    if not args.force and _manifest_is_reusable(
        manifest_path, expected_manifest_fields, args.world_size
    ):
        print(f"Reusing validated independent shards under {output_dir}")
        return
    if os.path.exists(manifest_path) and not args.force:
        raise RuntimeError(
            f"Existing independent shards are stale or incomplete: {output_dir}; "
            "rerun with --force after confirming they may be replaced"
        )

    records = read_responses(responses_file)
    group_counts = Counter(int(record["group_id"]) for record in records)
    bad_counts = {
        group_id: count
        for group_id, count in group_counts.items()
        if count != args.rollout_n
    }
    if bad_counts:
        preview = list(sorted(bad_counts.items()))[:20]
        raise ValueError(
            f"Expected exactly {args.rollout_n} responses per group; bad groups: {preview}"
        )
    if args.expected_groups is not None and len(group_counts) != args.expected_groups:
        raise ValueError(
            f"Expected {args.expected_groups} groups, found {len(group_counts)}"
        )
    expected_responses = len(group_counts) * args.rollout_n
    if len(records) != expected_responses:
        raise ValueError(f"Expected {expected_responses} responses, found {len(records)}")

    print(
        f"Loading exact stored token IDs for {len(records)} responses from "
        f"{len(group_counts)} groups (no text re-tokenization)"
    )
    entries = [
        _prepare_record(item, args.max_length) for item in enumerate(records)
    ]
    entries.sort(key=lambda entry: entry["global_index"])
    global_indices = [entry["global_index"] for entry in entries]
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("global_index values are not unique")

    grouped_entries: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped_entries[entry["group_id"]].append(entry)
    for group_id, group_entries in grouped_entries.items():
        group_entries.sort(key=lambda entry: entry["global_index"])
        if len(group_entries) != args.rollout_n:
            raise AssertionError(f"group_id={group_id} was corrupted during tokenization")

    worker_groups, worker_costs = greedy_assign_groups(
        grouped_entries, args.world_size
    )
    os.makedirs(output_dir, exist_ok=True)
    worker_manifest: Dict[str, Any] = {}
    seen_groups = set()
    for rank, assigned_groups in enumerate(worker_groups):
        if seen_groups.intersection(assigned_groups):
            raise AssertionError("Greedy partition produced duplicate group assignments")
        seen_groups.update(assigned_groups)
        shard_entries = [
            entry
            for group_id in assigned_groups
            for entry in grouped_entries[group_id]
        ]
        shard_path = os.path.join(output_dir, f"data_rank_{rank}.npz")
        save_shard(shard_path, shard_entries)
        valid_response_tokens = sum(
            int(entry["response_mask"].sum()) for entry in shard_entries
        )
        worker_manifest[str(rank)] = {
            "group_ids": assigned_groups,
            "group_count": len(assigned_groups),
            "response_count": len(shard_entries),
            "input_tokens": worker_costs[rank],
            "valid_response_tokens": valid_response_tokens,
            "shard_path": shard_path,
        }
        print(
            f"rank {rank}: groups={len(assigned_groups)}, "
            f"responses={len(shard_entries)}, input_tokens={worker_costs[rank]}"
        )
    if seen_groups != set(grouped_entries):
        raise AssertionError("Greedy partition has missing groups")

    signature_payload = {
        **expected_manifest_fields,
        "all_group_ids": sorted(seen_groups),
    }
    partition_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        **expected_manifest_fields,
        "analysis_minibatch_size": 1,
        "gradient_parameter_scope": "transformer_2d",
        "expected_groups": len(grouped_entries),
        "expected_responses": len(entries),
        "all_group_ids": sorted(seen_groups),
        "partition_signature": partition_signature,
        "workers": worker_manifest,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Saved independent partition manifest to {manifest_path}")


if __name__ == "__main__":
    main()
