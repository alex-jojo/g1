#!/usr/bin/env python3
"""Score generated responses with the same custom reward used by VERL."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import tempfile
from collections.abc import Callable
from typing import Any


REWARD_RESULT_KEYS = (
    "acc",
    "pred",
    "reward_verifier",
    "verification_status",
    "invalid",
    "verification_error",
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def response_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": row.get("data_source", ""),
        "group_id": row.get("group_id"),
        "sample_id": row.get("sample_id"),
        "prompt": row.get("prompt"),
        "msg": row.get("msg"),
        "response": row.get("response", ""),
        "expected_answer": row.get("expected_answer", ""),
    }


def response_input_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(response_input(row))).hexdigest()


def responses_content_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_json_bytes(response_input(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def load_reward_function(path: str, name: str) -> tuple[Callable[..., Any], str, str]:
    reward_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(reward_path):
        raise FileNotFoundError(f"Reward function does not exist: {reward_path}")

    reward_sha256 = file_sha256(reward_path)
    module_name = f"_gradalign_reward_{reward_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, reward_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reward module: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reward_function = getattr(module, name, None)
    if not callable(reward_function):
        raise AttributeError(f"Reward module {reward_path} has no callable {name!r}")
    return reward_function, reward_path, reward_sha256


def extract_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    original_data = row.get("original_data")
    if not isinstance(original_data, dict):
        return {}
    original_entry = original_data.get("original_entry")
    if isinstance(original_entry, dict):
        extra_info = original_entry.get("extra_info")
    else:
        extra_info = original_data.get("extra_info")
    return extra_info if isinstance(extra_info, dict) else {}


def resolve_reward_result(raw_result: Any) -> tuple[float, bool, dict[str, Any]]:
    if inspect.isawaitable(raw_result):
        raw_result = asyncio.run(raw_result)

    if isinstance(raw_result, dict):
        if "score" not in raw_result:
            raise ValueError("Custom reward returned a dict without a 'score' field")
        score = float(raw_result["score"])
        passed = bool(raw_result.get("acc", score > 0.0))
        details = {
            key: raw_result[key]
            for key in REWARD_RESULT_KEYS
            if key in raw_result
        }
    else:
        score = float(raw_result)
        passed = score > 0.0
        details = {}
    return score, passed, details


def atomic_write_json(path: str, value: Any, *, indent: int = 2) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp_reward_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent, default=str)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def score_responses(
    rows: list[dict[str, Any]],
    reward_function: Callable[..., Any],
    reward_path: str,
    reward_name: str,
    reward_sha256: str,
) -> int:
    changed = 0
    for index, row in enumerate(rows):
        input_sha256 = response_input_sha256(row)
        already_current = (
            row.get("reward_function_sha256") == reward_sha256
            and row.get("reward_function_name") == reward_name
            and row.get("reward_input_sha256") == input_sha256
            and "reward_score" in row
        )
        if already_current:
            continue

        raw_result = reward_function(
            data_source=str(row.get("data_source", "")),
            solution_str=str(row.get("response", "")),
            ground_truth=str(row.get("expected_answer", "")),
            extra_info=extract_extra_info(row),
        )
        score, passed, details = resolve_reward_result(raw_result)
        row["reward_score"] = score
        row["passed"] = passed
        row["is_correct"] = passed
        row["reward_function_path"] = reward_path
        row["reward_function_name"] = reward_name
        row["reward_function_sha256"] = reward_sha256
        row["reward_input_sha256"] = input_sha256
        row["reward_result"] = details
        changed += 1

        if (index + 1) % 1000 == 0:
            print(f"Scored {index + 1}/{len(rows)} responses")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a VERL-compatible custom reward to responses.json"
    )
    parser.add_argument("--responses_file", required=True)
    parser.add_argument("--reward_path", required=True)
    parser.add_argument("--reward_name", default="compute_score")
    parser.add_argument("--manifest_path", default=None)
    args = parser.parse_args()

    responses_file = os.path.abspath(os.path.expanduser(args.responses_file))
    if not os.path.isfile(responses_file):
        raise FileNotFoundError(f"Responses file does not exist: {responses_file}")
    with open(responses_file, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected a JSON list of response objects: {responses_file}")

    reward_function, reward_path, reward_sha256 = load_reward_function(
        args.reward_path, args.reward_name
    )
    changed = score_responses(
        rows,
        reward_function,
        reward_path,
        args.reward_name,
        reward_sha256,
    )
    content_sha256 = responses_content_sha256(rows)
    atomic_write_json(responses_file, rows)

    manifest_path = args.manifest_path or os.path.join(
        os.path.dirname(responses_file), "reward_scoring.json"
    )
    atomic_write_json(
        manifest_path,
        {
            "responses_file": responses_file,
            "response_count": len(rows),
            "response_content_sha256": content_sha256,
            "reward_path": reward_path,
            "reward_name": args.reward_name,
            "reward_sha256": reward_sha256,
        },
    )
    print(
        f"Custom reward applied: changed={changed}/{len(rows)}, "
        f"reward_sha256={reward_sha256}"
    )


if __name__ == "__main__":
    main()
