#!/usr/bin/env python3
"""Compute and cache U0/V0 for the fixed initial backbone exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from reference_basis import FORMAT_NAME


SELECT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(SELECT_DIR) not in sys.path:
    sys.path.insert(0, str(SELECT_DIR))

from svd_utils import is_matrix_parameter, truncated_svd_factors  # noqa: E402


def model_artifacts(model_path: str) -> list[dict[str, Any]]:
    records = []
    for name in sorted(os.listdir(model_path)):
        if not (
            name.endswith(".safetensors")
            or name.endswith(".bin")
            or name in {"config.json", "model.safetensors.index.json"}
        ):
            continue
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            stat = os.stat(path)
            records.append(
                {"name": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
    if not records:
        raise ValueError(f"No model artifacts found under {model_path}")
    return records


def signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: str, payload: dict[str, Any]) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare initial-backbone U0/V0")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--svd_rank", type=int, default=128)
    parser.add_argument("--svd_oversample", type=int, default=8)
    parser.add_argument("--svd_niter", type=int, default=2)
    parser.add_argument("--svd_seed", type=int, default=0)
    parser.add_argument(
        "--gradient_parameter_scope",
        choices=["qkvo_only", "transformer_2d"],
        default="transformer_2d",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    if not os.path.isdir(model_path):
        parser.error(f"Reference model directory not found: {model_path}")
    if args.svd_rank <= 0:
        parser.error("--svd_rank must be positive")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manifest_path = f"{output_path}.manifest.json"
    config = {
        "format": FORMAT_NAME,
        "model_path": model_path,
        "model_artifacts": model_artifacts(model_path),
        "rank": args.svd_rank,
        "oversample": args.svd_oversample,
        "niter": args.svd_niter,
        "seed": args.svd_seed,
        "parameter_scope": args.gradient_parameter_scope,
        "storage_dtype": "float32",
    }
    config_signature = signature(config)
    previous = read_json(manifest_path)
    if (
        os.path.isfile(output_path)
        and previous.get("config_signature") == config_signature
        and int(previous.get("cache_size", -1)) == os.path.getsize(output_path)
    ):
        print(f"Reference U0/V0 cache is compatible; reusing {output_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    parameters: dict[str, dict[str, Any]] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not is_matrix_parameter(
            name, parameter.shape, args.gradient_parameter_scope
        ):
            continue
        u, singular_values, v, _ = truncated_svd_factors(
            parameter,
            svd_rank=args.svd_rank,
            oversample=args.svd_oversample,
            niter=args.svd_niter,
            seed=args.svd_seed,
        )
        parameters[name] = {
            "shape": list(parameter.shape),
            "u": u.detach().cpu(),
            "singular_values": singular_values.detach().cpu(),
            "v": v.detach().cpu(),
        }
        del u, singular_values, v
        print(f"Prepared U0/V0: {name}", flush=True)
    if not parameters:
        raise RuntimeError("No transformer matrices found in the reference model")

    payload = {
        "format": FORMAT_NAME,
        "model_path": model_path,
        "rank": args.svd_rank,
        "parameter_scope": args.gradient_parameter_scope,
        "config_signature": config_signature,
        "parameters": parameters,
    }
    temporary = f"{output_path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    manifest = dict(config)
    manifest.update(
        {
            "config_signature": config_signature,
            "parameter_count": len(parameters),
            "cache_path": output_path,
            "cache_size": os.path.getsize(output_path),
        }
    )
    atomic_json(manifest_path, manifest)
    print(
        f"Saved {len(parameters)} initial-backbone U0/V0 bases to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
