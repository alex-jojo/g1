"""Shared gradient-SVD scoring utilities."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


PROJECTION_LABELS = {
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "o_proj": "O",
}

MLP_PROJECTION_LABELS = {
    "gate_proj": "GATE",
    "up_proj": "UP",
    "down_proj": "DOWN",
}
TRANSFORMER_MATRIX_LABELS = tuple(PROJECTION_LABELS.values()) + tuple(
    MLP_PROJECTION_LABELS.values()
)
PARAMETER_SCOPES = ("qkvo_only", "transformer_2d")

_ATTENTION_WEIGHT_PATTERN = re.compile(
    r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\."
    r".*?self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"
)
_MLP_WEIGHT_PATTERN = re.compile(
    r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\."
    r".*?mlp\.(gate_proj|up_proj|down_proj)\.weight$"
)


def qkvo_parameter_metadata(name: str) -> Optional[Tuple[int, str]]:
    """Return ``(layer_index, projection_label)`` for a Q/K/V/O weight."""
    match = _ATTENTION_WEIGHT_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1)), PROJECTION_LABELS[match.group(2)]


def is_qkvo_parameter(name: str, shape: Iterable[int]) -> bool:
    """Return whether ``name`` is a two-dimensional Q/K/V/O weight."""
    return len(tuple(shape)) == 2 and qkvo_parameter_metadata(name) is not None


def transformer_parameter_metadata(name: str) -> Optional[Tuple[int, str]]:
    """Return layer/family metadata for QKVO and gated-MLP matrix weights."""
    attention_metadata = qkvo_parameter_metadata(name)
    if attention_metadata is not None:
        return attention_metadata
    match = _MLP_WEIGHT_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1)), MLP_PROJECTION_LABELS[match.group(2)]


def is_matrix_parameter(
    name: str,
    shape: Iterable[int],
    parameter_scope: str,
) -> bool:
    """Return whether a parameter belongs to the requested matrix scope."""
    if parameter_scope not in PARAMETER_SCOPES:
        raise ValueError(f"Unsupported gradient parameter scope: {parameter_scope}")
    if len(tuple(shape)) != 2:
        return False
    if parameter_scope == "qkvo_only":
        return qkvo_parameter_metadata(name) is not None
    return transformer_parameter_metadata(name) is not None


def effective_rank(singular_values: Iterable[float]) -> float:
    """Compute ``exp(entropy)`` from a non-negative singular-value spectrum."""
    values = [float(value) for value in singular_values]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Singular values must be finite and non-negative")
    value_sum = math.fsum(values)
    if value_sum == 0.0:
        return 0.0
    probabilities = [value / value_sum for value in values if value > 0.0]
    entropy = -math.fsum(
        probability * math.log(probability) for probability in probabilities
    )
    return math.exp(entropy)


def truncated_svd(
    matrix: torch.Tensor,
    svd_rank: int,
    oversample: int = 8,
    niter: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute the same deterministic randomized top-k SVD used by FSDP mode."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, found shape {tuple(matrix.shape)}")
    if svd_rank <= 0 or oversample < 0 or niter < 0:
        raise ValueError("Invalid SVD rank/oversampling/iteration configuration")

    original_shape = tuple(int(value) for value in matrix.shape)
    target_rank = min(svd_rank, min(original_shape))
    lowrank_q = min(target_rank + oversample, min(original_shape))
    matrix_fp32 = matrix.detach().to(dtype=torch.float32)
    frobenius_norm_tensor = torch.linalg.vector_norm(matrix_fp32)
    rng_devices = [matrix_fp32.device.index] if matrix_fp32.is_cuda else []
    with torch.random.fork_rng(devices=rng_devices):
        if matrix_fp32.is_cuda:
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        left, singular_values, right = torch.svd_lowrank(
            matrix_fp32,
            q=lowrank_q,
            niter=niter,
        )
    singular_values = torch.sort(singular_values, descending=True).values[:target_rank]
    singular_values_cpu = singular_values.cpu()
    frobenius_norm = float(frobenius_norm_tensor.item())
    spectral_norm = (
        float(singular_values_cpu[0].item()) if target_rank > 0 else 0.0
    )
    topk_energy = float(torch.sum(singular_values_cpu.square()).item())
    frobenius_energy = frobenius_norm * frobenius_norm
    if frobenius_energy > 0.0:
        topk_energy_ratio = min(1.0, max(0.0, topk_energy / frobenius_energy))
    else:
        topk_energy_ratio = 0.0
    stable_rank = (
        frobenius_energy / (spectral_norm * spectral_norm)
        if spectral_norm > 0.0
        else 0.0
    )
    result = {
        "shape": list(original_shape),
        "num_singular_values": target_rank,
        "singular_values": singular_values_cpu.tolist(),
        "topk_singular_value_sum": float(singular_values_cpu.sum().item()),
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "stable_rank": stable_rank,
        "topk_energy_ratio": topk_energy_ratio,
    }
    del (
        left,
        right,
        matrix_fp32,
        frobenius_norm_tensor,
        singular_values,
        singular_values_cpu,
    )
    return result


def zero_svd_result(shape: Iterable[int], svd_rank: int) -> Dict[str, Any]:
    """Build an exact zero spectrum without running a randomized SVD."""
    original_shape = tuple(int(value) for value in shape)
    target_rank = min(svd_rank, min(original_shape))
    return {
        "shape": list(original_shape),
        "num_singular_values": target_rank,
        "singular_values": [0.0] * target_rank,
        "topk_singular_value_sum": 0.0,
        "frobenius_norm": 0.0,
        "spectral_norm": 0.0,
        "stable_rank": 0.0,
        "topk_energy_ratio": 0.0,
    }


def aggregate_qkvo_effective_rank(
    matrix_results: Dict[str, Dict[str, Any]],
    svd_rank: int,
) -> Dict[str, Any]:
    """Reproduce the existing sum-over-layers, equal-sum-over-QKVO score."""
    per_projection_layers: Dict[str, Dict[int, float]] = {
        label: {} for label in PROJECTION_LABELS.values()
    }
    for parameter_name, matrix_result in matrix_results.items():
        metadata = qkvo_parameter_metadata(parameter_name)
        if metadata is None:
            continue
        layer_index, projection_label = metadata
        if layer_index in per_projection_layers[projection_label]:
            raise RuntimeError(
                f"Duplicate {projection_label} projection gradient for layer {layer_index}"
            )
        rank_value = effective_rank(matrix_result["singular_values"])
        matrix_result["topk_effective_rank"] = rank_value
        per_projection_layers[projection_label][layer_index] = rank_value

    missing = [
        label for label, layer_values in per_projection_layers.items() if not layer_values
    ]
    if missing:
        raise RuntimeError(
            "Cannot compute effective-rank score: no layer matrices found for "
            + ", ".join(missing)
        )
    layer_counts = {
        label: len(layer_values)
        for label, layer_values in per_projection_layers.items()
    }
    if len(set(layer_counts.values())) != 1:
        raise RuntimeError(
            "Cannot compute effective-rank score with unequal Q/K/V/O layer counts: "
            f"{layer_counts}"
        )
    projection_sums = {
        label: math.fsum(layer_values.values())
        for label, layer_values in per_projection_layers.items()
    }
    return {
        "k": svd_rank,
        "zero_spectrum_effective_rank": 0.0,
        "aggregation": "sum_layers_then_equal_sum_qkvo",
        "per_projection_layer_count": layer_counts,
        "per_projection_layer_sum": projection_sums,
        "s": math.fsum(projection_sums.values()),
    }


def aggregate_transformer_effective_rank(
    matrix_results: Dict[str, Dict[str, Any]],
    svd_rank: int,
) -> Dict[str, Any]:
    """Sum effective rank over QKVO and gate/up/down in every layer."""
    per_family_layers: Dict[str, Dict[int, float]] = {
        label: {} for label in TRANSFORMER_MATRIX_LABELS
    }
    for parameter_name, matrix_result in matrix_results.items():
        metadata = transformer_parameter_metadata(parameter_name)
        if metadata is None:
            raise RuntimeError(
                f"Unexpected matrix in transformer_2d scope: {parameter_name}"
            )
        layer_index, family_label = metadata
        if layer_index in per_family_layers[family_label]:
            raise RuntimeError(
                f"Duplicate {family_label} gradient for layer {layer_index}"
            )
        rank_value = effective_rank(matrix_result["singular_values"])
        matrix_result["topk_effective_rank"] = rank_value
        per_family_layers[family_label][layer_index] = rank_value

    family_layer_sets = {
        label: set(layer_values)
        for label, layer_values in per_family_layers.items()
    }
    missing = [label for label, layers in family_layer_sets.items() if not layers]
    if missing:
        raise RuntimeError(
            "Cannot compute transformer effective-rank score: no matrices found for "
            + ", ".join(missing)
        )
    reference_layers = family_layer_sets[TRANSFORMER_MATRIX_LABELS[0]]
    inconsistent = {
        label: sorted(layers)
        for label, layers in family_layer_sets.items()
        if layers != reference_layers
    }
    if inconsistent:
        raise RuntimeError(
            "Transformer matrix families do not cover identical layers: "
            f"{inconsistent}"
        )

    family_counts = {
        label: len(per_family_layers[label]) for label in TRANSFORMER_MATRIX_LABELS
    }
    family_sums = {
        label: math.fsum(per_family_layers[label].values())
        for label in TRANSFORMER_MATRIX_LABELS
    }
    layer_counts = {
        str(layer): len(TRANSFORMER_MATRIX_LABELS)
        for layer in sorted(reference_layers)
    }
    layer_sums = {
        str(layer): math.fsum(
            per_family_layers[label][layer]
            for label in TRANSFORMER_MATRIX_LABELS
        )
        for layer in sorted(reference_layers)
    }
    return {
        "k": svd_rank,
        "zero_spectrum_effective_rank": 0.0,
        "aggregation": "sum_layers_then_equal_sum_qkvo_gate_up_down",
        "matrix_count": len(matrix_results),
        "per_family_layer_count": family_counts,
        "per_family_layer_sum": family_sums,
        "per_layer_matrix_count": layer_counts,
        "per_layer_matrix_sum": layer_sums,
        "s": math.fsum(family_sums.values()),
    }


def aggregate_effective_rank(
    matrix_results: Dict[str, Dict[str, Any]],
    svd_rank: int,
    parameter_scope: str,
) -> Dict[str, Any]:
    """Aggregate per-matrix effective ranks according to the configured scope."""
    if parameter_scope == "qkvo_only":
        return aggregate_qkvo_effective_rank(matrix_results, svd_rank)
    if parameter_scope == "transformer_2d":
        return aggregate_transformer_effective_rank(matrix_results, svd_rank)
    raise ValueError(f"Unsupported gradient parameter scope: {parameter_scope}")
