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
SCORE_SCOPES = ("qkvo_only", "ffn_only", "transformer_2d")
SCORE_SCOPE_FAMILIES = {
    "qkvo_only": tuple(PROJECTION_LABELS.values()),
    "ffn_only": tuple(MLP_PROJECTION_LABELS.values()),
    "transformer_2d": TRANSFORMER_MATRIX_LABELS,
}
SCORE_SCOPE_AGGREGATIONS = {
    "qkvo_only": "sum_layers_then_equal_sum_qkvo",
    "ffn_only": "sum_layers_then_equal_sum_gate_up_down",
    "transformer_2d": "sum_layers_then_equal_sum_qkvo_gate_up_down",
}

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
    left, singular_values, right, frobenius_norm = truncated_svd_factors(
        matrix,
        svd_rank=svd_rank,
        oversample=oversample,
        niter=niter,
        seed=seed,
    )
    result = summarize_svd(
        matrix.shape,
        singular_values,
        frobenius_norm=frobenius_norm,
    )
    del left, right, singular_values
    return result


def truncated_svd_factors(
    matrix: torch.Tensor,
    svd_rank: int,
    oversample: int = 8,
    niter: int = 2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return aligned, descending top-k ``U, S, V`` factors and ||matrix||_F."""
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
    order = torch.argsort(singular_values, descending=True)[:target_rank]
    left = left.index_select(1, order)
    singular_values = singular_values.index_select(0, order)
    right = right.index_select(1, order)
    frobenius_norm = float(frobenius_norm_tensor.item())
    del matrix_fp32, frobenius_norm_tensor, order
    return left, singular_values, right, frobenius_norm


def summarize_svd(
    shape: Iterable[int],
    singular_values: torch.Tensor,
    *,
    frobenius_norm: float,
) -> Dict[str, Any]:
    """Build the JSON-safe spectrum/statistics record from aligned SVD factors."""
    original_shape = tuple(int(value) for value in shape)
    singular_values_cpu = singular_values.detach().float().cpu()
    target_rank = int(singular_values_cpu.numel())
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
    return {
        "shape": list(original_shape),
        "num_singular_values": target_rank,
        "singular_values": singular_values_cpu.tolist(),
        "topk_singular_value_sum": float(singular_values_cpu.sum().item()),
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "stable_rank": stable_rank,
        "topk_energy_ratio": topk_energy_ratio,
    }


def truncated_svd_with_subspace(
    matrix: torch.Tensor,
    reference_u: torch.Tensor,
    reference_v: torch.Tensor,
    svd_rank: int,
    oversample: int = 8,
    niter: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute update SVD and the normalized U/V overlap with a fixed basis."""
    left, singular_values, right, frobenius_norm = truncated_svd_factors(
        matrix,
        svd_rank=svd_rank,
        oversample=oversample,
        niter=niter,
        seed=seed,
    )
    if reference_u.ndim != 2 or reference_v.ndim != 2:
        raise ValueError("Reference U and V must both be matrices")
    if reference_u.shape[0] != left.shape[0] or reference_v.shape[0] != right.shape[0]:
        raise ValueError(
            "Reference/update SVD shapes differ: "
            f"U {tuple(reference_u.shape)} vs {tuple(left.shape)}, "
            f"V {tuple(reference_v.shape)} vs {tuple(right.shape)}"
        )
    rank = min(
        int(left.shape[1]),
        int(right.shape[1]),
        int(reference_u.shape[1]),
        int(reference_v.shape[1]),
    )
    if rank <= 0:
        raise ValueError("Subspace rank must be positive")
    result = summarize_svd(
        matrix.shape,
        singular_values,
        frobenius_norm=frobenius_norm,
    )
    if frobenius_norm == 0.0:
        phi_u = 0.0
        phi_v = 0.0
    else:
        update_u = left[:, :rank]
        update_v = right[:, :rank]
        base_u = reference_u[:, :rank].to(
            device=update_u.device, dtype=update_u.dtype
        )
        base_v = reference_v[:, :rank].to(
            device=update_v.device, dtype=update_v.dtype
        )
        phi_u = float(
            (update_u.transpose(0, 1) @ base_u).square().sum().div(rank).item()
        )
        phi_v = float(
            (update_v.transpose(0, 1) @ base_v).square().sum().div(rank).item()
        )
        phi_u = min(1.0, max(0.0, phi_u))
        phi_v = min(1.0, max(0.0, phi_v))
    result.update(
        {
            "subspace_rank": rank,
            "subspace_phi_u": phi_u,
            "subspace_phi_v": phi_v,
        }
    )
    del left, singular_values, right
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
    score_scope: str = "transformer_2d",
) -> Dict[str, Any]:
    """Record all transformer matrices and select which families contribute to S."""
    if score_scope not in SCORE_SCOPES:
        raise ValueError(f"Unsupported SVD score scope: {score_scope}")

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
    scores_by_scope = {
        scope: math.fsum(family_sums[label] for label in families)
        for scope, families in SCORE_SCOPE_FAMILIES.items()
    }
    score_families = SCORE_SCOPE_FAMILIES[score_scope]
    score_layer_sums = {
        str(layer): math.fsum(per_family_layers[label][layer] for label in score_families)
        for layer in sorted(reference_layers)
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
        "score_scope": score_scope,
        "aggregation": SCORE_SCOPE_AGGREGATIONS[score_scope],
        "recorded_parameter_scope": "transformer_2d",
        "recorded_families": list(TRANSFORMER_MATRIX_LABELS),
        "score_families": list(score_families),
        "matrix_count": len(matrix_results),
        "per_family_layer_count": family_counts,
        "per_family_layer_sum": family_sums,
        "per_layer_matrix_count": layer_counts,
        "per_layer_matrix_sum": layer_sums,
        "score_per_layer_matrix_count": {
            str(layer): len(score_families) for layer in sorted(reference_layers)
        },
        "score_per_layer_sum": score_layer_sums,
        "scores_by_scope": scores_by_scope,
        "s_qkvo": scores_by_scope["qkvo_only"],
        "s_ffn": scores_by_scope["ffn_only"],
        "s_transformer_2d": scores_by_scope["transformer_2d"],
        "s": scores_by_scope[score_scope],
    }


def aggregate_effective_rank(
    matrix_results: Dict[str, Dict[str, Any]],
    svd_rank: int,
    score_scope: str,
) -> Dict[str, Any]:
    """Compute S from one of three scopes while retaining all seven matrix families."""
    return aggregate_transformer_effective_rank(
        matrix_results,
        svd_rank,
        score_scope=score_scope,
    )


def aggregate_subspace_similarity(
    matrix_results: Dict[str, Dict[str, Any]],
    svd_rank: int,
    score_scope: str,
    score_side: str = "u",
) -> Dict[str, Any]:
    """Average per-matrix backbone/update subspace overlap for selection."""
    if score_scope not in SCORE_SCOPES:
        raise ValueError(f"Unsupported subspace score scope: {score_scope}")
    if score_side not in {"u", "v", "mean"}:
        raise ValueError(f"Unsupported subspace score side: {score_side}")

    per_family: Dict[str, Dict[int, Dict[str, float]]] = {
        label: {} for label in TRANSFORMER_MATRIX_LABELS
    }
    for parameter_name, matrix_result in matrix_results.items():
        metadata = transformer_parameter_metadata(parameter_name)
        if metadata is None:
            raise RuntimeError(
                f"Unexpected matrix in transformer_2d scope: {parameter_name}"
            )
        layer_index, family_label = metadata
        if layer_index in per_family[family_label]:
            raise RuntimeError(
                f"Duplicate {family_label} update for layer {layer_index}"
            )
        phi_u = float(matrix_result["subspace_phi_u"])
        phi_v = float(matrix_result["subspace_phi_v"])
        if not 0.0 <= phi_u <= 1.0 or not 0.0 <= phi_v <= 1.0:
            raise ValueError(
                f"Subspace scores must lie in [0, 1], got {(phi_u, phi_v)}"
            )
        per_family[family_label][layer_index] = {"u": phi_u, "v": phi_v}

    layer_sets = {
        label: set(layer_values) for label, layer_values in per_family.items()
    }
    missing = [label for label, layers in layer_sets.items() if not layers]
    if missing:
        raise RuntimeError(
            "Cannot compute subspace score: no matrices found for "
            + ", ".join(missing)
        )
    reference_layers = layer_sets[TRANSFORMER_MATRIX_LABELS[0]]
    inconsistent = {
        label: sorted(layers)
        for label, layers in layer_sets.items()
        if layers != reference_layers
    }
    if inconsistent:
        raise RuntimeError(
            f"Transformer matrix families do not cover identical layers: {inconsistent}"
        )

    family_means = {
        label: {
            side: math.fsum(values[side] for values in per_family[label].values())
            / len(per_family[label])
            for side in ("u", "v")
        }
        for label in TRANSFORMER_MATRIX_LABELS
    }
    scores_by_scope: Dict[str, Dict[str, float]] = {}
    for scope, families in SCORE_SCOPE_FAMILIES.items():
        values_u = [
            per_family[label][layer]["u"]
            for label in families
            for layer in sorted(reference_layers)
        ]
        values_v = [
            per_family[label][layer]["v"]
            for label in families
            for layer in sorted(reference_layers)
        ]
        phi_u = math.fsum(values_u) / len(values_u)
        phi_v = math.fsum(values_v) / len(values_v)
        selected = (
            phi_u
            if score_side == "u"
            else phi_v
            if score_side == "v"
            else (phi_u + phi_v) / 2.0
        )
        scores_by_scope[scope] = {
            "phi_u": phi_u,
            "phi_v": phi_v,
            "score": selected,
        }

    selected_scope = scores_by_scope[score_scope]
    return {
        "k": svd_rank,
        "score_scope": score_scope,
        "score_side": score_side,
        "aggregation": "mean_normalized_projection_overlap_across_matrices",
        "formula_u": "||U_delta^T U_0||_F^2 / k",
        "formula_v": "||V_delta^T V_0||_F^2 / k",
        "recorded_parameter_scope": "transformer_2d",
        "recorded_families": list(TRANSFORMER_MATRIX_LABELS),
        "score_families": list(SCORE_SCOPE_FAMILIES[score_scope]),
        "matrix_count": len(matrix_results),
        "per_family_layer_count": {
            label: len(per_family[label]) for label in TRANSFORMER_MATRIX_LABELS
        },
        "per_family_mean": family_means,
        "scores_by_scope": scores_by_scope,
        "phi_u": selected_scope["phi_u"],
        "phi_v": selected_scope["phi_v"],
        "s": selected_scope["score"],
    }
