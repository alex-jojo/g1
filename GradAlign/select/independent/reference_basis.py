"""Fixed initial-backbone singular-vector bases for subspace selection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import torch


FORMAT_NAME = "subspace_reference_basis_v1"


def _name_candidates(name: str) -> list[str]:
    sanitized = name.replace("_fsdp_wrapped_module.", "")
    if sanitized.startswith("module."):
        sanitized = sanitized[len("module.") :]
    candidates = [name, sanitized]
    if sanitized.startswith("model."):
        candidates.append(sanitized[len("model.") :])
    else:
        candidates.append(f"model.{sanitized}")
    return list(dict.fromkeys(candidates))


@dataclass(frozen=True)
class ReferenceBasis:
    parameters: Mapping[str, Mapping[str, Any]]
    source_path: str
    model_path: str
    rank: int

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ReferenceBasis":
        source_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT_NAME:
            raise ValueError(f"{source_path} is not a {FORMAT_NAME} cache")
        parameters = payload.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError(f"Reference basis has no parameters: {source_path}")
        rank = int(payload.get("rank", 0))
        if rank <= 0:
            raise ValueError(f"Reference basis has invalid rank {rank}")
        return cls(
            parameters=parameters,
            source_path=source_path,
            model_path=str(payload.get("model_path", "")),
            rank=rank,
        )

    def factors(self, model_parameter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        for candidate in _name_candidates(model_parameter_name):
            entry = self.parameters.get(candidate)
            if entry is None:
                continue
            u = entry.get("u")
            v = entry.get("v")
            if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise ValueError(f"Malformed U/V factors for {candidate}")
            return u, v
        raise KeyError(
            f"No initial-backbone SVD basis for {model_parameter_name!r}; "
            "the reference and analyzed models may use different architectures"
        )
