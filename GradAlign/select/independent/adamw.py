"""Side-effect-free AdamW updates for independent gradient analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Mapping

import torch


FORMAT_NAME = "named_adamw_state_v1"
UPDATE_TARGETS = ("actual_data", "marginal_data", "full")


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
class NamedAdamWSnapshot:
    """Full optimizer state keyed by original model parameter names."""

    state: Mapping[str, Mapping[str, Any]]
    param_groups: list[Mapping[str, Any]]
    source_path: str

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "NamedAdamWSnapshot":
        source_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT_NAME:
            raise ValueError(
                f"{source_path} is not {FORMAT_NAME}; per-rank FSDP optimizer "
                "shards cannot be consumed directly"
            )
        optimizer_class = payload.get("optimizer_class")
        if not isinstance(optimizer_class, str) or not optimizer_class.endswith(
            ".AdamW"
        ):
            raise ValueError(
                f"Expected an AdamW optimizer snapshot, got {optimizer_class!r}"
            )
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError(f"Missing optimizer state in {source_path}")
        state = optimizer_state.get("state")
        param_groups = optimizer_state.get("param_groups")
        if not isinstance(state, Mapping) or not isinstance(param_groups, list):
            raise ValueError(f"Malformed optimizer state in {source_path}")
        if any(not isinstance(name, str) for name in state):
            raise ValueError("AdamW optimizer state must be keyed by parameter name")
        return cls(
            state=state,
            param_groups=param_groups,
            source_path=source_path,
        )

    def resolve_name(self, model_parameter_name: str) -> str:
        candidates = _name_candidates(model_parameter_name)
        available_names = set(self.state)
        for group in self.param_groups:
            available_names.update(group.get("params", []))
        for candidate in candidates:
            if candidate in available_names:
                return candidate
        raise KeyError(
            f"No AdamW state/group entry for {model_parameter_name!r}; model and "
            "optimizer snapshots may come from different checkpoints"
        )

    def state_and_group(
        self, model_parameter_name: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        resolved = self.resolve_name(model_parameter_name)
        for group in self.param_groups:
            if resolved in group.get("params", []):
                return self.state.get(resolved, {}), group
        raise KeyError(f"No optimizer param_group contains {resolved!r}")


@dataclass(frozen=True)
class AdamWDelta:
    actual_data: torch.Tensor
    marginal_data: torch.Tensor
    full: torch.Tensor
    next_step: int


def _step_as_int(step: int | float | torch.Tensor) -> int:
    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError("AdamW step must be a scalar")
        value = int(step.detach().cpu().item())
    else:
        value = int(step)
    if value < 0:
        raise ValueError(f"AdamW step must be non-negative, got {value}")
    return value


def _data_delta(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    next_step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    next_exp_avg = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
    next_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(
        grad, grad, value=1.0 - beta2
    )
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step
    denominator = next_exp_avg_sq.sqrt().div(bias_correction2**0.5).add(eps)
    return next_exp_avg.div(denominator).mul(-lr / bias_correction1)


@torch.no_grad()
def simulate_adamw_delta(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
) -> AdamWDelta:
    """Simulate the next PyTorch AdamW step without mutating shared state."""

    if group.get("amsgrad", False):
        raise NotImplementedError("AMSGrad optimizer snapshots are not supported")
    lr = float(group["lr"])
    beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
    eps = float(group.get("eps", 1e-8))
    weight_decay = float(group.get("weight_decay", 0.0))
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"Invalid AdamW betas: {(beta1, beta2)}")
    if lr < 0.0 or eps < 0.0 or weight_decay < 0.0:
        raise ValueError("lr, eps, and weight_decay must be non-negative")

    work_grad = grad.detach().float()
    if group.get("maximize", False):
        work_grad = -work_grad
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    step = state.get("step", 0)
    if exp_avg is None:
        exp_avg = torch.zeros_like(work_grad)
    else:
        exp_avg = exp_avg.detach().to(device=work_grad.device, dtype=torch.float32)
    if exp_avg_sq is None:
        exp_avg_sq = torch.zeros_like(work_grad)
    else:
        exp_avg_sq = exp_avg_sq.detach().to(
            device=work_grad.device, dtype=torch.float32
        )
    if exp_avg.shape != work_grad.shape or exp_avg_sq.shape != work_grad.shape:
        raise ValueError(
            f"AdamW state shape does not match gradient shape {tuple(work_grad.shape)}"
        )

    next_step = _step_as_int(step) + 1
    actual_data = _data_delta(
        work_grad,
        exp_avg,
        exp_avg_sq,
        next_step=next_step,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    zero_grad_data = _data_delta(
        torch.zeros_like(work_grad),
        exp_avg,
        exp_avg_sq,
        next_step=next_step,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    decay = param.detach().to(device=work_grad.device, dtype=torch.float32).mul(
        -lr * weight_decay
    )
    return AdamWDelta(
        actual_data=actual_data,
        marginal_data=actual_data - zero_grad_data,
        full=actual_data + decay,
        next_step=next_step,
    )


@torch.no_grad()
def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    result = float(squared_norm.sqrt().detach().cpu().item())
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"Gradient norm is not finite: {result}")
    return result


def grad_clip_coefficient(total_norm: float, max_norm: float | None) -> float:
    if max_norm is None:
        return 1.0
    if max_norm < 0.0:
        raise ValueError("AdamW gradient clip must be non-negative")
    return min(1.0, max_norm / (total_norm + 1e-6))
