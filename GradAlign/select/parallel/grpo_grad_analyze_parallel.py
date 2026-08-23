"""
Parallel GRPO (Group Relative Policy Optimization) Gradient Analyzer
Based on the verl implementation and the paper: https://arxiv.org/pdf/2402.03300

Modified to analyze gradients with model sharding across multiple GPUs.
Uses Accelerate + FSDP for efficient parallel processing.

MEMORY OPTIMIZATION TIPS:
- Use gradient_checkpointing=True to trade compute for memory
- Use cpu_offload=True if you have limited GPU memory
- Reduce mini_batch_size if hitting OOM errors
- Consider distribute_val_data=True for large validation sets
- Use mixed_precision="bf16" or "fp16" to reduce memory usage
- Reduce max_length if possible for your use case

DISTRIBUTED USAGE:
The analyzer now properly distributes work across GPUs:
- Training groups are split across processes (each GPU handles ~1/N groups)
- Validation gradients can be computed on main process only (default) or distributed
- Results are automatically gathered from all processes
"""

from pprint import pprint
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from copy import deepcopy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from torch.optim import Adam
from accelerate import Accelerator, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from typing import Any, Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
import os
import json
import math
import re
from collections import defaultdict
from time import time
import torch.distributed as dist

from grpo_loss import (
    compute_grpo_advantages, 
    compute_policy_loss, 
    compute_kl_loss, 
    compute_log_probs
)
from grpo_utils import (
    collate_batch,
    create_response_mask,
    print_training_stats,
    save_model_checkpoint,
    load_model_checkpoint,
    check_model_for_nan
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_to_full_state(
    raw_state: Dict[str, Any],
    optimizer,
) -> Dict[str, Any]:
    """Load optimizer state and filter out parameters that don't exist in the model.

    This handles cases where the optimizer state was saved from a model with a different
    architecture (e.g., with separate lm_head vs. weight tying).
    """
    # Get all parameter names that exist in the model
    # model_param_names = {name for name, _ in model.named_parameters()}
    step = raw_state['step']

    prefix = "optimizer.state."
    state: Dict[str, Dict[str, Any]] = {}
    # skipped_params = set()
    bias_sqrt = (1 - 0.999 ** step) ** 0.5
    eps = 1e-8

    mx = 1e-8
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        parts = remainder.split(".", 1)
        param_type = parts[0]
        if param_type == 'fp32_param':
            continue
        if param_type == 'exp_avg_sq':
            value = value.sqrt() / bias_sqrt + eps
            mx = max(mx, value.max())
    mult = max(1, 50 / mx)
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        parts = remainder.split(".", 1)
        param_type = parts[0]
        if param_type == 'fp32_param':
            continue
        if param_type == 'exp_avg_sq':
            # value = value.sqrt() + 1e-8
            value = value.sqrt() / bias_sqrt + eps
            value *= mult
            print(f'rank {dist.get_rank()} name: {remainder} value: {value.max()}, {value}')
            # assert value.max() < 100
        param_name = remainder[len(param_type) + 1:]
        if param_name != 'lm_head.weight':
            param_name = 'model.' + param_name

        # # Skip parameters that don't exist in the current model
        # if param_name not in model_param_names:
        #     skipped_params.add(param_name)
        #     continue

        if not param_name in state:
            state[param_name] = {'step': torch.tensor(1.0, dtype=value.dtype, device=value.device)}
        state[param_name][param_type] = value.clone()

    # # Log skipped parameters on rank 0
    # if skipped_params and dist.get_rank() == 0:
    #     print(f"Warning: Skipped optimizer state for {len(skipped_params)} missing parameters:")
    #     for param_name in sorted(skipped_params):
    #         print(f"  - {param_name}")

    param_groups = deepcopy(optimizer.param_groups)
    params = [x for x, y in state.items()]
    param_groups[0]['params'] = params

    return {
        "state": state,
        "param_groups": param_groups,
    }


class GRPOGradientAnalyzerParallel:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        learning_rate: float = 1e-6,
        kl_loss_coef: float = 0.001,
        clip_ratio: float = 0.2,
        norm_adv_by_std: bool = True,
        max_length: int = 8192,
        n_samples_per_prompt: int = 5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        epsilon: float = 1e-6,
        fsdp_auto_wrap_policy: str = "transformer_auto_wrap_policy",
        mixed_precision: str = "bf16",
        use_gradient_checkpointing: bool = True,
        cpu_offload: bool = False,
        distribute_val_data: bool = False,
        use_optimizer: bool = False,
        optimizer_state_path: Optional[str] = None,
        optimizer_epsilon: float = 1e-8,
        mode: str = 'sim',
        svd_rank: int = 128,
    ):
        """
        Initialize parallel GRPO gradient analyzer.
        
        Args:
            model_path: Path to the model to analyze
            tokenizer_path: Path to tokenizer (defaults to model_path)
            learning_rate: Learning rate for optimizer (not used for analysis)
            kl_loss_coef: Coefficient for KL divergence loss
            clip_ratio: PPO clipping ratio
            norm_adv_by_std: Whether to normalize advantages by std
            max_length: Maximum sequence length
            n_samples_per_prompt: Number of samples per prompt (group size)
            ppo_epochs: Number of PPO epochs per update (not used for analysis)
            mini_batch_size: Mini batch size for analysis
            epsilon: Small constant for numerical stability
            fsdp_auto_wrap_policy: FSDP auto wrap policy
            mixed_precision: Mixed precision mode
            use_gradient_checkpointing: Whether to enable gradient checkpointing for memory efficiency
            cpu_offload: Whether to offload parameters to CPU when not in use
            distribute_val_data: Whether to distribute validation data across processes (vs computing on main process only)
        """
        self.learning_rate = learning_rate
        self.kl_loss_coef = kl_loss_coef
        self.clip_ratio = clip_ratio
        self.norm_adv_by_std = norm_adv_by_std
        self.max_length = max_length
        self.n_samples_per_prompt = n_samples_per_prompt
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.epsilon = epsilon
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cpu_offload = cpu_offload
        self.distribute_val_data = distribute_val_data
        self.use_optimizer = use_optimizer
        self.optimizer_state_path = optimizer_state_path
        self.optimizer_epsilon = optimizer_epsilon
        if self.use_optimizer and self.optimizer_state_path is None:
            raise ValueError("optimizer_state_path must be provided when use_optimizer is True")
        torch.set_printoptions(precision=10, threshold=5)
        self.mode = mode
        self.svd_rank = svd_rank
        if self.svd_rank <= 0:
            raise ValueError("svd_rank must be positive")

        self.optimizer: Optional[Adam] = None
        self._name_to_param: Dict[str, torch.nn.Parameter] = {}

        # Configure FSDP plugin
        fsdp_plugin = FullyShardedDataParallelPlugin(
            # auto_wrap_policy=self._get_auto_wrap_policy(fsdp_auto_wrap_policy),
            # backward_prefetch="backward_pre",
            # forward_prefetch=True,
            # use_orig_params=True,
            # sync_module_states=True,
            # limit_all_gathers=True,
            cpu_offload=cpu_offload,
        )
        
        # Configure Accelerator with FSDP
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            mixed_precision=(mixed_precision if mixed_precision != "None" else None),
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=[ddp_kwargs],
        )
        
        # Initialize on main process first
        if self.accelerator.is_main_process:
            logger.info(f"Initializing parallel GRPO gradient analyzer")
            logger.info(f"Using {self.accelerator.num_processes} processes")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Gradient checkpointing: {use_gradient_checkpointing}")
            logger.info(f"CPU offload: {cpu_offload}")
            logger.info(f"Distribute validation data: {distribute_val_data}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and prepare models
        self._load_models()

        # FSDP turns an original 2-D parameter into rank-local flat shards.
        # Keep the original names and shapes so SVD mode can reconstruct one
        # complete matrix at a time after backward.
        self._parameter_shapes = {
            self._sanitize_name(name): tuple(param.shape)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Setup optimizer (needed for gradient computation structure)
        # self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Prepare with accelerator
        self.model = self.accelerator.prepare(self.model)
        # self.ref_model = self.accelerator.prepare(self.ref_model)

        if self.mode == 'svd':
            svd_matrix_count = len(self._svd_parameter_specs())
            if svd_matrix_count == 0:
                raise ValueError(
                    "SVD mode found no 2-D weight matrices inside transformer layers; "
                    "check the model parameter naming convention"
                )
            if self.accelerator.is_main_process:
                logger.info(
                    f"SVD mode will analyze {svd_matrix_count} transformer-layer "
                    f"matrices per question (top {self.svd_rank})"
                )

        if self.use_optimizer:
            self._setup_optimizer()
        
        # Enable gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.accelerator.is_main_process:
                logger.info("Gradient checkpointing enabled")
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {total_params:,}")
            logger.info("Models successfully sharded across GPUs")
    
    def _get_auto_wrap_policy(self, policy_name: str):
        """Get the appropriate auto wrap policy."""
        if policy_name == "transformer_auto_wrap_policy":
            # Determine the transformer layer class based on model type
            # This is a heuristic - you might need to adjust for other model types
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                return transformer_auto_wrap_policy(LlamaDecoderLayer)
            except:
                try:
                    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                    return transformer_auto_wrap_policy(MistralDecoderLayer)
                except:
                    # Fallback - wrap by parameter count
                    return lambda module, recurse, nonwrapped_numel: nonwrapped_numel >= 1e6
        else:
            # Size-based wrapping
            return lambda module, recurse, nonwrapped_numel: nonwrapped_numel >= 1e6
    
    def _load_models(self):
        """Load the main model and reference model."""
        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.model.train()


    def _setup_optimizer(self) -> None:
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # for name, param in self.model.named_parameters():
        #     print(f'rank {dist.get_rank()} name: {name}, shape: {param.shape}')
        # return
        if self.accelerator.is_main_process:
            # full_state = torch.load('~/data_selection/data/huggingface/tmp/full_state.pth', map_location="cpu")
            full_state = torch.load(self.optimizer_state_path, map_location="cpu")
            full_state = load_to_full_state(full_state, self.optimizer)
        else:
            full_state = {}
        # return

        sharded_state = FSDP.scatter_full_optim_state_dict(
            full_state,
            self.model,
            optim=self.optimizer,
        )
        self.optimizer.load_state_dict(sharded_state)
        print(f'rank {dist.get_rank()} sharded_state keys: {sharded_state["state"].keys()}')

        # Build reliable mappings between model parameters and optimizer state
        param_to_name = {param: name for name, param in self.model.named_parameters()}
        self._name_to_param = {name: param for name, param in self.model.named_parameters()}
        self.exp_avg_sg_map = {}

        # Validate shapes between each optimizer state's exp_avg_sq and its parameter
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                name = param_to_name[p]
                # if name is None:
                #     continue
                state = self.optimizer.state[p]
                # if not state:
                #     continue
                # print('in', name, 'state', state)
                v = state.get("exp_avg_sq")
                if v is None:
                    continue
                self.exp_avg_sg_map[name] = v.clone().detach()
                print(f'rank {dist.get_rank()} name: {name}, exp_avg_sg_map: {self.exp_avg_sg_map[name]}')
                p_view = getattr(p, "_local_shard", None)
                p_eff = p_view if isinstance(p_view, torch.Tensor) else p
                if p_eff.numel() == 0 or v.numel() == 0:
                    continue
                # print(f"{name} param {tuple(p_eff.shape)} v {tuple(v.shape)}")
                # assert p_eff.shape == v.shape, f"Shape mismatch for {name}: param {p_eff.shape} vs v {v.shape}"
        # for name, param in self.optimizer.state_dict()["state"].items():
        #     name = idx_to_name[int(name)]
        #     param = param["exp_avg_sq"]
        #     print(name, param.shape, self.model[name].shape)
        #     assert param.shape == self.model[name].shape
        # exit()
        # self.optimizer.zero_grad(set_to_none=True)
        del full_state
        del sharded_state
        del param_to_name
        del self._name_to_param
        del self.optimizer
        torch.cuda.empty_cache()

    def _sanitize_name(self, name: str) -> str:
        name = name.replace("_fsdp_wrapped_module.", "")
        if name.startswith("module."):
            name = name[len("module."):]
        return name

    def _generate_name_candidates(self, name: str) -> List[str]:
        candidates = []
        sanitized = self._sanitize_name(name)
        candidates.append(name)
        candidates.append(sanitized)
        if sanitized.startswith("model."):
            candidates.append(sanitized[len("model."):])
        else:
            candidates.append(f"model.{sanitized}")
        return list(dict.fromkeys(filter(None, candidates)))


    # def _apply_optimizer_scaling(
    #     self,
    #     param_name: str,
    #     grad_tensor: torch.Tensor,
    # ) -> torch.Tensor:

    #     param = self._name_to_param.get(param_name)
    #     if param is None or self.optimizer is None:
    #         return grad_tensor
    #     state = self.optimizer.state.get(param)
    #     if not state:
    #         return grad_tensor
    #     v_tensor = state.get("exp_avg_sq")
    #     if v_tensor is None:
    #         return grad_tensor

    #     v_device = v_tensor.to(self.accelerator.device, dtype=grad_tensor.dtype, non_blocking=True)
    #     scaled_grad = grad_tensor / torch.sqrt(v_device + self.optimizer_epsilon)
    #     return scaled_grad
    
    def prepare_data(self, prompts: List[str], responses: List[str], 
                    rewards: List[float], group_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Deprecated: preparation moved to prepare.py. Kept for backward-compat only."""
        raise NotImplementedError("Use precomputed shards from prepare.py instead of prepare_data")
    
    def compute_loss_and_gradients_distributed(
        self,
        batch_data: List[Dict[str, torch.Tensor]],
        scale_with_optimizer: bool = False,
        perform_norm: bool = True,
    ) -> Tuple[float, Dict[str, torch.Tensor], Optional[float]]:
        """
        Compute loss and gradients for a batch of data using all GPUs in data-parallel fashion.
        All GPUs participate in computing gradients, then gradients are aggregated across processes.
        """
        total_start_time = time()
        # Clear gradients using model's zero_grad method
        self.model.zero_grad()
        print('scale', scale_with_optimizer)
        
        # Clear any existing gradients
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         param.grad.zero_()
        #         param.requires_grad = True
        
        if self.accelerator.is_main_process:
            print(f"Computing distributed gradients for {len(batch_data)} samples across {self.accelerator.num_processes} GPUs")
        
        data_per_process = batch_data
        
        if self.accelerator.is_main_process:
            print(f"Process {self.accelerator.process_index} handling {len(data_per_process)} samples")

        # ``compute_policy_loss`` returns a token mean for each mini-batch.  To
        # make gradient accumulation invariant to how a question is split into
        # mini-batches, weight every local mini-batch by its fraction of all
        # valid response tokens across all data-parallel ranks.
        #
        # Each stored response mask has already been shifted once by
        # ``create_response_mask``; the loss path applies a second shift below,
        # so mirror that shift when counting tokens here.
        local_response_tokens = sum(
            int(item['response_mask'][..., 1:].sum().item())
            for item in data_per_process
        )
        local_response_tokens_tensor = torch.tensor(
            float(local_response_tokens),
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        global_response_tokens = self.accelerator.reduce(
            local_response_tokens_tensor,
            reduction="sum",
        )
        if global_response_tokens.item() <= 0:
            raise ValueError("Cannot compute gradients: batch has no valid response tokens")

        if self.accelerator.is_main_process:
            print(f"Global valid response tokens: {int(global_response_tokens.item())}")

        total_loss_numerator = torch.tensor(
            0.0,
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        # Process local data in mini-batches
        # print("len", len(data_per_process))
        for i in range(0, len(data_per_process), self.mini_batch_size):
            start_time = time()
            mini_batch = data_per_process[i:i + self.mini_batch_size]
            
            # Pad batch
            padded_batch = collate_batch(mini_batch, self.tokenizer.pad_token_id)
            
            # Move to accelerator device
            input_ids = padded_batch['input_ids'].to(self.accelerator.device)
            attention_mask = padded_batch['attention_mask'].to(self.accelerator.device)
            response_mask = padded_batch['response_mask'].to(self.accelerator.device)
            advantages = padded_batch['advantages'].to(self.accelerator.device)
            
            # Compute new log probabilities
            new_log_probs = compute_log_probs(
                self.model, input_ids, attention_mask, response_mask
            )[..., 1:]
            
            # Adjust tensors for shifted sequences
            response_mask = response_mask[..., 1:]
            old_log_probs = new_log_probs.detach()
            
            # Broadcast advantages to token level
            advantages_expanded = advantages.unsqueeze(-1).expand_as(response_mask)
            
            # Compute losses
            policy_loss = compute_policy_loss(
                old_log_probs, new_log_probs, advantages_expanded, response_mask, self.clip_ratio, 
                self.accelerator
            )

            mini_batch_tokens = response_mask.sum(dtype=torch.float32)
            if mini_batch_tokens.item() <= 0:
                raise ValueError("Cannot compute gradients: mini-batch has no valid response tokens")

            # FSDP averages gradients across data-parallel ranks. Multiplying
            # by num_processes cancels that average, while the token fraction
            # makes the accumulated gradient exactly the global token mean:
            #   sum_b (tokens_b / total_tokens) * grad(mean_loss_b).
            token_fraction = mini_batch_tokens / global_response_tokens
            batch_loss = (
                policy_loss
                * token_fraction.to(dtype=policy_loss.dtype)
                * self.accelerator.num_processes
            )

            # Track the unscaled token-loss numerator for meaningful logging.
            total_loss_numerator += policy_loss.detach().to(torch.float32) * mini_batch_tokens.detach()

            # Backward pass to accumulate gradients
            self.accelerator.backward(batch_loss)
            # print('batch_loss', batch_loss)
            if self.accelerator.is_main_process:
                print(f"Backward pass time: {time() - start_time:.2f}s")
        # if scale_with_optimizer:
        #     self.optimizer.step()
        #     full_state = FSDP.full_optim_state_dict(self.model, self.optimizer)
        #     if self.accelerator.is_main_process:
        #         torch.save(full_state, '~/data_selection/data/huggingface/tmp/full_state.pth')
        #         print('State' + '=' * 20)
        #         pprint(self.optimizer.state_dict())
        #         print('Full state' + '=' * 20)
        #         pprint(full_state)
            # pprint(self.optimizer.state_dict())
            # print('optimizer after step', self.optimizer.state_dict())
        
        # Report the same global token-mean loss whose gradient was computed.
        global_loss_numerator = self.accelerator.reduce(
            total_loss_numerator,
            reduction="sum",
        )
        global_avg_loss = (global_loss_numerator / global_response_tokens).item()
        
        # Extract and aggregate gradients across all processes
        gradients: Dict[str, torch.Tensor] = {}
        norm_sq = torch.tensor(0.0, device=self.accelerator.device, dtype=torch.float32)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_tensor = param.grad.clone().detach().to(torch.float32)
                if scale_with_optimizer:
                    # print('fa')
                    grad_tensor = grad_tensor / self.exp_avg_sg_map[name]
                    # grad_tensor = self._apply_optimizer_scaling(name, grad_tensor)
                gradients[name] = grad_tensor
                if perform_norm:
                    norm_sq += torch.sum(grad_tensor ** 2)
        if perform_norm:
            norm_sq = self.accelerator.reduce(norm_sq, reduction="sum")
            norm_sq = norm_sq ** 0.5 + 1e-5
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = gradients[name] / norm_sq
        
        if self.accelerator.is_main_process:
            print("Total loss: ", global_avg_loss)
            print(f"Distributed gradient computation time: {time() - total_start_time:.2f}s")
            print(f"Global average loss: {global_avg_loss:.4f}")
        
        return global_avg_loss, gradients, (norm_sq.item() if perform_norm else None)
    
    def compute_cosine_similarity_distributed(self, grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Compute global dot product and per-module dot product distribution using distributed computation.
        
        Returns:
            total_dot (float): Global dot product sum across all params and GPUs
            distribution (Dict[str, float]): Mapping of module name -> global dot product sum
        """
        start_time = time()
        
        device = self.accelerator.device
        # Global accumulator
        dot_product = torch.tensor(0.0, device=device, dtype=torch.float32)
        # Local per-module accumulators on this process
        local_module_dot: Dict[str, torch.Tensor] = {}
        
        # Compute local dot products for this GPU's shard
        for name in grad1.keys():
            assert name in grad2
            g1 = grad1[name].to(device).flatten()
            g2 = grad2[name].to(device).flatten()
            dp = torch.sum(g1 * g2)
            dot_product += dp
            # Derive module name by stripping the last component (e.g., ".weight", ".bias")
            module_name = name.rsplit('.', 1)[0] if '.' in name else name
            # print('name', name)
            if module_name in local_module_dot:
                local_module_dot[module_name] = local_module_dot[module_name] + dp
            else:
                local_module_dot[module_name] = dp
        
        # All-reduce total dot product across processes
        dot_product = self.accelerator.reduce(dot_product, reduction="sum")
        
        # Build a consistent set of module names across all processes using model structure
        module_names_set = set()
        for pname, _ in self.model.named_parameters():
            module_names_set.add(pname.rsplit('.', 1)[0] if '.' in pname else pname)
        module_names = sorted(module_names_set)
        
        # Reduce per-module dot products across all processes
        distribution: Dict[str, float] = {}
        for module_name in module_names:
            local_val = local_module_dot.get(module_name, torch.tensor(0.0, device=device, dtype=torch.float32))
            global_val = self.accelerator.reduce(local_val, reduction="sum")
            # Only main process materializes the value; others will return empty dict later if needed
            if self.accelerator.is_main_process:
                distribution[module_name] = float(global_val.item())
        
        if self.accelerator.is_main_process:
            print(f"Cosine similarity computation time: {time() - start_time:.2f}s")
        
        return dot_product.item(), distribution

    @staticmethod
    def _is_transformer_layer_matrix(name: str, shape: Tuple[int, ...]) -> bool:
        """Return whether a parameter is a 2-D weight inside a transformer block."""
        if len(shape) != 2 or not name.endswith(".weight"):
            return False
        return re.search(r"(?:^|\.)(?:layers|h|blocks|block)\.\d+(?:\.|$)", name) is not None

    def _svd_parameter_specs(self) -> List[Tuple[str, Tuple[int, int]]]:
        """Return a deterministic list of transformer-layer matrices to analyze."""
        return [
            (name, shape)
            for name, shape in sorted(self._parameter_shapes.items())
            if self._is_transformer_layer_matrix(name, shape)
        ]

    def _gather_full_gradient_matrix(
        self,
        local_gradient: Optional[torch.Tensor],
        parameter_name: str,
        original_shape: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """Reconstruct one original FSDP gradient matrix on the main rank."""
        device = self.accelerator.device
        if local_gradient is None:
            local_flat = torch.empty(0, device=device, dtype=torch.float32)
        else:
            local_flat = local_gradient.detach().reshape(-1).to(
                device=device,
                dtype=torch.float32,
            )

        if self.accelerator.num_processes == 1:
            if not self.accelerator.is_main_process:
                return None
            expected_numel = math.prod(original_shape)
            if local_flat.numel() != expected_numel:
                raise RuntimeError(
                    f"Cannot reconstruct gradient for {parameter_name}: "
                    f"found {local_flat.numel()} values, expected {expected_numel} "
                    f"for shape {original_shape}"
                )
            return local_flat.reshape(original_shape)

        local_size = torch.tensor([local_flat.numel()], device=device, dtype=torch.long)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self.accelerator.num_processes)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(size.item()) for size in gathered_sizes]
        max_size = max(sizes)

        padded = torch.zeros(max_size, device=device, dtype=torch.float32)
        if local_flat.numel() > 0:
            padded[:local_flat.numel()].copy_(local_flat)
        gathered_shards = [torch.empty_like(padded) for _ in range(self.accelerator.num_processes)]
        dist.all_gather(gathered_shards, padded)

        if not self.accelerator.is_main_process:
            return None

        full_flat = torch.cat([
            shard[:shard_size]
            for shard, shard_size in zip(gathered_shards, sizes)
            if shard_size > 0
        ])
        expected_numel = math.prod(original_shape)
        if full_flat.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot reconstruct gradient for {parameter_name}: "
                f"gathered {full_flat.numel()} values, expected {expected_numel} "
                f"for shape {original_shape}; local shard sizes={sizes}"
            )
        return full_flat.reshape(original_shape)

    def compute_svd_distributed(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute and retain only the leading singular values of each layer matrix."""
        gradients_by_name = {
            self._sanitize_name(name): gradient
            for name, gradient in gradients.items()
        }
        parameter_specs = self._svd_parameter_specs()
        results: Dict[str, Dict[str, Any]] = {}

        if self.accelerator.is_main_process:
            print(
                f"Running truncated SVD for {len(parameter_specs)} transformer-layer "
                f"gradient matrices (top {self.svd_rank})"
            )

        for parameter_name, original_shape in parameter_specs:
            matrix = self._gather_full_gradient_matrix(
                gradients_by_name.get(parameter_name),
                parameter_name,
                original_shape,
            )
            if not self.accelerator.is_main_process:
                continue

            target_rank = min(self.svd_rank, min(original_shape))
            # Oversampling improves the randomized top-k approximation while
            # still avoiding a full SVD. Only target_rank values are retained.
            lowrank_q = min(target_rank + 8, min(original_shape))
            # svd_lowrank is randomized. Fork and restore RNG state so this
            # analysis never changes dropout/randomness in the next question.
            rng_devices = [matrix.device.index] if matrix.is_cuda else []
            with torch.random.fork_rng(devices=rng_devices):
                if matrix.is_cuda:
                    torch.cuda.manual_seed(0)
                else:
                    torch.manual_seed(0)
                _, singular_values, _ = torch.svd_lowrank(
                    matrix,
                    q=lowrank_q,
                    niter=2,
                )
            singular_values = torch.sort(singular_values, descending=True).values[:target_rank]
            singular_values_cpu = singular_values.cpu()
            results[parameter_name] = {
                "shape": list(original_shape),
                "num_singular_values": target_rank,
                "singular_values": singular_values_cpu.tolist(),
                # Singular values are non-negative, so this is also the
                # absolute-value sum. It is a top-k/Ky Fan norm, not the full
                # nuclear norm when target_rank < min(original_shape).
                "topk_singular_value_sum": float(singular_values_cpu.sum().item()),
            }
            del matrix, singular_values, singular_values_cpu

        return results

    def _zero_svd_results(self) -> Dict[str, Dict[str, Any]]:
        """Materialize the exact all-zero spectrum for a zero-advantage group."""
        if not self.accelerator.is_main_process:
            return {}
        return {
            parameter_name: {
                "shape": list(original_shape),
                "num_singular_values": min(self.svd_rank, min(original_shape)),
                "singular_values": [0.0] * min(self.svd_rank, min(original_shape)),
                "topk_singular_value_sum": 0.0,
            }
            for parameter_name, original_shape in self._svd_parameter_specs()
        }

    @staticmethod
    def _effective_rank_from_singular_values(singular_values: List[float]) -> float:
        """Compute exp(entropy) from one matrix's saved top-k singular values."""
        values = [float(value) for value in singular_values]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Singular values must be finite and non-negative")

        singular_value_sum = math.fsum(values)
        # A zero GRPO gradient has no normalized singular-value distribution.
        # Use the explicit convention requested for this analysis: R = 0.
        if singular_value_sum == 0.0:
            return 0.0

        probabilities = [
            value / singular_value_sum
            for value in values
            if value > 0.0
        ]
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in probabilities
        )
        return math.exp(entropy)

    def compute_effective_rank_score(
        self,
        svd_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate top-k effective ranks over layers and Q/K/V/O projections."""
        projection_labels = {
            "q_proj": "Q",
            "k_proj": "K",
            "v_proj": "V",
            "o_proj": "O",
        }
        per_projection_layers: Dict[str, Dict[int, float]] = {
            label: {} for label in projection_labels.values()
        }
        attention_pattern = re.compile(
            r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\."
            r".*?self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"
        )

        for parameter_name, matrix_result in svd_results.items():
            match = attention_pattern.search(parameter_name)
            if match is None:
                continue

            layer_index = int(match.group(1))
            projection_label = projection_labels[match.group(2)]
            if layer_index in per_projection_layers[projection_label]:
                raise RuntimeError(
                    f"Duplicate {projection_label} projection gradient for layer {layer_index}"
                )

            effective_rank = self._effective_rank_from_singular_values(
                matrix_result["singular_values"]
            )
            matrix_result["topk_effective_rank"] = effective_rank
            per_projection_layers[projection_label][layer_index] = effective_rank

        missing_projections = [
            label
            for label, layer_values in per_projection_layers.items()
            if not layer_values
        ]
        if missing_projections:
            raise RuntimeError(
                "Cannot compute effective-rank score: no layer matrices found for "
                + ", ".join(missing_projections)
            )

        layer_counts = {
            label: len(layer_values)
            for label, layer_values in per_projection_layers.items()
        }
        if len(set(layer_counts.values())) != 1:
            raise RuntimeError(
                f"Cannot compute effective-rank score with unequal Q/K/V/O layer counts: {layer_counts}"
            )

        projection_sums = {
            label: math.fsum(layer_values.values())
            for label, layer_values in per_projection_layers.items()
        }
        final_score = math.fsum(projection_sums.values())
        return {
            "k": self.svd_rank,
            "zero_spectrum_effective_rank": 0.0,
            "aggregation": "sum_layers_then_equal_sum_qkvo",
            "per_projection_layer_count": layer_counts,
            "per_projection_layer_sum": projection_sums,
            "s": final_score,
        }

    def save_svd_result_incrementally(
        self,
        group_id: int,
        loss: float,
        svd_results: Dict[str, Dict[str, Any]],
        output_path: str,
        zero_advantage: bool = False,
    ) -> None:
        """Append one question's per-matrix truncated SVD result to JSONL."""
        if not self.accelerator.is_main_process:
            return
        effective_rank_score = self.compute_effective_rank_score(svd_results)
        result = {
            "group_id": group_id,
            "loss": loss,
            "zero_advantage": zero_advantage,
            "svd_rank": self.svd_rank,
            "svd_method": "torch.svd_lowrank",
            "svd_niter": 2,
            "svd_seed": 0,
            "matrices": svd_results,
            "effective_rank_topk": effective_rank_score,
            "timestamp": time(),
        }
        with open(output_path, "a") as output_file:
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()
        print(f"Saved top-{self.svd_rank} SVD result for group {group_id} to {output_path}")
    
    def save_result_incrementally(self, group_id: int, similarity: float, loss: float, output_path: str, distribution: Optional[Dict[str, float]] = None):
        """Save a single group's result to the JSONL files.
        
        - Main file (output_path): summary without full distribution
        - Distributions file (derived from output_path): full per-module distribution
        """
        if not self.accelerator.is_main_process:
            return
        
        # Build top-5 sorted list by absolute dot value if distribution provided
        # Normalize names to be concise by stripping FSDP wrapper prefixes and merging duplicates
        sorted_top = []
        if distribution:
            def _normalize_name(name: str) -> str:
                # Remove any occurrences of FSDP wrapper prefixes
                return name.replace("_fsdp_wrapped_module.", "")
            norm_dist: Dict[str, float] = {}
            for k, v in distribution.items():
                nk = _normalize_name(k)
                norm_dist[nk] = norm_dist.get(nk, 0.0) + float(v)
            sorted_top = sorted(norm_dist.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        result = {
            'group_id': group_id,
            'h_sorted': sorted_top,
            'similarity': similarity,
            'loss': loss,
            'timestamp': time()
        }
        
        # Append to JSONL file
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')
            f.flush()  # Ensure immediate write to disk
        
        # Also write full distribution to a separate JSONL file if provided
        if distribution:
            base_dir = os.path.dirname(output_path)
            base_name = os.path.basename(output_path)
            root, ext = os.path.splitext(base_name)
            dist_name = f"{root}_distributions{ext}"
            dist_path = os.path.join(base_dir, dist_name)
            dist_record = {
                'group_id': group_id,
                'distribution': distribution,
                'timestamp': time()
            }
            with open(dist_path, 'a') as fdist:
                fdist.write(json.dumps(dist_record) + '\n')
                fdist.flush()
            print(f"Saved distribution for group {group_id} to {dist_path}")
        
        print(f"Saved result for group {group_id} to {output_path}")
    
    def analyze_gradients(self, train_data: List[Dict], val_data: List[Dict], 
                         output_path: str = None, processed_groups: set = None, norm_before_accumlation=False) -> Dict[str, float]:
        """
        Analyze gradients between validation data and train groups using distributed processing.
        All GPUs participate in computing gradients for each group, then similarities are computed on main process.
        
        Args:
            train_data: Training data grouped by group_id
            val_data: Validation data
            output_path: Path to save incremental results (JSONL format)
            processed_groups: Set of already processed group IDs to skip
        """
        
        if processed_groups is None:
            processed_groups = set()
        
        # Step 1: Compute validation gradients using all GPUs
        if self.accelerator.is_main_process:
            if self.mode == 'svd':
                print("SVD mode: validation gradient and cosine alignment are disabled")
            else:
                print("Computing validation gradients using all GPUs...")

        if self.mode not in {'norm', 'svd'}:
            if norm_before_accumlation and self.mode != 'dot':
                cnt = 0
                val_gradients = {}
                data_map = {}
                for item in val_data:
                    if item['group_id'] not in data_map:
                        data_map[item['group_id']] = []
                    data_map[item['group_id']].append(item)
                for group_id, group_data in data_map.items():
                    cnt += 1
                    print('in', group_id)
                    val_loss, grad, norm_sq = self.compute_loss_and_gradients_distributed(
                        group_data,
                        scale_with_optimizer=self.use_optimizer,
                    )
                    for name, param in grad.items():
                        # print(name, param.shape)
                        if param is not None:
                            if name not in val_gradients:
                                val_gradients[name] = param.clone().detach()
                            else:
                                val_gradients[name] = val_gradients[name] + param
                sq_sum = 0
                for name, param in val_gradients.items():
                    sq_sum += torch.sum(param ** 2)
                sq_sum = self.accelerator.reduce(sq_sum, reduction="sum")
                sq_sum = sq_sum ** 0.5 + 1e-5
                # print('sq_sum', sq_sum)
                    
                for name in val_gradients.keys():
                    val_gradients[name] = val_gradients[name] / sq_sum
            else:
                val_loss, val_gradients, norm_sq = self.compute_loss_and_gradients_distributed(
                    val_data,
                    scale_with_optimizer=self.use_optimizer,
                    perform_norm=True,#(self.mode != 'dot'),
                )
        
        # if self.accelerator.is_main_process:
        #     print(f"Validation loss: {val_loss:.4f}")
        
        # Step 2: Group training data by group_id
        train_groups = defaultdict(list)
        for item in train_data:
            train_groups[item['group_id']].append(item)
        
        if self.accelerator.is_main_process:
            total_groups = len(train_groups)
            unprocessed_groups = [gid for gid in train_groups.keys() if gid not in processed_groups]
            print(f"Found {total_groups} training groups total")
            print(f"Already processed: {len(processed_groups)} groups")
            print(f"Remaining to process: {len(unprocessed_groups)} groups")
        
        # Step 3: Compute gradients for each unprocessed training group using all GPUs
        all_similarities = {}
        all_group_items = [(gid, data) for gid, data in train_groups.items() if gid not in processed_groups]
        
        if self.accelerator.is_main_process and all_group_items:
            print(f"Computing gradients for {len(all_group_items)} unprocessed groups using all {self.accelerator.num_processes} GPUs")
            progress_bar = tqdm(all_group_items, desc="Analyzing training groups")
        else:
            progress_bar = all_group_items
        
        for group_id, group_data in progress_bar:
            if self.accelerator.is_main_process:
                print(f"\nProcessing group {group_id} ({len(group_data)} samples) on all GPUs...")
            
            # All GPUs participate in computing gradients for this group
            zero_advantage = True
            for x in group_data:
                # print(x)
                # print(x['advantages'])
                # print(str(self.accelerator.process_index) + " " + str(x['advantages']) + ' ' + str(x['group_id']) + '' + str(x['input_ids'].shape))
                if abs(x['advantages'].item()) >= 1e-6:
                    zero_advantage = False
            # if self.accelerator.is_main_process:
            # print(str(self.accelerator.process_index) + " " + str(zero_advantage))
            if zero_advantage:# and False:
                if self.accelerator.is_main_process:
                    print(f"Group {group_id} has all-zero advantages")
                group_loss = 0.0
                similarity = 0.0
                distribution = {}
                svd_results = self._zero_svd_results() if self.mode == 'svd' else {}
            else:
                group_loss, group_gradients, norm_sq = self.compute_loss_and_gradients_distributed(
                    group_data,
                    scale_with_optimizer=(self.use_optimizer and self.mode != 'svd'),
                    perform_norm=(self.mode not in {'dot', 'svd'}),
                )
                
                # Compute global dot product and per-module distribution
                if self.mode == 'svd':
                    svd_results = self.compute_svd_distributed(group_gradients)
                    similarity = 0.0
                    distribution = {}
                elif self.mode !='norm':
                    similarity, distribution = self.compute_cosine_similarity_distributed(val_gradients, group_gradients)
                else:
                    similarity = norm_sq
                    distribution = {}
            # Only main process collects, prints, and saves results
            if self.accelerator.is_main_process:
                all_similarities[group_id] = similarity
                if self.mode == 'svd':
                    print(f"Group {group_id}: loss={group_loss:.4f}, matrices={len(svd_results)}")
                else:
                    print(f"Group {group_id}: loss={group_loss:.4f}, similarity={similarity:.4f}")
                
                # Save result incrementally if output path is provided
                if output_path:
                    if self.mode == 'svd':
                        self.save_svd_result_incrementally(
                            group_id,
                            group_loss,
                            svd_results,
                            output_path,
                            zero_advantage=zero_advantage,
                        )
                    else:
                        self.save_result_incrementally(group_id, similarity, group_loss, output_path, distribution)
            
            # Synchronize all processes before moving to next group
            self.accelerator.wait_for_everyone()
        
        # Ensure all processes synchronize before returning
        self.accelerator.wait_for_everyone()
        
        # Only main process returns the actual results, others return empty dict
        if not self.accelerator.is_main_process:
            all_similarities = {}
        
        if self.accelerator.is_main_process:
            if all_similarities:
                print(f"\nCompleted gradient analysis for {len(all_similarities)} new groups")
            else:
                print("\nNo new groups were processed in this run")
        
        return all_similarities
    
    def _distribute_data_across_processes(self, data: List[Dict]) -> List[Dict]:
        """Distribute data across processes using round-robin assignment."""
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        # Assign data to this process using modulo
        process_data = [item for i, item in enumerate(data) if i % num_processes == process_index]
        return process_data

    def _gather_sort_and_broadcast_batch(self, local_batch: List[Dict]) -> List[Dict]:
        """Gather local batches to rank 0, sort deterministically, and broadcast to all ranks.

        Deterministic sort key:
        - group_id (ascending)
        - input length (ascending)
        - global_index (ascending) to break ties
        """
        # Prepare local payload: filter out placeholders where input_ids is scalar zero tensor
        real_items = []
        for item in local_batch:
            is_placeholder = (len(item['input_ids'].shape)==0)
            if not is_placeholder:
                real_items.append(item)

        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index

        merged_unique = None

        # Gather objects on rank 0
        gathered_lists = [None] * world_size if self.accelerator.is_main_process else None
        dist.gather_object(real_items, gathered_lists, dst=0)
        # print('haha' + str(gathered_lists))

        merged = []
        if self.accelerator.is_main_process:
            for lst in gathered_lists:
                if lst:
                    merged.extend(lst)

            # Deduplicate by global_index if duplicates arise
            # seen = {}
            # merged_unique = list(seen.values())

            def _seq_len(t):
                return t.shape[0]

            merged.sort(key=lambda it: (int(it['group_id']), _seq_len(it['input_ids'])))

        # Broadcast the result from rank 0 to all ranks
        obj_list = [merged]
        dist.broadcast_object_list(obj_list, src=0)
        merged = obj_list[0]
        # print('`haha`' + str(merged))
        return merged

        # Re-introduce placeholders so every rank has a full list for sync
        # world_size = self.accelerator.num_processes
        # rank = self.accelerator.process_index

        # def _make_placeholder_like(item):
        #     return {
        #         'input_ids': torch.tensor([0], dtype=torch.long),
        #         'attention_mask': torch.tensor([0], dtype=torch.long),
        #         'response_mask': torch.tensor([0], dtype=torch.long),
        #         'advantages': item['advantages'],
        #         'group_id': item['group_id']
        #     }

        # expanded = []
        # # for idx, it in enumerate(merged):
        # for idx in range(len(merged) * world_size):
        #     if idx % world_size == rank:
        #         expanded.append(merged[idx // world_size])
        #     else:
        #         expanded.append(_make_placeholder_like(merged[idx // world_size]))

        # return expanded
    
    def _distribute_groups_across_processes(self, group_items: List[Tuple]) -> List[Tuple]:
        """Distribute training groups across processes."""
        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index
        
        # Assign groups to this process using modulo
        process_groups = [group for i, group in enumerate(group_items) if i % num_processes == process_index]
        return process_groups
    
    def _broadcast_gradients(self, gradients: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Broadcast gradients from main process to all other processes."""
        if gradients is None:
            # Create empty dict with same structure as what main process has
            # We need to know the parameter names and shapes
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = torch.zeros_like(param.grad)
        
        # Broadcast each gradient tensor
        for name in gradients:
            gradients[name] = self.accelerator.broadcast(gradients[name], from_process=0)
        
        return gradients
    
    def _gather_similarities_from_all_processes(self, local_similarities: Dict[str, float]) -> Dict[str, float]:
        """Gather similarity results from all processes."""
        # Convert to list of (group_id, similarity) pairs for gathering
        local_items = [(group_id, sim) for group_id, sim in local_similarities.items()]
        
        # Gather all results on main process
        all_results = self.accelerator.gather_for_metrics(local_items)
        
        # Convert back to dict
        if self.accelerator.is_main_process:
            all_similarities = {}
            for process_results in all_results:
                for group_id, similarity in process_results:
                    all_similarities[group_id] = similarity
            return all_similarities
        else:
            return {}
    
    def save_model(self, save_path: str):
        """Save the model using accelerator's unwrap method."""
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            
            # Unwrap the model from FSDP
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
    
    def print_memory_stats(self):
        """Print current GPU memory usage for debugging."""
        if torch.cuda.is_available():
            device = self.accelerator.device
            if self.accelerator.is_main_process:
                print(f"\n=== Memory Stats for Process {self.accelerator.process_index} ===")
                print(f"GPU {device}: {torch.cuda.get_device_name(device)}")
                print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
                print(f"Max allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
                print("=" * 50)
                
            # Ensure all processes print their stats
            self.accelerator.wait_for_everyone()
            if not self.accelerator.is_main_process:
                print(f"Process {self.accelerator.process_index} - GPU {device}:")
                print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    def cleanup(self):
        """Clean up distributed resources."""
        if hasattr(self.accelerator, 'state') and self.accelerator.state.initialized:
            self.accelerator.end_training()


# Alias for backward compatibility
GRPOTrainerParallel = GRPOGradientAnalyzerParallel
