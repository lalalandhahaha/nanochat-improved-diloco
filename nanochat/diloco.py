"""
DiLoCo (Distributed Low-Communication) training support.

DiLoCo enables distributed training with reduced communication by:
1. Running inner optimizers (AdamW, Muon, etc.) locally for H steps
2. Synchronizing and averaging model weights across workers
3. Applying outer optimizer (SGD with momentum) on the averaged pseudo-gradients

Reference: https://arxiv.org/abs/2311.08105
"""
import torch
import time
import torch.distributed as dist
from typing import List, Optional


class DiLoCoWrapper:
    """
    Wrapper for DiLoCo distributed training.

    DiLoCo performs:
    - Inner optimization: standard optimizers (AdamW, Muon) run locally for H steps
    - Outer optimization: SGD with Nesterov momentum on pseudo-gradients every H steps

    Args:
        inner_optimizers: List of inner optimizers (e.g., [adamw_optimizer, muon_optimizer])
        model: The model being trained
        outer_lr: Learning rate for outer SGD optimizer
        outer_momentum: Momentum coefficient for outer SGD optimizer (default: 0.9)
        H: Number of inner steps before outer synchronization (communication interval)
        nesterov: Whether to use Nesterov momentum in outer optimizer (default: True)
        pre_sync_callback: Optional callback function called before all_reduce, for analysis
    """

    def __init__(
        self,
        inner_optimizers: List[torch.optim.Optimizer],
        model: torch.nn.Module,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        H: int = 500,
        nesterov: bool = True,
        pre_sync_callback: Optional[callable] = None,
    ):
        self.inner_optimizers = inner_optimizers
        self.model = model
        self.H = H
        self.pre_sync_callback = pre_sync_callback
        
        # Collect all parameters from inner optimizers in order
        # This ensures outer optimizer has same parameter order as inner optimizers
        all_params = []
        for opt in inner_optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.requires_grad:
                        all_params.append(param)
        
        # Create outer optimizer with the same parameters in the same order
        self.outer_optimizer = torch.optim.SGD(
            all_params,
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=nesterov,
        )
        
        # Track inner steps
        self.inner_step_count = 0
        
        # Store offloaded parameters (CPU clones) for computing pseudo-gradients
        self.params_offloaded = self._get_offloaded_params()
    
    @property
    def param_groups(self):
        """
        Flattened view of the inner optimizers' param groups, so callers can drive
        LR / momentum / weight-decay schedules on the wrapper exactly as they would on
        a bare optimizer. Note these are the *inner* groups; the outer SGD is untouched.
        """
        return [group for opt in self.inner_optimizers for group in opt.param_groups]

    def _get_offloaded_params(self):
        """CPU snapshot of outer optimizer parameters."""
        return [
            param.data.detach().clone().to("cpu")
            for group in self.outer_optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]
    
    def step(self):
        """
        Perform one step of DiLoCo training.
        
        This should be called after accumulating gradients and before zeroing them.
        It will:
        1. Step all inner optimizers
        2. Every H steps, perform outer optimization (sync + SGD with momentum)
        """
        # Step all inner optimizers
        for opt in self.inner_optimizers:
            opt.step()
        
        self.inner_step_count += 1
        
        # Check if it's time for outer optimization
        if self.inner_step_count % self.H == 0:
            self._outer_step()
    
    @torch.no_grad()
    def _outer_step(self):
        """
        Perform outer optimization step on device (parity with original):
        1. [CALLBACK] Call pre_sync_callback if set (for analysis before sync)
        2. Compute pseudo-gradients: g = (w_old - w_new) on device
        3. Average pseudo-gradients across workers via all_reduce (AVG)
        4. Restore params to offloaded values (w_old)
        5. Apply outer optimizer (SGD with momentum)
        6. Zero outer gradients
        7. Update offloaded params snapshot
        """
        # Call pre-sync callback BEFORE any synchronization
        if self.pre_sync_callback is not None:
            self.pre_sync_callback()

        # Get parameters from outer optimizer (already in correct order)
        main_params = [
            param
            for opt in self.inner_optimizers
            for group in opt.param_groups
            for param in group["params"]
            if param.requires_grad
        ]

        # Compute pseudo-grads and sync across workers on device
        for param_offloaded_cpu, param in zip(self.params_offloaded, main_params):
            param_offloaded_on_device = param_offloaded_cpu.to(param.device, non_blocking=True)
            # g = w_old - w_new
            param.grad = param_offloaded_on_device - param.data
            if dist.is_initialized():
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
            # Restore weights to w_old before applying the outer step
            param.data.copy_(param_offloaded_on_device)

        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

        # Refresh offloaded snapshot after the outer step
        self.params_offloaded = self._get_offloaded_params()
    
    def sync_if_pending(self):
        """
        Force an outer step if the last inner step did not land on an H boundary.

        Between outer steps every rank holds its own locally-drifted weights, so anything
        that reads the weights globally (final eval, checkpointing from rank 0) sees rank 0's
        local copy rather than the synchronized model. Call this before such a read.
        No-op when the weights are already synchronized.
        """
        if self.inner_step_count % self.H != 0:
            self._outer_step()

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all inner optimizers."""
        for opt in self.inner_optimizers:
            for group in opt.param_groups:
                for param in group['params']:
                    if set_to_none:
                        param.grad = None
                    else:
                        if param.grad is not None:
                            param.grad.zero_()
    
    def state_dict(self):
        """
        Return state dict for checkpointing.
        Includes inner optimizer states, outer optimizer state, and offloaded params.
        """
        return {
            'inner_optimizers': [opt.state_dict() for opt in self.inner_optimizers],
            'outer_optimizer': self.outer_optimizer.state_dict(),
            'params_offloaded': self.params_offloaded,
            'inner_step_count': self.inner_step_count,
            'H': self.H,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        for opt, opt_state in zip(self.inner_optimizers, state_dict['inner_optimizers']):
            opt.load_state_dict(opt_state)
        self.outer_optimizer.load_state_dict(state_dict['outer_optimizer'])
        self.params_offloaded = state_dict['params_offloaded']
        self.inner_step_count = state_dict['inner_step_count']
        self.H = state_dict.get('H', self.H)
