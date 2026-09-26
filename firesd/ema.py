"""EMA shadow and safe in-place swap for trainable LoRA tensors."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class LoRAEMA:
    """Float32 EMA shadow of only trainable parameters.

    This is intentionally one model object: we temporarily copy the EMA adapter tensors
    into the same PEFT model for detached teacher forwards, then restore the student.
    """

    def __init__(self, model: torch.nn.Module, alpha: float):
        self.alpha = float(alpha)
        self.names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not self.names:
            raise ValueError("EMA found no trainable parameters; did LoRA attach correctly?")
        params = dict(model.named_parameters())
        self.shadow = {name: params[name].detach().float().clone() for name in self.names}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        params = dict(model.named_parameters())
        for name in self.names:
            self.shadow[name].mul_(1.0 - self.alpha).add_(params[name].detach().float(), alpha=self.alpha)

    @contextmanager
    @torch.no_grad()
    def swap_into(self, model: torch.nn.Module) -> Iterator[None]:
        params = dict(model.named_parameters())
        backup = {name: params[name].detach().clone() for name in self.names}
        try:
            for name in self.names:
                params[name].copy_(self.shadow[name].to(device=params[name].device, dtype=params[name].dtype))
            yield
        finally:
            for name in self.names:
                params[name].copy_(backup[name])
