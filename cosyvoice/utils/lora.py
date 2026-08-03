"""Small native LoRA implementation for CosyVoice adaptation.

This intentionally has no PEFT dependency. It injects adapters into the Qwen
projection layers while leaving the original weights available for the
unmodified base model.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merge_(self) -> None:
        self.base.weight.add_(self.lora_B @ self.lora_A, alpha=self.scaling)


TARGET_LINEAR_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _replace_target_linears(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in TARGET_LINEAR_NAMES:
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
            count += 1
        else:
            count += _replace_target_linears(child, rank, alpha, dropout)
    return count


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> int:
    """Freeze the model and inject LoRA into Qwen projections.

    The CosyVoice speech embedding and decoder remain trainable because they
    are the language-adaptation head; all other parameters remain frozen.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    count = _replace_target_linears(model, rank, alpha, dropout)
    train_head_names = ("llm_decoder", "speech_embedding")
    for name, parameter in model.named_parameters():
        if any(
            name == head
            or name.startswith(f"{head}.")
            or f".{head}." in name
            for head in train_head_names
        ):
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    model._lora_enabled = True
    model._lora_target_count = count
    return count


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only LoRA and adaptation-head weights for a small checkpoint."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if name not in {"step", "epoch"}]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected[:8]}")
    missing_trainable = [name for name in missing if name in state]
    if missing_trainable:
        raise RuntimeError(f"Could not load LoRA checkpoint keys: {missing_trainable[:8]}")


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
