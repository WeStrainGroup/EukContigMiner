from __future__ import annotations
import torch
from torch import nn
PAD_TOKEN = 5

class MaskedGroupNorm1d(nn.Module):
    """Group normalization over channels and valid sequence positions only."""

    def __init__(self, channels: int, *, groups: int, eps: float=1e-05) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError('invalid masked GroupNorm dimensions')
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or mask.shape != (len(values), 1, values.shape[2]):
            raise ValueError('masked GroupNorm inputs have invalid shapes')
        source_dtype = values.dtype
        source = values.float() if source_dtype in {torch.float16, torch.bfloat16} else values
        valid = mask.to(dtype=source.dtype)
        grouped = source.reshape(len(source), self.groups, self.channels_per_group, source.shape[2])
        group_mask = valid.unsqueeze(1)
        count = (valid.sum(dim=2, keepdim=False) * self.channels_per_group).clamp_min(1)
        mean = (grouped * group_mask).sum(dim=(2, 3)) / count
        centered = grouped - mean.unsqueeze(2).unsqueeze(3)
        variance = (centered.square() * group_mask).sum(dim=(2, 3)) / count
        normalized = centered * torch.rsqrt(variance.unsqueeze(2).unsqueeze(3) + self.eps)
        normalized = normalized.reshape_as(source).to(source_dtype)
        weight = self.weight.to(source_dtype).view(1, -1, 1)
        bias = self.bias.to(source_dtype).view(1, -1, 1)
        return (normalized * weight + bias) * mask

def reverse_complement_batch(tokens: torch.Tensor, lengths: torch.Tensor, *, pad_token: int=PAD_TOKEN) -> torch.Tensor:
    """Reverse-complement a padded token batch on its current device."""
    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError('tokens and lengths have incompatible shapes')
    width = tokens.shape[1]
    if torch.any(lengths <= 0) or torch.any(lengths > width):
        raise ValueError('sequence lengths are invalid')
    positions = torch.arange(width, device=tokens.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    source = lengths.unsqueeze(1) - 1 - positions
    source = source.clamp(min=0, max=max(0, width - 1))
    reversed_tokens = tokens.gather(1, source)
    complement = torch.tensor([3, 2, 1, 0, 4, pad_token], dtype=torch.long, device=tokens.device)
    complemented = complement[reversed_tokens.long()]
    return torch.where(valid, complemented.to(tokens.dtype), torch.full_like(tokens, pad_token))
