"""Hierarchical complete-contig model with an internal short-scale expert.

The v1.2 global branch remains intact.  A second RC-aware branch sees every
base for short contigs and a deterministic uniform synopsis for long contigs,
then contributes through a smooth length gate.  The external interface remains
one complete contig to one ``p_euk``; there is no external window voting.
"""

from __future__ import annotations

from math import log

import torch
from torch import nn

from .hierarchical import (
    ConvNeXtTCNBlock,
    HierarchicalKmerTCN,
    HierarchicalKmerTCNConfig,
    _group_count,
    _valid_mask,
)
from .sequence import MaskedGroupNorm1d, reverse_complement_batch


SHORT_SYNOPSIS_CAP = 4096
SHORT_BRANCH_CHANNELS = 128
SHORT_STEM_BRANCH_CHANNELS = 32
SHORT_STEM_KERNELS = (5, 9, 17, 33)
SHORT_STEM_STRIDE = 2
SHORT_DILATIONS = (1, 2, 4, 8, 16, 32)
SHORT_GATE_CENTER_BP = 3500.0
SHORT_GATE_LOG_TEMPERATURE = 0.45


def uniform_contig_synopsis(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    maximum_width: int = SHORT_SYNOPSIS_CAP,
    pad_token: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic full-span token samples without external windows."""

    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError("tokens and lengths have incompatible shapes")
    if maximum_width <= 0 or torch.any(lengths <= 0) or torch.any(
        lengths > tokens.shape[1]
    ):
        raise ValueError("synopsis inputs differ")
    synopsis_lengths = torch.clamp(lengths, max=maximum_width)
    width = int(synopsis_lengths.max().item())
    slots = torch.arange(width, device=tokens.device).unsqueeze(0)
    valid = slots < synopsis_lengths.unsqueeze(1)
    denominator = synopsis_lengths.unsqueeze(1).clamp_min(1)
    positions = torch.div(
        slots * lengths.unsqueeze(1), denominator, rounding_mode="floor"
    )
    positions = torch.minimum(positions, lengths.unsqueeze(1) - 1)
    gathered = torch.gather(tokens, 1, positions)
    gathered = torch.where(valid, gathered, torch.full_like(gathered, pad_token))
    return gathered, synopsis_lengths


def short_gate(lengths: torch.Tensor) -> torch.Tensor:
    """Smoothly emphasize the short expert while retaining long-contig use."""

    values = lengths.to(torch.float32).clamp_min(1)
    center = torch.tensor(SHORT_GATE_CENTER_BP, device=values.device)
    return torch.sigmoid(
        (torch.log(center) - torch.log(values)) / SHORT_GATE_LOG_TEMPERATURE
    )


class HierarchicalKmerTCNShortExpert(HierarchicalKmerTCN):
    """v1.2 global model plus an internal high-resolution short expert."""

    def __init__(self, config: HierarchicalKmerTCNConfig) -> None:
        super().__init__(config)
        self.short_stems = nn.ModuleList(
            nn.Conv1d(
                config.embedding_dimension,
                SHORT_STEM_BRANCH_CHANNELS,
                kernel,
                stride=SHORT_STEM_STRIDE,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in SHORT_STEM_KERNELS
        )
        self.short_projection = nn.Conv1d(
            SHORT_STEM_BRANCH_CHANNELS * len(SHORT_STEM_KERNELS),
            SHORT_BRANCH_CHANNELS,
            1,
        )
        self.short_stem_normalization = MaskedGroupNorm1d(
            SHORT_BRANCH_CHANNELS, groups=_group_count(SHORT_BRANCH_CHANNELS)
        )
        self.short_blocks = nn.ModuleList(
            ConvNeXtTCNBlock(
                SHORT_BRANCH_CHANNELS,
                kernel_size=5,
                dilation=dilation,
                dropout=config.dropout,
                layer_scale_initial=config.layer_scale_initial,
            )
            for dilation in SHORT_DILATIONS
        )
        self.short_final_normalization = MaskedGroupNorm1d(
            SHORT_BRANCH_CHANNELS, groups=_group_count(SHORT_BRANCH_CHANNELS)
        )
        global_dimension = int(self.shared[0].in_features)
        combined_dimension = global_dimension + SHORT_BRANCH_CHANNELS * 4 + 1
        self.shared = nn.Sequential(
            nn.Linear(combined_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dimension, config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.binary_head = nn.Linear(config.hidden_dimension, config.binary_classes)
        self.detail_head = nn.Linear(config.hidden_dimension, config.detail_classes)

    def _short_features(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        synopsis, synopsis_lengths = uniform_contig_synopsis(
            tokens,
            lengths,
            maximum_width=SHORT_SYNOPSIS_CAP,
            pad_token=self.config.pad_token,
        )
        embedded = self.embedding(synopsis).transpose(1, 2)
        hidden = torch.cat([stem(embedded) for stem in self.short_stems], dim=1)
        hidden = self.short_projection(hidden)
        reduced_lengths = torch.div(
            synopsis_lengths + SHORT_STEM_STRIDE - 1,
            SHORT_STEM_STRIDE,
            rounding_mode="floor",
        )
        mask = _valid_mask(reduced_lengths, hidden.shape[2], dtype=hidden.dtype)
        hidden = nn.functional.gelu(
            self.short_stem_normalization(hidden, mask)
        ) * mask
        for block in self.short_blocks:
            hidden = block(hidden, mask)
        hidden = self.short_final_normalization(hidden, mask) * mask

        denominator = reduced_lengths.to(hidden.dtype).clamp_min(1).unsqueeze(1)
        mean = hidden.sum(dim=2) / denominator
        centered = (hidden - mean.unsqueeze(2)) * mask
        standard_deviation = torch.sqrt(
            centered.square().sum(dim=2) / denominator + 1e-6
        )
        valid = mask.bool()
        maximum = hidden.masked_fill(
            ~valid, torch.finfo(hidden.dtype).min
        ).amax(dim=2)
        log_mean_exp = (
            torch.logsumexp(
                hidden.float().masked_fill(~valid, float("-inf")), dim=2
            )
            - torch.log(denominator.float())
        ).to(hidden.dtype)
        gate = short_gate(lengths).to(hidden.dtype).unsqueeze(1)
        pooled = torch.cat((mean, standard_deviation, maximum, log_mean_exp), dim=1)
        return pooled * gate, gate

    def forward_features(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        global_features = super().forward_features(tokens, lengths)
        local_features, gate = self._short_features(tokens, lengths)
        return torch.cat((global_features, local_features, gate), dim=1)


def reverse_complement_averaged_p_euk_v1_3(
    model: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    pad_token: int = 5,
) -> torch.Tensor:
    reverse = reverse_complement_batch(tokens, lengths, pad_token=pad_token)
    forward_logits, _ = model(tokens, lengths)
    reverse_logits, _ = model(reverse, lengths)
    forward = torch.softmax(forward_logits, dim=1)[:, 1]
    backward = torch.softmax(reverse_logits, dim=1)[:, 1]
    return (forward + backward) * 0.5
