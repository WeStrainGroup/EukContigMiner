"""Exact-RC-symmetric raw-base motif expert for EukContigMiner v5.1.

The selected v1.3 model already contains a high-resolution short branch, but
its global statistical pooling can dilute sparse short-contig motifs.  This
module adds a genuinely different residual raw-base encoder with learned
multi-query pooling.  The same encoder sees a contig and its reverse
complement; symmetric mean and absolute-difference features make the new
representation exactly invariant to strand orientation.

The public contract is unchanged: one complete contig produces one binary
``p_euk``.  The parent handles the full length range.  The new branch sees all
bases up to a fixed synopsis cap and a deterministic full-span synopsis for
longer contigs, then contributes through a smooth short-length gate.  Its
fusion head is initialized to zero, so a locked v1.3 parent is preserved
exactly at initialization while independent auxiliary heads provide an
immediate training signal to the new encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import torch
from torch import nn

from .short_expert import (
    HierarchicalKmerTCNShortExpert,
    uniform_contig_synopsis,
)
from .hierarchical import (
    HierarchicalKmerTCNConfig,
    _group_count,
    _valid_mask,
)
from .sequence import MaskedGroupNorm1d, reverse_complement_batch


@dataclass(frozen=True)
class RCMotifExpertConfig:
    maximum_synopsis_width: int = 4096
    stem_kernel_sizes: tuple[int, ...] = (9, 17, 33, 65)
    stem_branch_channels: int = 24
    stem_stride: int = 4
    channels: int = 128
    residual_kernel_size: int = 7
    residual_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    attention_heads: int = 4
    hidden_dimension: int = 256
    dropout: float = 0.10
    layer_scale_initial: float = 1e-3
    gate_center_bp: float = 3000.0
    gate_log_temperature: float = 0.45
    fusion_raw_weight_initial: float = -3.0

    def validate(self) -> None:
        positive = (
            self.maximum_synopsis_width,
            self.stem_branch_channels,
            self.stem_stride,
            self.channels,
            self.residual_kernel_size,
            self.attention_heads,
            self.hidden_dimension,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("RC motif expert dimensions must be positive")
        if not self.stem_kernel_sizes or any(
            value <= 0 or value % 2 == 0 for value in self.stem_kernel_sizes
        ):
            raise ValueError("RC motif stem kernels must be positive odd integers")
        if self.residual_kernel_size % 2 == 0:
            raise ValueError("RC motif residual kernel must be odd")
        if not self.residual_dilations or any(
            value <= 0 for value in self.residual_dilations
        ):
            raise ValueError("RC motif residual dilations must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("RC motif dropout must be in [0,1)")
        if not 0.0 < self.layer_scale_initial <= 1.0:
            raise ValueError("RC motif layer scale must be in (0,1]")
        if self.gate_center_bp <= 0.0 or self.gate_log_temperature <= 0.0:
            raise ValueError("RC motif length gate parameters must be positive")


class RCMotifResidualBlock(nn.Module):
    """Padding-aware gated residual motif block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        layer_scale_initial: float,
    ) -> None:
        super().__init__()
        self.normalization = MaskedGroupNorm1d(
            channels, groups=_group_count(channels)
        )
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise_in = nn.Conv1d(channels, channels * 2, 1)
        self.pointwise_out = nn.Conv1d(channels, channels, 1)
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1), float(layer_scale_initial))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(values, mask)
        hidden = self.depthwise(hidden)
        hidden = nn.functional.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.dropout(self.pointwise_out(hidden))
        return (values + self.layer_scale * hidden) * mask


class MultiQueryMotifPool(nn.Module):
    """Pool both abundant and sparse motif evidence with learned queries."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0:
            raise ValueError("motif pooling dimensions must be positive")
        self.channels = channels
        self.heads = heads
        self.queries = nn.Parameter(torch.empty(heads, channels))
        nn.init.normal_(self.queries, mean=0.0, std=1.0 / sqrt(channels))

    @property
    def output_dimension(self) -> int:
        return self.channels * (self.heads + 3)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or mask.shape != (
            len(values),
            1,
            values.shape[2],
        ):
            raise ValueError("motif pooling inputs have incompatible shapes")
        valid = mask.bool()
        denominator = valid.sum(dim=2).clamp_min(1).to(values.dtype)
        mean = (values * mask).sum(dim=2) / denominator
        centered = (values - mean.unsqueeze(2)) * mask
        standard_deviation = torch.sqrt(
            centered.square().sum(dim=2) / denominator + 1e-6
        )
        maximum = values.masked_fill(
            ~valid, torch.finfo(values.dtype).min
        ).amax(dim=2)

        attention_logits = torch.einsum(
            "bcw,hc->bhw", values.float(), self.queries.float()
        ) / sqrt(self.channels)
        attention_logits = attention_logits.masked_fill(
            ~valid.expand(-1, self.heads, -1), float("-inf")
        )
        attention = torch.softmax(attention_logits, dim=2).to(values.dtype)
        queried = torch.einsum("bhw,bcw->bhc", attention, values)
        return torch.cat(
            (mean, standard_deviation, maximum, queried.flatten(1)), dim=1
        )


class RCInvariantMotifEncoder(nn.Module):
    """Shared raw-base encoder with exactly symmetric strand aggregation."""

    def __init__(
        self,
        *,
        embedding: nn.Embedding,
        pad_token: int,
        config: RCMotifExpertConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.embedding = embedding
        self.pad_token = int(pad_token)
        self.config = config
        embedding_dimension = int(embedding.embedding_dim)
        self.stems = nn.ModuleList(
            nn.Conv1d(
                embedding_dimension,
                config.stem_branch_channels,
                kernel,
                stride=config.stem_stride,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in config.stem_kernel_sizes
        )
        self.projection = nn.Conv1d(
            config.stem_branch_channels * len(config.stem_kernel_sizes),
            config.channels,
            1,
        )
        self.stem_normalization = MaskedGroupNorm1d(
            config.channels, groups=_group_count(config.channels)
        )
        self.blocks = nn.ModuleList(
            RCMotifResidualBlock(
                config.channels,
                kernel_size=config.residual_kernel_size,
                dilation=dilation,
                dropout=config.dropout,
                layer_scale_initial=config.layer_scale_initial,
            )
            for dilation in config.residual_dilations
        )
        self.final_normalization = MaskedGroupNorm1d(
            config.channels, groups=_group_count(config.channels)
        )
        self.pool = MultiQueryMotifPool(config.channels, config.attention_heads)

    @property
    def output_dimension(self) -> int:
        # mean(strand features), absolute strand difference, and log length.
        return self.pool.output_dimension * 2 + 1

    def _encode_one(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        synopsis, synopsis_lengths = uniform_contig_synopsis(
            tokens,
            lengths,
            maximum_width=self.config.maximum_synopsis_width,
            pad_token=self.pad_token,
        )
        embedded = self.embedding(synopsis).transpose(1, 2)
        hidden = torch.cat([stem(embedded) for stem in self.stems], dim=1)
        hidden = self.projection(hidden)
        reduced_lengths = torch.div(
            synopsis_lengths + self.config.stem_stride - 1,
            self.config.stem_stride,
            rounding_mode="floor",
        )
        mask = _valid_mask(reduced_lengths, hidden.shape[2], dtype=hidden.dtype)
        hidden = nn.functional.gelu(
            self.stem_normalization(hidden, mask)
        ) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_normalization(hidden, mask) * mask
        return self.pool(hidden, mask)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.shape != (len(tokens),):
            raise ValueError("tokens and lengths have incompatible shapes")
        reverse = reverse_complement_batch(
            tokens, lengths, pad_token=self.pad_token
        )
        forward_features = self._encode_one(tokens, lengths)
        reverse_features = self._encode_one(reverse, lengths)
        symmetric_mean = 0.5 * (forward_features + reverse_features)
        symmetric_difference = torch.abs(forward_features - reverse_features)
        normalized_length = (
            torch.log(lengths.to(torch.float32))
            / log(float(self.config.maximum_synopsis_width))
        ).to(forward_features.dtype).unsqueeze(1)
        return torch.cat(
            (symmetric_mean, symmetric_difference, normalized_length), dim=1
        )


def motif_length_gate(
    lengths: torch.Tensor, config: RCMotifExpertConfig
) -> torch.Tensor:
    values = lengths.to(torch.float32).clamp_min(1)
    center = torch.as_tensor(config.gate_center_bp, device=values.device)
    return torch.sigmoid(
        (torch.log(center) - torch.log(values)) / config.gate_log_temperature
    )


class HierarchicalKmerTCNRCMotifExpert(HierarchicalKmerTCNShortExpert):
    """Locked v1.3 parent plus an exact-RC raw-base motif expert."""

    def __init__(
        self,
        base_config: HierarchicalKmerTCNConfig,
        motif_config: RCMotifExpertConfig = RCMotifExpertConfig(),
    ) -> None:
        super().__init__(base_config)
        motif_config.validate()
        self.motif_config = motif_config
        self.return_motif_aux = False
        self.motif_encoder = RCInvariantMotifEncoder(
            embedding=self.embedding,
            pad_token=base_config.pad_token,
            config=motif_config,
        )
        self.motif_shared = nn.Sequential(
            nn.Linear(
                self.motif_encoder.output_dimension,
                motif_config.hidden_dimension,
            ),
            nn.LayerNorm(motif_config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(motif_config.dropout),
            nn.Linear(motif_config.hidden_dimension, motif_config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(motif_config.dropout),
        )
        # Auxiliary heads train the encoder immediately.  The separate fusion
        # head is zero-initialized, which preserves every parent logit exactly.
        self.motif_binary_aux = nn.Linear(motif_config.hidden_dimension, 2)
        self.motif_detail_aux = nn.Linear(
            motif_config.hidden_dimension, base_config.detail_classes
        )
        self.motif_margin_correction = nn.Linear(
            motif_config.hidden_dimension, 1
        )
        nn.init.zeros_(self.motif_margin_correction.weight)
        nn.init.zeros_(self.motif_margin_correction.bias)
        self.motif_fusion_raw_weight = nn.Parameter(
            torch.tensor(float(motif_config.fusion_raw_weight_initial))
        )

    def forward_with_motif_aux(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_features = super().forward_features(tokens, lengths)
        parent_shared = self.shared(parent_features)
        parent_binary = self.binary_head(parent_shared)
        parent_detail = self.detail_head(parent_shared)

        motif = self.motif_shared(self.motif_encoder(tokens, lengths))
        motif_binary = self.motif_binary_aux(motif)
        motif_detail = self.motif_detail_aux(motif)
        correction = self.motif_margin_correction(motif).squeeze(1)
        gate = motif_length_gate(lengths, self.motif_config).to(correction.dtype)
        weight = nn.functional.softplus(self.motif_fusion_raw_weight)
        margin_correction = weight * gate * correction
        binary = parent_binary + torch.stack(
            (-0.5 * margin_correction, 0.5 * margin_correction), dim=1
        )
        return binary, parent_detail, motif_binary, motif_detail

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        binary, detail, motif_binary, motif_detail = self.forward_with_motif_aux(
            tokens, lengths
        )
        if self.return_motif_aux:
            return binary, detail, motif_binary, motif_detail
        return binary, detail


def reverse_complement_averaged_p_euk_v2_0(
    model: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    pad_token: int = 5,
) -> torch.Tensor:
    """Return the single public p_euk with exact outer RC averaging."""

    reverse = reverse_complement_batch(tokens, lengths, pad_token=pad_token)
    forward_logits, _ = model(tokens, lengths)
    reverse_logits, _ = model(reverse, lengths)
    return 0.5 * (
        torch.softmax(forward_logits.float(), dim=1)[:, 1]
        + torch.softmax(reverse_logits.float(), dim=1)[:, 1]
    )
