"""Large complete-contig RC-aware sequence model for EukContigMiner v5.1.

The network combines two complementary representations without cutting a
contig into voting windows:

* a strided, dilated ConvNeXt/TCN branch retains sequence order and long-range
  context; and
* exact canonical multi-k embedding bags retain the composition signal that
  was strongest in the locked Train-only k-mer experiments.

The training interface returns a binary head and a seven-class auxiliary head.
Deployment still exposes exactly one ``p_euk``; the auxiliary head is never a
public label.  Forward/reverse-complement probabilities are averaged at
inference, while the canonical k-mer branch is invariant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import nn

from .sequence import MaskedGroupNorm1d, reverse_complement_batch


@dataclass(frozen=True)
class HierarchicalKmerTCNConfig:
    vocabulary_size: int = 6
    pad_token: int = 5
    embedding_dimension: int = 24
    stem_branch_channels: int = 48
    stem_kernel_sizes: tuple[int, ...] = (7, 15, 31, 63)
    stem_stride: int = 8
    channels: int = 192
    residual_kernel_size: int = 7
    residual_dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    )
    kmer_orders: tuple[int, ...] = (3, 4, 5, 6)
    kmer_dimensions: tuple[int, ...] = (24, 32, 48, 64)
    hidden_dimension: int = 512
    dropout: float = 0.12
    layer_scale_initial: float = 1e-3
    maximum_length: int = 100_000
    binary_classes: int = 2
    detail_classes: int = 7

    def validate(self) -> None:
        positive = (
            self.vocabulary_size,
            self.embedding_dimension,
            self.stem_branch_channels,
            self.stem_stride,
            self.channels,
            self.residual_kernel_size,
            self.hidden_dimension,
            self.maximum_length,
            self.binary_classes,
            self.detail_classes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.pad_token < self.vocabulary_size:
            raise ValueError("padding token is outside vocabulary")
        if not self.stem_kernel_sizes or any(
            value <= 0 or value % 2 == 0 for value in self.stem_kernel_sizes
        ):
            raise ValueError("stem kernels must be positive odd integers")
        if self.residual_kernel_size % 2 == 0:
            raise ValueError("residual kernel must be odd")
        if not self.residual_dilations or any(
            value <= 0 for value in self.residual_dilations
        ):
            raise ValueError("residual dilations must be positive")
        if (
            not self.kmer_orders
            or len(self.kmer_orders) != len(self.kmer_dimensions)
            or any(order <= 0 or order > 8 for order in self.kmer_orders)
            or any(dimension <= 0 for dimension in self.kmer_dimensions)
        ):
            raise ValueError("canonical k-mer configuration is invalid")
        if len(set(self.kmer_orders)) != len(self.kmer_orders):
            raise ValueError("canonical k-mer orders must be unique")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 < self.layer_scale_initial <= 1.0:
            raise ValueError("layer scale must be in (0,1]")


def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups


def _valid_mask(
    lengths: torch.Tensor, width: int, *, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(1).to(dtype=dtype)


def canonical_kmer_indices(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    order: int,
    invalid_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical base-4 k-mer ids and their validity mask."""

    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError("tokens and lengths have incompatible shapes")
    if order <= 0 or invalid_index < 4**order:
        raise ValueError("k-mer order or invalid index differs")
    if tokens.shape[1] < order:
        empty = torch.empty(
            (len(tokens), 0), dtype=torch.long, device=tokens.device
        )
        return empty, empty.bool()
    windows = tokens.unfold(1, order, 1)
    positions = torch.arange(windows.shape[1], device=tokens.device).unsqueeze(0)
    valid = (positions + order <= lengths.unsqueeze(1)) & torch.all(
        windows < 4, dim=2
    )
    values = windows.long().clamp(max=3)
    powers_forward = torch.tensor(
        [4 ** (order - 1 - index) for index in range(order)],
        dtype=torch.long,
        device=tokens.device,
    )
    powers_reverse = torch.tensor(
        [4**index for index in range(order)],
        dtype=torch.long,
        device=tokens.device,
    )
    forward = torch.sum(values * powers_forward, dim=2)
    reverse = torch.sum((3 - values) * powers_reverse, dim=2)
    canonical = torch.minimum(forward, reverse)
    canonical = torch.where(
        valid, canonical, torch.full_like(canonical, invalid_index)
    )
    return canonical, valid


class ConvNeXtTCNBlock(nn.Module):
    """Padding-aware depthwise long-context block with guarded residual scale."""

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
        self.pointwise_in = nn.Conv1d(channels, channels * 4, 1)
        self.pointwise_out = nn.Conv1d(channels * 4, channels, 1)
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1), layer_scale_initial)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(values, mask)
        hidden = self.depthwise(hidden)
        hidden = nn.functional.gelu(self.pointwise_in(hidden))
        hidden = self.dropout(self.pointwise_out(hidden))
        return (values + self.layer_scale * hidden) * mask


class HierarchicalKmerTCN(nn.Module):
    """Complete-contig multi-scale TCN with canonical multi-k composition."""

    def __init__(self, config: HierarchicalKmerTCNConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
            padding_idx=config.pad_token,
        )
        self.stems = nn.ModuleList(
            nn.Conv1d(
                config.embedding_dimension,
                config.stem_branch_channels,
                kernel_size,
                stride=config.stem_stride,
                padding=kernel_size // 2,
                bias=False,
            )
            for kernel_size in config.stem_kernel_sizes
        )
        stem_channels = config.stem_branch_channels * len(config.stem_kernel_sizes)
        self.stem_projection = nn.Conv1d(stem_channels, config.channels, 1)
        self.stem_normalization = MaskedGroupNorm1d(
            config.channels, groups=_group_count(config.channels)
        )
        self.blocks = nn.ModuleList(
            ConvNeXtTCNBlock(
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
        self.kmer_embeddings = nn.ModuleDict(
            {
                str(order): nn.EmbeddingBag(
                    4**order + 1,
                    dimension,
                    mode="sum",
                    padding_idx=4**order,
                )
                for order, dimension in zip(
                    config.kmer_orders, config.kmer_dimensions, strict=True
                )
            }
        )
        feature_dimension = (
            config.channels * 4 + sum(config.kmer_dimensions) + 6
        )
        self.shared = nn.Sequential(
            nn.Linear(feature_dimension, config.hidden_dimension),
            nn.LayerNorm(config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dimension, config.hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.binary_head = nn.Linear(config.hidden_dimension, config.binary_classes)
        self.detail_head = nn.Linear(config.hidden_dimension, config.detail_classes)

    def downsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            lengths + self.config.stem_stride - 1,
            self.config.stem_stride,
            rounding_mode="floor",
        )

    def _canonical_kmer_pool(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        pooled = []
        for order in self.config.kmer_orders:
            padding_index = 4**order
            identifiers, valid = canonical_kmer_indices(
                tokens,
                lengths,
                order=order,
                invalid_index=padding_index,
            )
            embedding = self.kmer_embeddings[str(order)]
            if identifiers.shape[1] == 0:
                pooled.append(
                    embedding.weight[:1].expand(len(tokens), -1) * 0.0
                )
                continue
            flattened = identifiers.reshape(-1)
            offsets = torch.arange(
                0,
                flattened.numel(),
                identifiers.shape[1],
                dtype=torch.long,
                device=tokens.device,
            )
            summed = embedding(flattened, offsets)
            counts = valid.sum(dim=1).clamp_min(1).to(summed.dtype).unsqueeze(1)
            pooled.append(summed / counts)
        return torch.cat(pooled, dim=1)

    def _base_composition(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        inside = positions < lengths.unsqueeze(1)
        values = []
        denominator = lengths.to(torch.float32).clamp_min(1).unsqueeze(1)
        for token in range(4):
            values.append(torch.sum(inside & (tokens == token), dim=1, keepdim=True))
        values.append(torch.sum(inside & (tokens >= 4), dim=1, keepdim=True))
        composition = torch.cat(values, dim=1).to(torch.float32) / denominator
        normalized_length = (
            torch.log(lengths.to(torch.float32)) / log(self.config.maximum_length)
        ).unsqueeze(1)
        return torch.cat((composition, normalized_length), dim=1)

    def forward_features(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.shape != (len(tokens),):
            raise ValueError("tokens and lengths have incompatible shapes")
        if tokens.shape[1] > self.config.maximum_length:
            raise ValueError("input exceeds configured maximum length")
        if torch.any(lengths <= 0) or torch.any(lengths > tokens.shape[1]):
            raise ValueError("sequence lengths are invalid")

        input_mask = _valid_mask(lengths, tokens.shape[1], dtype=torch.bool)
        masked_tokens = torch.where(
            input_mask.squeeze(1),
            tokens,
            torch.full_like(tokens, self.config.pad_token),
        )
        embedded = self.embedding(masked_tokens).transpose(1, 2)
        hidden = torch.cat([stem(embedded) for stem in self.stems], dim=1)
        hidden = self.stem_projection(hidden)
        reduced_lengths = self.downsampled_lengths(lengths)
        mask = _valid_mask(reduced_lengths, hidden.shape[2], dtype=hidden.dtype)
        hidden = nn.functional.gelu(self.stem_normalization(hidden, mask)) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_normalization(hidden, mask) * mask

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
        kmer = self._canonical_kmer_pool(masked_tokens, lengths).to(hidden.dtype)
        composition = self._base_composition(masked_tokens, lengths).to(hidden.dtype)
        return torch.cat(
            (mean, standard_deviation, maximum, log_mean_exp, kmer, composition),
            dim=1,
        )

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(self.forward_features(tokens, lengths))
        return self.binary_head(shared), self.detail_head(shared)


def reverse_complement_averaged_p_euk(
    model: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    *,
    pad_token: int = 5,
) -> torch.Tensor:
    """Return exact forward/RC averaged binary ``p_euk``."""

    reverse = reverse_complement_batch(tokens, lengths, pad_token=pad_token)
    forward_logits, _forward_detail = model(tokens, lengths)
    reverse_logits, _reverse_detail = model(reverse, lengths)
    forward = torch.softmax(forward_logits, dim=1)[:, 1]
    backward = torch.softmax(reverse_logits, dim=1)[:, 1]
    return (forward + backward) * 0.5
