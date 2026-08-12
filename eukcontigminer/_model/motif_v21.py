"""Independent-window correction for the v2.0 RC-motif expert.

Short contigs are encoded intact.  Long contigs use deterministic contiguous
full-span windows that are encoded independently; only window embeddings are
aggregated, so convolutions never cross distant source positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .window_plan import (
    independent_contiguous_window_plan,
)
from .hierarchical import _valid_mask
from .motif_v20 import (
    HierarchicalKmerTCNRCMotifExpert,
    RCInvariantMotifEncoder,
    RCMotifExpertConfig,
)
from .hierarchical import HierarchicalKmerTCNConfig


@dataclass(frozen=True)
class RCMotifWindowExpertConfig(RCMotifExpertConfig):
    maximum_windows: int = 4

    def validate(self) -> None:
        super().validate()
        if self.maximum_windows < 2:
            raise ValueError("motif window count must be at least two")
        if self.maximum_synopsis_width < self.maximum_windows:
            raise ValueError("motif synopsis budget is smaller than its window count")


class IndependentWindowRCMotifEncoder(RCInvariantMotifEncoder):
    """RC-symmetric encoder without cross-window artificial motifs."""

    def __init__(
        self,
        *,
        embedding: nn.Embedding,
        pad_token: int,
        config: RCMotifWindowExpertConfig,
    ) -> None:
        super().__init__(embedding=embedding, pad_token=pad_token, config=config)
        self.config = config
        dimension = self.pool.output_dimension
        self.window_projection = nn.Linear(dimension * 2, dimension)
        with torch.no_grad():
            self.window_projection.weight.zero_()
            identity = torch.eye(dimension)
            self.window_projection.weight[:, :dimension].copy_(0.5 * identity)
            self.window_projection.weight[:, dimension:].copy_(0.5 * identity)
            self.window_projection.bias.zero_()

    def _encode_contiguous(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode only genuine contiguous source sequences."""

        embedded = self.embedding(tokens).transpose(1, 2)
        hidden = torch.cat([stem(embedded) for stem in self.stems], dim=1)
        hidden = self.projection(hidden)
        reduced_lengths = torch.div(
            lengths + self.config.stem_stride - 1,
            self.config.stem_stride,
            rounding_mode="floor",
        )
        mask = _valid_mask(reduced_lengths, hidden.shape[2], dtype=hidden.dtype)
        hidden = nn.functional.gelu(self.stem_normalization(hidden, mask)) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_normalization(hidden, mask) * mask
        return self.pool(hidden, mask)

    def _aggregate_windows(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[2] != self.pool.output_dimension:
            raise ValueError("window feature tensor differs")
        mean = features.mean(dim=1)
        maximum = features.amax(dim=1)
        return self.window_projection(torch.cat((mean, maximum), dim=1))

    def _encode_one(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 2 or lengths.shape != (len(tokens),):
            raise ValueError("motif window inputs have incompatible shapes")
        plan = independent_contiguous_window_plan(
            lengths,
            maximum_total_bases=self.config.maximum_synopsis_width,
            maximum_windows=self.config.maximum_windows,
        )
        dimension = self.pool.output_dimension
        output: torch.Tensor | None = None

        short_rows = torch.nonzero(
            plan.full_sequence, as_tuple=False
        ).flatten()
        if short_rows.numel():
            short_lengths = lengths.index_select(0, short_rows)
            width = int(short_lengths.max().item())
            short_tokens = tokens.index_select(0, short_rows)[:, :width]
            short_features = self._encode_contiguous(short_tokens, short_lengths)
            short_features = self._aggregate_windows(short_features.unsqueeze(1))
            output = short_features.new_zeros((len(tokens), dimension))
            output = output.index_copy(0, short_rows, short_features)

        long_rows = torch.nonzero(
            ~plan.full_sequence, as_tuple=False
        ).flatten()
        if long_rows.numel():
            starts = plan.starts.index_select(0, long_rows)
            ends = plan.ends.index_select(0, long_rows)
            width = int((ends[:, 0] - starts[:, 0]).max().item())
            if not torch.all((ends - starts) == width):
                raise RuntimeError("long motif windows do not have a fixed width")
            offsets = torch.arange(width, device=tokens.device, dtype=torch.int64)
            indices = starts.unsqueeze(2) + offsets.view(1, 1, -1)
            source = tokens.index_select(0, long_rows)
            source = source.unsqueeze(1).expand(-1, self.config.maximum_windows, -1)
            windows = torch.gather(source, 2, indices)
            flat = windows.reshape(-1, width)
            flat_lengths = torch.full(
                (len(flat),), width, device=lengths.device, dtype=lengths.dtype
            )
            window_features = self._encode_contiguous(flat, flat_lengths)
            window_features = window_features.reshape(
                len(long_rows), self.config.maximum_windows, -1
            )
            long_features = self._aggregate_windows(window_features)
            if output is None:
                output = long_features.new_zeros((len(tokens), dimension))
            elif long_features.dtype != output.dtype:
                long_features = long_features.to(dtype=output.dtype)
            output = output.index_copy(0, long_rows, long_features)
        if output is None:
            output = self.embedding.weight.new_zeros((0, dimension))
        return output


class HierarchicalKmerTCNRCMotifWindowExpert(HierarchicalKmerTCNRCMotifExpert):
    """v2.0 expert with the scientifically corrected long-contig path."""

    def __init__(
        self,
        base_config: HierarchicalKmerTCNConfig,
        motif_config: RCMotifWindowExpertConfig = RCMotifWindowExpertConfig(),
    ) -> None:
        motif_config.validate()
        super().__init__(base_config, motif_config)
        self.motif_encoder = IndependentWindowRCMotifEncoder(
            embedding=self.embedding,
            pad_token=base_config.pad_token,
            config=motif_config,
        )
        self.motif_config = motif_config
