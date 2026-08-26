"""Small residual head on frozen previous-model representations."""

from __future__ import annotations

import torch
from torch import nn


class FrozenResidualHead(nn.Module):
    """Learn a bounded-complexity correction around the unified old margin."""

    def __init__(
        self,
        input_dimension: int = 771,
        hidden_dimension: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dimension < 1 or hidden_dimension < 1:
            raise ValueError("head dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.dropout = dropout
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        standardized_features: torch.Tensor,
        unified_margin: torch.Tensor,
    ) -> torch.Tensor:
        if standardized_features.ndim != 2:
            raise ValueError("standardized features must be a matrix")
        if standardized_features.shape[1] != self.input_dimension:
            raise ValueError("frozen feature dimension differs from checkpoint")
        if unified_margin.shape != (len(standardized_features),):
            raise ValueError("unified margin shape differs")
        return unified_margin + self.residual(standardized_features).squeeze(1)
