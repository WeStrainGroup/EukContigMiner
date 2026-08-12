"""Directional residual fusion for the independent-window RC-motif expert.

The v2.1.2 signed residual learned useful short-Euk signal but could also move
short negatives upward and long positives downward.  This revision makes the
two corrections structural: the short-gated branch can only rescue Euk and
the all-length branch can only veto Euk.  A straight-through non-negative
projection keeps the locked v1.3 parent bit-exact at initialization without
removing the gradient from either zero-initialized head.
"""

from __future__ import annotations

import torch
from torch import nn

from .hierarchical import HierarchicalKmerTCNConfig
from .motif_v20 import motif_length_gate
from .motif_v21 import (
    HierarchicalKmerTCNRCMotifWindowExpert,
    RCMotifWindowExpertConfig,
)


class HierarchicalKmerTCNRCMotifDirectionalExpert(
    HierarchicalKmerTCNRCMotifWindowExpert
):
    """Independent-window expert with short rescue and global veto paths."""

    def __init__(
        self,
        base_config: HierarchicalKmerTCNConfig,
        motif_config: RCMotifWindowExpertConfig = RCMotifWindowExpertConfig(),
    ) -> None:
        super().__init__(base_config, motif_config)
        hidden = motif_config.hidden_dimension
        del self.motif_margin_correction
        del self.motif_fusion_raw_weight
        self.motif_short_rescue_correction = nn.Linear(hidden, 1)
        self.motif_global_veto_correction = nn.Linear(hidden, 1)
        nn.init.zeros_(self.motif_short_rescue_correction.weight)
        nn.init.zeros_(self.motif_short_rescue_correction.bias)
        nn.init.zeros_(self.motif_global_veto_correction.weight)
        nn.init.zeros_(self.motif_global_veto_correction.bias)
        self.motif_fusion_raw_weights = nn.Parameter(
            torch.full((2,), float(motif_config.fusion_raw_weight_initial))
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
        short_rescue = straight_through_nonnegative(
            self.motif_short_rescue_correction(motif).squeeze(1)
        )
        global_veto = straight_through_nonnegative(
            self.motif_global_veto_correction(motif).squeeze(1)
        )
        short_gate = motif_length_gate(lengths, self.motif_config).to(
            short_rescue.dtype
        )
        weights = nn.functional.softplus(self.motif_fusion_raw_weights)
        margin_correction = (
            weights[0] * short_gate * short_rescue
            - weights[1] * global_veto
        )
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


def straight_through_nonnegative(value: torch.Tensor) -> torch.Tensor:
    """Project the forward value to ``[0, inf)`` with identity gradient.

    The detached projection supplies the directional forward semantics while
    ``value - value.detach()`` supplies a unit local gradient.  Consequently
    zero-initialized correction heads preserve the parent exactly and can
    still start learning on the first optimization step.
    """

    return torch.relu(value).detach() + value - value.detach()
