"""Shared whole-contig DNA ensemble used by training and deployment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .dna_inference import whole_contig_base_features
from .frozen_head import FrozenResidualHead


ENSEMBLE_WEIGHTS = (0.8, 0.2)
DETAIL_EUK_WITH_ORGANELLE = (2, 3, 4, 5, 6)
DETAIL_LABEL = {
    "Archaea": 0,
    "Bacteria": 1,
    "Fungi": 2,
    "Metazoa": 3,
    "Organelle": 4,
    "Other_Eukaryota": 5,
    "Viridiplantae": 6,
}


def load_frozen_head(
    path: Path, hidden_dimension: int
) -> tuple[FrozenResidualHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "eukcontigminer.frozen_residual_head.v1"
        or payload.get("input_dimension") != 771
        or payload.get("hidden_dimension") != hidden_dimension
    ):
        raise ValueError(f"unexpected frozen residual head: {path}")
    head = FrozenResidualHead(
        input_dimension=771,
        hidden_dimension=hidden_dimension,
        dropout=float(payload["dropout"]),
    )
    head.load_state_dict(payload["model_state_dict"], strict=True)
    return head, payload


class TrainableDNAEnsemble(nn.Module):
    """Exact h64/h32 logit ensemble with selectable trainable DNA layers."""

    def __init__(
        self,
        base: nn.Module,
        heads: tuple[FrozenResidualHead, FrozenResidualHead],
        head_payloads: tuple[dict[str, Any], dict[str, Any]],
        reverse_complement_batch: Callable,
        motif_length_gate: Callable,
        unfreeze_scope: str,
    ) -> None:
        super().__init__()
        self.base = base
        self.heads = nn.ModuleList(heads)
        self.reverse_complement_batch = reverse_complement_batch
        self.motif_length_gate = motif_length_gate
        self.full_backbone = unfreeze_scope == "full"
        for index, payload in enumerate(head_payloads):
            self.register_buffer(
                f"feature_mean_{index}", payload["feature_mean"].float()
            )
            self.register_buffer(
                f"feature_std_{index}",
                payload["feature_standard_deviation"].float(),
            )
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        core_modules = [
            self.base.final_normalization,
            self.base.motif_encoder.final_normalization,
            *self.base.blocks[-2:],
            *self.base.motif_encoder.blocks[-2:],
        ]
        if self.full_backbone:
            self.trainable_base_modules = [self.base]
        elif unfreeze_scope == "heads-only":
            self.trainable_base_modules = []
        elif unfreeze_scope == "last2":
            self.trainable_base_modules = [
                self.base.shared,
                self.base.motif_shared,
                self.base.motif_encoder.window_projection,
                *core_modules,
            ]
        elif unfreeze_scope == "tcn-last2":
            self.trainable_base_modules = core_modules
        else:
            raise ValueError(f"unsupported unfreeze scope: {unfreeze_scope}")
        for module in self.trainable_base_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(False)
        if mode:
            self.heads.train(True)
            if self.full_backbone:
                self.base.train(True)
            else:
                for module in self.trainable_base_modules:
                    module.train(True)
        return self

    def _components(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parent = self.base.shared(
            whole_contig_base_features(self.base, tokens, lengths)
        )
        motif = self.base.motif_shared(self.base.motif_encoder(tokens, lengths))
        parent_binary = self.base.binary_head(parent)
        short_rescue_raw = self.base.motif_short_rescue_correction(motif).squeeze(1)
        global_veto_raw = self.base.motif_global_veto_correction(motif).squeeze(1)
        short_rescue = (
            torch.relu(short_rescue_raw).detach()
            + short_rescue_raw
            - short_rescue_raw.detach()
        )
        global_veto = (
            torch.relu(global_veto_raw).detach()
            + global_veto_raw
            - global_veto_raw.detach()
        )
        short_gate = self.motif_length_gate(lengths, self.base.motif_config).to(
            short_rescue.dtype
        )
        weights = nn.functional.softplus(self.base.motif_fusion_raw_weights)
        correction = weights[0] * short_gate * short_rescue - weights[1] * global_veto
        binary = parent_binary + torch.stack(
            (-0.5 * correction, 0.5 * correction), dim=1
        )
        return parent, motif, binary, self.base.detail_head(parent)

    def _forward_outputs(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reverse = self.reverse_complement_batch(tokens, lengths, pad_token=5)
        forward_parent, forward_motif, forward_binary, forward_detail = (
            self._components(tokens, lengths)
        )
        reverse_parent, reverse_motif, reverse_binary, reverse_detail = (
            self._components(reverse, lengths)
        )
        features = torch.cat(
            (
                0.5 * (forward_parent + reverse_parent),
                0.5 * (forward_motif + reverse_motif),
            ),
            dim=1,
        ).to(torch.float16).float()
        released = 0.5 * (
            torch.softmax(forward_binary.double(), dim=1)[:, 1]
            + torch.softmax(reverse_binary.double(), dim=1)[:, 1]
        )
        forward_detail_probability = torch.softmax(forward_detail.double(), dim=1)
        reverse_detail_probability = torch.softmax(reverse_detail.double(), dim=1)
        unified = 0.5 * (
            forward_detail_probability[:, DETAIL_EUK_WITH_ORGANELLE].sum(dim=1)
            + reverse_detail_probability[:, DETAIL_EUK_WITH_ORGANELLE].sum(dim=1)
        )
        released_margin = torch.logit(released.clamp(1e-8, 1.0 - 1e-8)).float()
        unified_margin = torch.logit(unified.clamp(1e-8, 1.0 - 1e-8)).float()
        normalized_length = (
            torch.log(lengths.float()) / math.log(100_000.0)
        ).unsqueeze(1)
        combined = torch.cat(
            (
                features,
                released_margin.unsqueeze(1),
                unified_margin.unsqueeze(1),
                normalized_length,
            ),
            dim=1,
        )
        logits = []
        for index, head in enumerate(self.heads):
            mean = getattr(self, f"feature_mean_{index}")
            standard_deviation = getattr(self, f"feature_std_{index}")
            standardized = (combined - mean) / standard_deviation
            logits.append(head(standardized, unified_margin))
        return (
            ENSEMBLE_WEIGHTS[0] * logits[0] + ENSEMBLE_WEIGHTS[1] * logits[1],
            forward_detail_probability,
            reverse_detail_probability,
        )

    def forward_with_detail(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deployed, forward_detail, reverse_detail = self._forward_outputs(
            tokens, lengths
        )
        detail_log_probability = torch.log(
            (0.5 * (forward_detail + reverse_detail)).clamp_min(1e-12)
        ).float()
        return deployed, detail_log_probability

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self._forward_outputs(tokens, lengths)[0]


__all__ = [
    "DETAIL_LABEL",
    "ENSEMBLE_WEIGHTS",
    "TrainableDNAEnsemble",
    "load_frozen_head",
]

