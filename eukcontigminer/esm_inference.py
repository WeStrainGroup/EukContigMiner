"""Direct whole-contig ORF featurization for a frozen ESM-2 encoder."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Sequence

import torch

from .esm_features import select_long_orfs_from_sequence


def _feature_only_transformer_layer_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor | None = None,
    self_attn_padding_mask: torch.Tensor | None = None,
    need_head_weights: bool = False,
):
    residual = x
    x = self.self_attn_layer_norm(x)
    x, attention = self.self_attn(
        query=x,
        key=x,
        value=x,
        key_padding_mask=self_attn_padding_mask,
        need_weights=need_head_weights,
        need_head_weights=need_head_weights,
        attn_mask=self_attn_mask,
    )
    x = residual + x
    residual = x
    x = self.final_layer_norm(x)
    from esm.modules import gelu

    x = gelu(self.fc1(x))
    x = self.fc2(x)
    return residual + x, attention


def optimize_esm2_feature_inference(model: torch.nn.Module) -> int:
    """Avoid materializing unused attention weights during feature inference."""

    layers = getattr(model, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or not layers:
        raise ValueError("unexpected ESM-2 transformer layer container")
    required = (
        "self_attn",
        "self_attn_layer_norm",
        "final_layer_norm",
        "fc1",
        "fc2",
    )
    if any(any(not hasattr(layer, name) for name in required) for layer in layers):
        raise ValueError("unexpected ESM-2 transformer layer architecture")
    optimized = 0
    for layer in layers:
        if getattr(layer, "_eukcontigminer_feature_inference", False):
            continue
        layer.forward = MethodType(_feature_only_transformer_layer_forward, layer)
        layer._eukcontigminer_feature_inference = True
        optimized += 1
    return optimized


@dataclass(frozen=True)
class ESM2ORFInferenceConfig:
    maximum_orfs: int = 2
    minimum_orf_length: int = 20
    maximum_orf_length: int = 1_000
    aggregation: str = "mean_max"
    token_budget: int = 16_384
    attention_budget: int = 2_000_000

    def validate(self) -> None:
        if (
            min(
                self.maximum_orfs,
                self.minimum_orf_length,
                self.maximum_orf_length,
                self.token_budget,
                self.attention_budget,
            )
            < 1
            or self.maximum_orf_length < self.minimum_orf_length
            or self.aggregation not in {"mean_max", "ordered"}
        ):
            raise ValueError("invalid ESM-2 ORF inference configuration")


def select_orfs_from_contig(
    sequence: str | bytes,
    config: ESM2ORFInferenceConfig = ESM2ORFInferenceConfig(),
) -> tuple[str, ...]:
    """Select the same RC-invariant ORFs used by frozen feature extraction."""

    config.validate()
    return select_long_orfs_from_sequence(
        sequence,
        maximum_orfs=config.maximum_orfs,
        minimum_length=config.minimum_orf_length,
        maximum_length=config.maximum_orf_length,
    )


def aggregate_orf_features(
    values: torch.Tensor, *, mode: str, maximum_orfs: int
) -> torch.Tensor:
    """Aggregate one contig exactly as the frozen ESM feature extractor."""

    if (
        values.ndim != 2
        or not values.is_floating_point()
        or not 1 <= len(values) <= maximum_orfs
        or maximum_orfs < 1
    ):
        raise ValueError("invalid per-ORF feature matrix")
    if mode == "mean_max":
        return torch.cat((values.mean(0), values.amax(0)))
    if mode == "ordered":
        if len(values) < maximum_orfs:
            values = torch.cat(
                (values, values[-1:].expand(maximum_orfs - len(values), -1))
            )
        return values.flatten()
    raise ValueError(f"unknown ORF aggregation mode: {mode}")


def _length_sorted_peptide_batches(
    peptides: Sequence[tuple[str, str]],
    *,
    token_budget: int,
    attention_budget: int,
) -> list[list[int]]:
    order = sorted(range(len(peptides)), key=lambda index: len(peptides[index][1]))
    batches: list[list[int]] = []
    start = 0
    while start < len(order):
        stop = start
        longest = 0
        total = 0
        while stop < len(order):
            length = len(peptides[order[stop]][1]) + 2
            candidate_longest = max(longest, length)
            candidate_count = stop - start + 1
            if stop > start and (
                total + length > token_budget
                or candidate_count * candidate_longest * candidate_longest
                > attention_budget
            ):
                break
            total += length
            longest = candidate_longest
            stop += 1
        batches.append(order[start:stop])
        start = stop
    return batches


def esm2_features_from_orfs(
    model: torch.nn.Module,
    batch_converter: Callable,
    orfs_by_contig: Sequence[tuple[str, ...]],
    *,
    representation_layer: int,
    device: torch.device,
    config: ESM2ORFInferenceConfig = ESM2ORFInferenceConfig(),
    quantize_float16: bool = True,
) -> torch.Tensor:
    """Return one frozen ESM feature row per contig.

    DNA length affects six-frame translation, but never the transformer input:
    each contig contributes at most ``maximum_orfs`` peptides, each bounded by
    ``maximum_orf_length``.  The optional float16 round trip reproduces the
    persisted feature shards used to train the downstream probe.
    """

    config.validate()
    if representation_layer < 1 or not orfs_by_contig:
        raise ValueError("invalid ESM-2 feature request")
    if any(
        not 1 <= len(orfs) <= config.maximum_orfs
        or any(
            not sequence or len(sequence) > config.maximum_orf_length
            for sequence in orfs
        )
        for orfs in orfs_by_contig
    ):
        raise ValueError("selected ORFs differ from the inference contract")

    flattened: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []
    for contig_index, orfs in enumerate(orfs_by_contig):
        start = len(flattened)
        flattened.extend(
            (f"{contig_index}:{rank}", sequence)
            for rank, sequence in enumerate(orfs)
        )
        slices.append((start, len(flattened)))

    peptide_features: list[torch.Tensor | None] = [None] * len(flattened)
    batches = _length_sorted_peptide_batches(
        flattened,
        token_budget=config.token_budget,
        attention_budget=config.attention_budget,
    )
    for selected_indices in batches:
        selected_rows = [flattened[index] for index in selected_indices]
        _labels, sequences, tokens = batch_converter(selected_rows)
        tokens = tokens.to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            representation = model(
                tokens,
                repr_layers=[representation_layer],
                return_contacts=False,
            )["representations"][representation_layer]
        pooled = torch.stack(
            [
                representation[row, 1 : len(sequence) + 1].float().mean(0)
                for row, sequence in enumerate(sequences)
            ]
        ).cpu()
        for row, original in enumerate(selected_indices):
            peptide_features[original] = pooled[row]
    if any(value is None for value in peptide_features):
        raise RuntimeError("ESM-2 peptide feature was not filled")
    stacked = torch.stack(
        [value for value in peptide_features if value is not None]
    )
    combined = torch.stack(
        [
            aggregate_orf_features(
                stacked[start:stop],
                mode=config.aggregation,
                maximum_orfs=config.maximum_orfs,
            )
            for start, stop in slices
        ]
    )
    if not torch.isfinite(combined).all():
        raise FloatingPointError("ESM-2 emitted non-finite contig features")
    if quantize_float16:
        combined = combined.to(torch.float16).float()
    return combined


__all__ = [
    "ESM2ORFInferenceConfig",
    "aggregate_orf_features",
    "esm2_features_from_orfs",
    "optimize_esm2_feature_inference",
    "select_orfs_from_contig",
]
