"""Direct whole-contig ORF featurization for a frozen ESM-2 encoder."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import re
from types import MethodType, SimpleNamespace
from typing import Callable, Sequence

import torch

from .esm_features import select_long_orfs_from_sequence


def _parse_cuda_architecture(name: str) -> tuple[int, int] | None:
    """Parse a PyTorch ``sm_XX`` or ``compute_XX`` architecture label."""

    matched = re.fullmatch(r"(?:sm|compute)_(\d+)", name)
    if matched is None:
        return None
    digits = matched.group(1)
    if len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def _cuda_build_supports_capability(
    capability: tuple[int, int], architectures: Sequence[str]
) -> bool:
    """Return whether compiled CUDA code can execute on the observed GPU.

    NVIDIA cubins are forward compatible within a compute-capability major
    generation, so for example ``sm_86`` covers an Ada ``sm_89`` device. PTX
    is forward compatible when its virtual capability is no newer than the
    device. A newer cubin does not cover an older device: ``sm_75`` therefore
    cannot be used as evidence for a V100 at ``sm_70``.
    """

    for name in architectures:
        parsed = _parse_cuda_architecture(str(name))
        if parsed is None:
            continue
        if str(name).startswith("sm_"):
            if parsed[0] == capability[0] and parsed[1] <= capability[1]:
                return True
        elif parsed <= capability:
            return True
    return False


def _esmc_hardware_policy(device: torch.device) -> dict[str, object]:
    """Resolve the supported CPU or NVIDIA server-GPU execution policy.

    Volta is the oldest explicitly supported CUDA generation because V100 is
    a required deployment target.  Volta and Turing use FP16; Ampere and newer
    use BF16 when the installed PyTorch build reports it as supported, with a
    safe FP16 fallback.  CPU inference remains architecture-neutral FP32.
    """

    if device.type == "cpu":
        return {
            "compute_dtype": None,
            "compute_capability": None,
            "compiled_cuda_architectures": [],
            "compiled_architecture_compatible": None,
            "compatibility_class": "cpu_fp32",
            "minimum_cuda_compute_capability": "7.0",
        }
    if device.type != "cuda":
        raise ValueError("ESM-C inference supports only CPU or CUDA devices")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if len(capability) != 2 or capability < (7, 0):
        observed = ".".join(str(value) for value in capability)
        raise RuntimeError(
            "ESM-C CUDA inference requires compute capability 7.0 or newer "
            f"(V100/Volta minimum); observed {observed}"
        )
    architectures = tuple(str(value) for value in torch.cuda.get_arch_list())
    if not architectures or not _cuda_build_supports_capability(
        capability, architectures
    ):
        observed = ".".join(str(value) for value in capability)
        compiled = ", ".join(architectures) if architectures else "none reported"
        raise RuntimeError(
            "the installed PyTorch CUDA build cannot execute on compute "
            f"capability {observed}; compiled architectures: {compiled}. "
            "Install a PyTorch build that includes this GPU generation "
            "(V100 requires sm_70)."
        )
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = (
        capability >= (8, 0)
        and callable(is_bf16_supported)
        and bool(is_bf16_supported())
    )
    if capability == (7, 0):
        compatibility_class = "volta_v100_fp16"
    elif capability < (8, 0):
        compatibility_class = "turing_fp16"
    elif use_bf16:
        compatibility_class = "ampere_or_newer_bf16"
    else:
        compatibility_class = "ampere_or_newer_fp16_fallback"
    return {
        "compute_dtype": torch.bfloat16 if use_bf16 else torch.float16,
        "compute_capability": capability,
        "compiled_cuda_architectures": list(architectures),
        "compiled_architecture_compatible": True,
        "compatibility_class": compatibility_class,
        "minimum_cuda_compute_capability": "7.0",
    }


def _esmc_autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Choose the dtype from the validated ESM-C hardware policy."""

    return _esmc_hardware_policy(device)["compute_dtype"]  # type: ignore[return-value]


def _feature_only_transformer_layer_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor | None = None,
    self_attn_padding_mask: torch.Tensor | None = None,
    need_head_weights: bool = False,
):
    """fair-ESM TransformerLayer forward without discarded attention maps."""

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
    # Use fair-ESM's exact GELU definition so the hidden representation remains
    # bit-identical to the released implementation.
    from esm.modules import gelu

    x = gelu(self.fc1(x))
    x = self.fc2(x)
    x = residual + x
    return x, attention


def _sdpa_feature_only_transformer_layer_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor | None = None,
    self_attn_padding_mask: torch.Tensor | None = None,
    need_head_weights: bool = False,
):
    """ESM-2 layer forward using fused scaled-dot-product attention.

    The deployment path never requests attention maps. Preserve the released
    fair-ESM path when maps or a general attention mask are requested, and use
    PyTorch SDPA only for frozen representation-only self-attention. Learned
    bias key/value and rotary embedding remain part of the exact model path.
    """

    if need_head_weights or self_attn_mask is not None:
        return _feature_only_transformer_layer_forward(
            self,
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_head_weights=need_head_weights,
        )

    residual = x
    x = self.self_attn_layer_norm(x)
    attention = self.self_attn
    target_length, batch_size, embedding_dimension = x.shape
    head_count = int(attention.num_heads)
    head_dimension = int(attention.head_dim)
    if embedding_dimension != head_count * head_dimension:
        raise RuntimeError("ESM-2 SDPA head dimensions differ")

    query = attention.q_proj(x)
    key = attention.k_proj(x)
    value = attention.v_proj(x)
    if attention.bias_k is not None:
        if attention.bias_v is None:
            raise RuntimeError("ESM-2 SDPA bias value is missing")
        key = torch.cat((key, attention.bias_k.repeat(1, batch_size, 1)))
        value = torch.cat((value, attention.bias_v.repeat(1, batch_size, 1)))
        if self_attn_padding_mask is not None:
            self_attn_padding_mask = torch.cat(
                (
                    self_attn_padding_mask,
                    self_attn_padding_mask.new_zeros((batch_size, 1)),
                ),
                dim=1,
            )

    source_length = key.shape[0]
    query = (
        query.contiguous()
        .view(target_length, batch_size, head_count, head_dimension)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * head_count, target_length, head_dimension)
    )
    key = (
        key.contiguous()
        .view(source_length, batch_size, head_count, head_dimension)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * head_count, source_length, head_dimension)
    )
    value = (
        value.contiguous()
        .view(source_length, batch_size, head_count, head_dimension)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * head_count, source_length, head_dimension)
    )
    if attention.rot_emb is not None:
        query, key = attention.rot_emb(query, key)
    query = query.view(batch_size, head_count, target_length, head_dimension)
    key = key.view(batch_size, head_count, source_length, head_dimension)
    value = value.view(batch_size, head_count, source_length, head_dimension)

    allowed = None
    if self_attn_padding_mask is not None:
        if self_attn_padding_mask.shape != (batch_size, source_length):
            raise RuntimeError("ESM-2 SDPA padding mask differs")
        # SDPA boolean masks use True for allowed positions, the inverse of
        # fair-ESM's key-padding convention.
        allowed = (~self_attn_padding_mask.to(torch.bool))[:, None, None, :]
    attended = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=allowed,
        dropout_p=0.0,
        is_causal=False,
    )
    attended = (
        attended.permute(2, 0, 1, 3)
        .contiguous()
        .view(target_length, batch_size, embedding_dimension)
    )
    x = residual + attention.out_proj(attended)

    residual = x
    x = self.final_layer_norm(x)
    from esm.modules import gelu

    x = gelu(self.fc1(x))
    x = self.fc2(x)
    return residual + x, None


def optimize_esm2_feature_inference(model: torch.nn.Module) -> int:
    """Avoid materializing unused attention weights in frozen ESM inference.

    fair-ESM 2.0.0 requests an attention matrix in every TransformerLayer even
    when ``need_head_weights`` is false and the model immediately discards it.
    The value path is unchanged; contact and explicit head-weight requests keep
    the released behavior.  Return the number of newly optimized layers.
    """

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


def optimize_esm2_sdpa_feature_inference(model: torch.nn.Module) -> int:
    """Opt an ESM-2 encoder into representation-only PyTorch SDPA inference.

    Fail closed unless every layer matches the released ESM-2 attention
    contract used by the hash-bound deployment model. This is explicit opt-in
    because fused kernels may introduce small floating-point differences that
    require end-to-end score and label parity validation.
    """

    layers = getattr(model, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or not layers:
        raise ValueError("unexpected ESM-2 transformer layer container")
    optimized = 0
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        required = (
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "num_heads",
            "head_dim",
            "bias_k",
            "bias_v",
            "rot_emb",
            "dropout",
            "add_zero_attn",
        )
        if attention is None or any(
            not hasattr(attention, name) for name in required
        ):
            raise ValueError("unexpected ESM-2 SDPA attention architecture")
        if (
            (attention.bias_k is None) != (attention.bias_v is None)
            or attention.rot_emb is None
            or float(attention.dropout) != 0.0
            or bool(attention.add_zero_attn)
        ):
            raise ValueError("unexpected ESM-2 SDPA attention contract")
        if getattr(layer, "_eukcontigminer_sdpa_feature_inference", False):
            continue
        layer.forward = MethodType(
            _sdpa_feature_only_transformer_layer_forward, layer
        )
        layer._eukcontigminer_feature_inference = True
        layer._eukcontigminer_sdpa_feature_inference = True
        optimized += 1
    return optimized


def _esmc_feature_only_forward(
    self,
    sequence_tokens: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
):
    """ESM-C forward that returns only the final residue embeddings.

    The official local ESM-C 3.2.1 forward always evaluates the unused
    sequence regression head and stacks every layer's hidden state.  Frozen
    EukContigMiner feature extraction consumes neither.  This path preserves
    the released embedding, transformer-block, final-norm, padding and optional
    Flash Attention operations while avoiding those two allocations.
    """

    if not isinstance(sequence_tokens, torch.Tensor) or sequence_tokens.ndim != 2:
        raise ValueError("ESM-C feature tokens must be a rank-two tensor")
    pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("ESM-C tokenizer has no padding token")
    if sequence_id is None:
        sequence_id = sequence_tokens != int(pad_token_id)

    x = self.embed(sequence_tokens)
    batch_size, sequence_length = x.shape[:2]
    flash_attention = bool(self._use_flash_attn)
    indices = None
    if flash_attention:
        if sequence_id.dtype != torch.bool or sequence_id.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("ESM-C Flash Attention requires a boolean token mask")
        from esm.models.esmc import unpad_input

        if unpad_input is None:
            raise RuntimeError("ESM-C Flash Attention unpadding is unavailable")
        x, indices, *_ = unpad_input(x, sequence_id)

    transformer = self.transformer
    representation_layer = getattr(
        self, "_eukcontigminer_representation_layer", None
    )
    target_layer = (
        len(transformer.blocks)
        if representation_layer is None
        else int(representation_layer)
    )
    chain_id = torch.ones(
        size=x.shape[:-1], dtype=torch.int64, device=x.device
    )
    for layer, block in enumerate(transformer.blocks, start=1):
        x = block(x, sequence_id, None, None, chain_id)
        if layer == target_layer:
            break
    if representation_layer is None:
        x = transformer.norm(x)

    if flash_attention:
        from esm.models.esmc import pad_input

        if pad_input is None or indices is None:
            raise RuntimeError("ESM-C Flash Attention padding is unavailable")
        x = pad_input(x, indices, batch_size, sequence_length)
    return SimpleNamespace(embeddings=x)


def optimize_esmc_feature_inference(
    model: torch.nn.Module, *, representation_layer: int | None = None
) -> int:
    """Opt an official local ESM-C encoder into embedding-only inference.

    ``representation_layer=None`` preserves the released final normalized
    embedding.  A one-based layer returns the released raw hidden state after
    that transformer block and exits before later blocks.  Fail closed unless
    the model matches the ESM-C 3.2.1 architecture used by the preregistered
    300M/600M canary.  Return one when newly optimized and zero when the same
    model was already optimized for the same representation.
    """

    required = ("embed", "transformer", "sequence_head", "tokenizer")
    if any(not hasattr(model, name) for name in required) or not hasattr(
        model, "_use_flash_attn"
    ):
        raise ValueError("unexpected ESM-C model architecture")
    transformer = model.transformer
    if (
        not isinstance(getattr(transformer, "blocks", None), torch.nn.ModuleList)
        or not transformer.blocks
        or not isinstance(getattr(transformer, "norm", None), torch.nn.Module)
        or getattr(model.tokenizer, "pad_token_id", None) is None
    ):
        raise ValueError("unexpected ESM-C transformer architecture")
    if representation_layer is not None and (
        not isinstance(representation_layer, int)
        or isinstance(representation_layer, bool)
        or not 1 <= representation_layer <= len(transformer.blocks)
    ):
        raise ValueError("ESM-C representation layer is out of range")
    if getattr(model, "_eukcontigminer_feature_inference", False):
        if (
            getattr(model, "_eukcontigminer_representation_layer", None)
            != representation_layer
        ):
            raise ValueError(
                "ESM-C model is already optimized for a different representation"
            )
        return 0
    model.forward = MethodType(_esmc_feature_only_forward, model)
    model._eukcontigminer_feature_inference = True
    model._eukcontigminer_representation_layer = representation_layer
    return 1


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
        # Preserve the frozen per-peptide float32 mean exactly, but queue every
        # reduction in the batch before synchronizing the GPU once.  Copying
        # each row separately caused one small device-to-host transfer (and a
        # synchronization) per ORF, which is disproportionately expensive for
        # the two-ORF deployment path.
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


def esmc_features_from_orfs(
    model: torch.nn.Module,
    tokenize: Callable[[list[str]], torch.Tensor],
    orfs_by_contig: Sequence[tuple[str, ...]],
    *,
    device: torch.device,
    config: ESM2ORFInferenceConfig = ESM2ORFInferenceConfig(),
    representation_layer: int | None = None,
    quantize_float16: bool = True,
) -> torch.Tensor:
    """Return final or one selected hidden-layer ESM-C ORF features.

    ``tokenize`` is the official local ESM-C batch tokenizer
    (``model._tokenize`` in esm 3.2.1).  Keeping it injected makes this module
    importable and testable without installing the isolated ESM-C environment.
    Hidden layers use the official one-based block order and raw pre-final-norm
    state; the default remains the released final normalized embedding.
    """

    config.validate()
    if representation_layer is not None and (
        not isinstance(representation_layer, int)
        or isinstance(representation_layer, bool)
        or representation_layer < 1
    ):
        raise ValueError("ESM-C representation layer must be a positive integer")
    if not orfs_by_contig or any(
        not 1 <= len(orfs) <= config.maximum_orfs
        or any(
            not sequence or len(sequence) > config.maximum_orf_length
            for sequence in orfs
        )
        for orfs in orfs_by_contig
    ):
        raise ValueError("selected ORFs differ from the ESM-C inference contract")

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
    for selected_indices in _length_sorted_peptide_batches(
        flattened,
        token_budget=config.token_budget,
        attention_budget=config.attention_budget,
    ):
        sequences = [flattened[index][1] for index in selected_indices]
        tokens = tokenize(sequences)
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 2
            or tokens.shape[0] != len(sequences)
            or any(tokens.shape[1] < len(sequence) + 2 for sequence in sequences)
        ):
            raise ValueError("ESM-C tokenizer output differs from the batch contract")
        tokens = tokens.to(device, non_blocking=device.type == "cuda")
        autocast_dtype = _esmc_autocast_dtype(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = model(sequence_tokens=tokens)
            if representation_layer is None:
                representation = getattr(output, "embeddings", None)
            else:
                hidden_states = getattr(output, "hidden_states", None)
                if (
                    not isinstance(hidden_states, torch.Tensor)
                    or hidden_states.ndim != 4
                    or representation_layer > hidden_states.shape[0]
                ):
                    raise ValueError(
                        "ESM-C output lacks the requested hidden representation"
                    )
                representation = hidden_states[representation_layer - 1]
        if (
            not isinstance(representation, torch.Tensor)
            or representation.ndim != 3
            or representation.shape[:2] != tokens.shape
        ):
            raise ValueError("ESM-C output differs from the embedding contract")
        pooled = torch.stack(
            [
                representation[row, 1 : len(sequence) + 1].float().mean(0)
                for row, sequence in enumerate(sequences)
            ]
        ).cpu()
        for row, original in enumerate(selected_indices):
            peptide_features[original] = pooled[row]
    if any(value is None for value in peptide_features):
        raise RuntimeError("ESM-C peptide feature was not filled")
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
        raise FloatingPointError("ESM-C emitted non-finite contig features")
    if quantize_float16:
        combined = combined.to(torch.float16).float()
    return combined


__all__ = [
    "ESM2ORFInferenceConfig",
    "aggregate_orf_features",
    "esm2_features_from_orfs",
    "esmc_features_from_orfs",
    "optimize_esm2_feature_inference",
    "optimize_esm2_sdpa_feature_inference",
    "optimize_esmc_feature_inference",
    "select_orfs_from_contig",
]
