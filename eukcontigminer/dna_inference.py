"""Whole-contig DNA backbone inference beyond the training length ceiling.

The public DNA backbone uses ``config.maximum_length`` both as an input guard
and as the fixed denominator of a learned log-length feature.  Raising that
configuration value would therefore change predictions inside the validated
1--100 kb range.  This module removes only the input guard: records within the
public ceiling still use the released method, while longer records reproduce
the same feature computation with the original length-feature denominator.
"""

from __future__ import annotations

from math import log

import torch


def _valid_mask(
    lengths: torch.Tensor, width: int, *, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(1).to(dtype=dtype)


def _unbounded_global_features(
    base: torch.nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the released global features without its width rejection."""

    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError("tokens and lengths have incompatible shapes")
    if torch.any(lengths <= 0) or torch.any(lengths > tokens.shape[1]):
        raise ValueError("sequence lengths are invalid")

    config = base.config
    reference_length = int(config.maximum_length)
    if reference_length <= 1:
        raise ValueError("DNA length-feature reference must exceed one base")

    input_mask = _valid_mask(lengths, tokens.shape[1], dtype=torch.bool)
    masked_tokens = torch.where(
        input_mask.squeeze(1),
        tokens,
        torch.full_like(tokens, config.pad_token),
    )
    embedded = base.embedding(masked_tokens).transpose(1, 2)
    hidden = torch.cat([stem(embedded) for stem in base.stems], dim=1)
    hidden = base.stem_projection(hidden)
    reduced_lengths = base.downsampled_lengths(lengths)
    mask = _valid_mask(reduced_lengths, hidden.shape[2], dtype=hidden.dtype)
    hidden = torch.nn.functional.gelu(
        base.stem_normalization(hidden, mask)
    ) * mask
    for block in base.blocks:
        hidden = block(hidden, mask)
    hidden = base.final_normalization(hidden, mask) * mask

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
    kmer = base._canonical_kmer_pool(masked_tokens, lengths).to(hidden.dtype)
    composition = base._base_composition(masked_tokens, lengths).to(hidden.dtype)

    # ``_base_composition`` uses config.maximum_length.  Keeping the released
    # configuration unchanged is what preserves this learned feature exactly;
    # make the invariant explicit rather than resizing the model configuration.
    expected_length = (
        torch.log(lengths.to(torch.float32)) / log(float(reference_length))
    ).to(composition.dtype)
    if not torch.equal(composition[:, -1], expected_length):
        raise RuntimeError("DNA backbone length-feature definition differs")
    return torch.cat(
        (mean, standard_deviation, maximum, log_mean_exp, kmer, composition),
        dim=1,
    )


def _unbounded_parent_features(
    base: torch.nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    global_features = _unbounded_global_features(base, tokens, lengths)
    short_features = getattr(base, "_short_features", None)
    if not callable(short_features):
        return global_features
    local_features, gate = short_features(tokens, lengths)
    return torch.cat((global_features, local_features, gate), dim=1)


def whole_contig_base_features(
    base: torch.nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return backbone features for any positive whole-contig length.

    Rows at or below the released ceiling are deliberately evaluated by the
    original method.  In a mixed batch they are separated from longer rows, so
    the validated path is never silently replaced by the reimplemented path.
    """

    if tokens.ndim != 2 or lengths.shape != (len(tokens),):
        raise ValueError("tokens and lengths have incompatible shapes")
    if torch.any(lengths <= 0) or torch.any(lengths > tokens.shape[1]):
        raise ValueError("sequence lengths are invalid")
    config = getattr(base, "config", None)
    if config is None or not hasattr(config, "maximum_length"):
        # Lightweight test doubles and unrelated backbones have no released
        # length contract; retain their native forward unchanged.
        return base.forward_features(tokens, lengths)
    maximum_length = int(config.maximum_length)
    within = lengths <= maximum_length
    if bool(torch.all(within)):
        width = int(lengths.max().item())
        return base.forward_features(tokens[:, :width], lengths)
    if bool(torch.all(~within)):
        width = int(lengths.max().item())
        return _unbounded_parent_features(base, tokens[:, :width], lengths)

    result: torch.Tensor | None = None
    for selected, unbounded in ((within, False), (~within, True)):
        rows = torch.nonzero(selected, as_tuple=False).flatten()
        selected_lengths = lengths.index_select(0, rows)
        width = int(selected_lengths.max().item())
        selected_tokens = tokens.index_select(0, rows)[:, :width]
        features = (
            _unbounded_parent_features(base, selected_tokens, selected_lengths)
            if unbounded
            else base.forward_features(selected_tokens, selected_lengths)
        )
        if result is None:
            result = features.new_empty((len(tokens), features.shape[1]))
        elif features.shape[1] != result.shape[1]:
            raise RuntimeError("bounded and unbounded DNA feature widths differ")
        result = result.index_copy(0, rows, features)
    if result is None:
        raise RuntimeError("DNA feature routing produced no rows")
    return result


__all__ = ["whole_contig_base_features"]
