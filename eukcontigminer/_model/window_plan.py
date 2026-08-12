"""Vectorized independent-window plans for long-contig motif encoders.

The returned ranges are never concatenated before convolution.  A consumer
must encode every valid range independently and aggregate only the resulting
window embeddings or logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndependentWindowPlan:
    starts: torch.Tensor
    ends: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor
    full_sequence: torch.Tensor


def independent_contiguous_window_plan(
    lengths: torch.Tensor,
    *,
    maximum_total_bases: int = 4096,
    maximum_windows: int = 4,
) -> IndependentWindowPlan:
    """Plan full short sequences or independent full-span long windows.

    Contigs at or below ``maximum_total_bases`` receive one full-sequence
    range.  Longer contigs receive ``maximum_windows`` equal-width ranges,
    spread from the first to the last source base.  Since the long-contig
    sampled width is no greater than the budget, ranges cannot overlap.
    """

    if (
        lengths.ndim != 1
        or lengths.numel() == 0
        or lengths.dtype == torch.bool
        or not torch.is_floating_point(lengths)
        and lengths.dtype
        not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
    ):
        raise ValueError("lengths must be a non-empty numeric vector")
    if maximum_total_bases < 1 or maximum_windows < 2:
        raise ValueError("window budget and count are invalid")
    if maximum_total_bases < maximum_windows:
        raise ValueError("window budget must provide at least one base per window")
    if torch.is_floating_point(lengths):
        if not torch.all(lengths == torch.floor(lengths)):
            raise ValueError("lengths must contain integers")
    values = lengths.to(torch.int64)
    if torch.any(values <= 0):
        raise ValueError("contig lengths must be positive")

    device = values.device
    rows = values.shape[0]
    window_ids = torch.arange(maximum_windows, device=device, dtype=torch.int64)
    window_ids = window_ids.unsqueeze(0).expand(rows, -1)
    full = values <= int(maximum_total_bases)
    counts = torch.where(
        full,
        torch.ones_like(values),
        torch.full_like(values, int(maximum_windows)),
    )
    valid = window_ids < counts.unsqueeze(1)

    long_width = int(maximum_total_bases) // int(maximum_windows)
    span = (values - long_width).clamp_min(0).unsqueeze(1)
    denominator = int(maximum_windows) - 1
    # Integer rounding to nearest with exact first and last endpoints.
    long_starts = (window_ids * span + denominator // 2) // denominator
    starts = torch.where(full.unsqueeze(1), torch.zeros_like(long_starts), long_starts)
    widths = torch.where(
        full.unsqueeze(1),
        values.unsqueeze(1).expand_as(starts),
        torch.full_like(starts, long_width),
    )
    ends = starts + widths
    starts = torch.where(valid, starts, torch.zeros_like(starts))
    ends = torch.where(valid, ends, torch.zeros_like(ends))

    if torch.any(ends > values.unsqueeze(1)):
        raise RuntimeError("window plan extends beyond a contig")
    return IndependentWindowPlan(
        starts=starts,
        ends=ends,
        valid=valid,
        counts=counts,
        full_sequence=full,
    )
