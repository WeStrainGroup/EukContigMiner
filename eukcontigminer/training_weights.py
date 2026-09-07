"""Training-only weighting utilities kept independent of the model runtime."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def binary_class_mass_factors(truth: np.ndarray, negative_mass: float) -> np.ndarray:
    """Return mean-one factors assigning the requested mass to each class."""
    labels = np.asarray(truth)
    if labels.ndim != 1 or len(labels) == 0 or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("truth must be a non-empty one-dimensional binary vector")
    if not math.isfinite(negative_mass) or not 0.0 < negative_mass < 1.0:
        raise ValueError("negative class mass must be finite and strictly between zero and one")
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("truth must contain both binary classes")
    desired = np.asarray([negative_mass, 1.0 - negative_mass], dtype=np.float64)
    factors = desired[labels.astype(np.int64)] * len(labels) / counts[labels.astype(np.int64)]
    if not math.isclose(float(factors.mean()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("class-mass factors are not normalized")
    return factors


def group_balance_factors(
    truth: np.ndarray,
    bands: np.ndarray,
    group_ids: np.ndarray,
    power: float,
) -> np.ndarray:
    """Return inverse-frequency factors normalized inside truth-length strata."""
    if truth.ndim != 1 or bands.shape != truth.shape or group_ids.shape != truth.shape:
        raise ValueError("group balancing arrays must be aligned one-dimensional vectors")
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("group balance power must be finite and non-negative")
    factors = np.ones(len(truth), dtype=np.float64)
    if power == 0.0:
        return factors
    for label in (0, 1):
        for band in range(3):
            selected = (truth == label) & (bands == band)
            if not np.any(selected):
                continue
            counts = Counter(group_ids[selected])
            raw = np.asarray(
                [counts[group_id] ** (-power) for group_id in group_ids[selected]],
                dtype=np.float64,
            )
            raw /= raw.mean()
            factors[selected] = raw
    return factors


def length_bounded_batches(
    indices: np.ndarray,
    lengths: np.ndarray,
    *,
    max_records: int,
    max_padded_bases: int,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Bucket rows by length and enforce a padded-base bound per batch."""
    rows = np.asarray(indices, dtype=np.int64)
    observed_lengths = np.asarray(lengths)
    if rows.ndim != 1 or observed_lengths.ndim != 1:
        raise ValueError("indices and lengths must be one-dimensional")
    if max_records < 1 or max_padded_bases < 1:
        raise ValueError("batch limits must be positive")
    if len(rows) and (
        rows.min() < 0
        or rows.max() >= len(observed_lengths)
        or np.any(observed_lengths[rows] < 1)
        or np.any(observed_lengths[rows] > max_padded_bases)
    ):
        raise ValueError("batch row or length exceeds its bound")
    ordered = rows[np.argsort(observed_lengths[rows], kind="stable")]
    batches: list[np.ndarray] = []
    start = 0
    while start < len(ordered):
        longest = int(observed_lengths[ordered[start]])
        stop = start + 1
        while stop < len(ordered) and stop - start < max_records:
            candidate_longest = int(observed_lengths[ordered[stop]])
            if (stop - start + 1) * candidate_longest > max_padded_bases:
                break
            longest = candidate_longest
            stop += 1
        if (stop - start) * longest > max_padded_bases:
            raise AssertionError("constructed batch exceeds padded-base bound")
        batches.append(ordered[start:stop].copy())
        start = stop
    if seed is not None:
        np.random.default_rng(seed).shuffle(batches)
    return batches
