"""Numerically stable direct DNA plus ESM-2 inference mathematics."""

from __future__ import annotations

import math

import numpy as np


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def standardized_probe_evidence(
    probe_logit: np.ndarray,
    *,
    center: float,
    scale: float,
) -> np.ndarray:
    """Standardize raw probe logits using frozen all-Train parameters."""

    values = np.asarray(probe_logit, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or not math.isfinite(center)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("invalid ESM-2 probe standardization inputs")
    evidence = (values - center) / scale
    if not np.all(np.isfinite(evidence)):
        raise FloatingPointError("standardized ESM-2 evidence is non-finite")
    return evidence


def fuse_dna_esm_probability(
    dna_probability: np.ndarray,
    standardized_probe: np.ndarray,
    *,
    positive_alpha: float,
    negative_alpha: float,
) -> np.ndarray:
    """Fuse one candidate-DNA score and one signed ESM evidence value per row."""

    dna = np.asarray(dna_probability, dtype=np.float64)
    probe = np.asarray(standardized_probe, dtype=np.float64)
    if (
        dna.ndim != 1
        or probe.shape != dna.shape
        or not len(dna)
        or not np.all(np.isfinite(dna))
        or np.any((dna < 0.0) | (dna > 1.0))
        or not np.all(np.isfinite(probe))
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (positive_alpha, negative_alpha)
        )
    ):
        raise ValueError("invalid DNA and ESM-2 fusion inputs")
    correction = positive_alpha * np.maximum(probe, 0.0) + (
        negative_alpha * np.minimum(probe, 0.0)
    )
    result = _sigmoid(_logit(dna) + correction)
    if not np.all(np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise FloatingPointError("joint DNA and ESM-2 probability is invalid")
    return result


def direct_joint_probability(
    dna_probability: np.ndarray,
    probe_logit: np.ndarray,
    *,
    probe_center: float,
    probe_scale: float,
    positive_alpha: float,
    negative_alpha: float,
) -> np.ndarray:
    """Compute final ``p_euk`` without scoring the obsolete reference DNA model."""

    evidence = standardized_probe_evidence(
        probe_logit, center=probe_center, scale=probe_scale
    )
    return fuse_dna_esm_probability(
        dna_probability,
        evidence,
        positive_alpha=positive_alpha,
        negative_alpha=negative_alpha,
    )


def dual_probe_piecewise_probability(
    reference_probability: np.ndarray,
    secondary_probe_logit: np.ndarray,
    lengths_bp: np.ndarray,
    *,
    secondary_probe_center: float,
    secondary_probe_scale: float,
    secondary_source_alpha: float,
    short_alpha: float,
    long_alpha: float,
    boundary_bp: int,
) -> np.ndarray:
    """Apply the frozen length-piecewise second-probe correction."""

    reference = np.asarray(reference_probability, dtype=np.float64)
    secondary = np.asarray(secondary_probe_logit, dtype=np.float64)
    lengths = np.asarray(lengths_bp)
    if (
        reference.ndim != 1
        or secondary.shape != reference.shape
        or lengths.shape != reference.shape
        or not len(reference)
        or not np.all(np.isfinite(reference))
        or np.any((reference < 0.0) | (reference > 1.0))
        or not np.all(np.isfinite(secondary))
        or np.any(lengths < 1)
        or not all(
            math.isfinite(value) and value > 0.0
            for value in (
                secondary_probe_scale,
                secondary_source_alpha,
                short_alpha,
                long_alpha,
            )
        )
        or not math.isfinite(secondary_probe_center)
        or boundary_bp < 1_000
    ):
        raise ValueError("invalid dual-probe piecewise fusion inputs")
    evidence = standardized_probe_evidence(
        secondary,
        center=secondary_probe_center,
        scale=secondary_probe_scale,
    )
    piecewise_alpha = np.where(lengths <= boundary_bp, short_alpha, long_alpha)
    result = _sigmoid(
        _logit(reference) + secondary_source_alpha * piecewise_alpha * evidence
    )
    if not np.all(np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise FloatingPointError("dual-probe piecewise probability is invalid")
    return result


__all__ = [
    "direct_joint_probability",
    "dual_probe_piecewise_probability",
    "fuse_dna_esm_probability",
    "standardized_probe_evidence",
]
