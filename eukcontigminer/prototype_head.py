"""Compact reference-free density head over frozen sequence embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeDensityHead(nn.Module):
    """Score embeddings from small learned positive and negative prototypes.

    Prototypes are ordinary checkpoint tensors, not sequences or searchable
    reference records. Inference is a bounded matrix multiplication followed
    by either max or log-sum-exp aggregation.
    """

    def __init__(
        self,
        prototypes: torch.Tensor,
        prototype_positive: torch.Tensor,
        *,
        aggregation: str = "logsumexp",
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if prototypes.ndim != 2 or prototypes.shape[0] < 2 or prototypes.shape[1] < 1:
            raise ValueError("prototypes must be a matrix with at least two rows")
        if not torch.is_floating_point(prototypes) or not torch.isfinite(prototypes).all():
            raise ValueError("prototypes must be finite floating-point values")
        if prototype_positive.shape != (len(prototypes),):
            raise ValueError("prototype class mask shape differs")
        positive = prototype_positive.to(dtype=torch.bool, device="cpu")
        if not positive.any() or positive.all():
            raise ValueError("positive and negative prototypes are both required")
        if aggregation not in {"max", "logsumexp"}:
            raise ValueError("aggregation must be max or logsumexp")
        if not 0.0 < temperature <= 1.0:
            raise ValueError("temperature must be in (0, 1]")
        values = prototypes.detach().to(dtype=torch.float32, device="cpu").clone()
        norms = torch.linalg.vector_norm(values, dim=1)
        if torch.any(norms <= 1e-12):
            raise ValueError("prototype vectors must be nonzero")
        values = F.normalize(values, p=2, dim=1, eps=1e-12)
        self.feature_dimension = int(values.shape[1])
        self.aggregation = aggregation
        self.temperature = float(temperature)
        self.register_buffer("prototypes", values)
        self.register_buffer("prototype_positive", positive)

    def extra_repr(self) -> str:
        return (
            f"feature_dimension={self.feature_dimension}, "
            f"prototypes={len(self.prototypes)}, aggregation={self.aggregation!r}, "
            f"temperature={self.temperature}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dimension:
            raise ValueError("frozen feature matrix shape differs from prototype head")
        values = features.to(dtype=self.prototypes.dtype)
        values = F.normalize(values, p=2, dim=1, eps=1e-12)
        similarity = values @ self.prototypes.transpose(0, 1)
        positive = similarity[:, self.prototype_positive]
        negative = similarity[:, ~self.prototype_positive]
        if self.aggregation == "max":
            return positive.max(dim=1).values - negative.max(dim=1).values
        scale = self.temperature
        return scale * (
            torch.logsumexp(positive / scale, dim=1)
            - torch.logsumexp(negative / scale, dim=1)
        )


def weighted_prototype_means(
    features: torch.Tensor,
    assignments: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    prototype_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic weighted means for preassigned prototype groups."""

    if features.ndim != 2 or not torch.is_floating_point(features):
        raise ValueError("features must be a floating-point matrix")
    if assignments.shape != (len(features),) or sample_weight.shape != (len(features),):
        raise ValueError("prototype assignments or weights have invalid shape")
    if prototype_count < 1:
        raise ValueError("prototype_count must be positive")
    if assignments.dtype not in (torch.int32, torch.int64):
        raise ValueError("prototype assignments must be integer tensors")
    if (
        not torch.isfinite(features).all()
        or not torch.isfinite(sample_weight).all()
        or torch.any(sample_weight < 0)
        or torch.any(assignments < 0)
        or torch.any(assignments >= prototype_count)
    ):
        raise ValueError("prototype inputs contain invalid values")
    values = features.to(dtype=torch.float64, device="cpu")
    groups = assignments.to(dtype=torch.int64, device="cpu")
    weights = sample_weight.to(dtype=torch.float64, device="cpu")
    weighted_sum = torch.zeros(
        (prototype_count, features.shape[1]), dtype=torch.float64
    )
    mass = torch.zeros(prototype_count, dtype=torch.float64)
    weighted_sum.index_add_(0, groups, values * weights[:, None])
    mass.index_add_(0, groups, weights)
    if torch.any(mass <= 0):
        raise ValueError("every prototype must receive positive sample mass")
    means = weighted_sum / mass[:, None]
    return means.to(dtype=torch.float32), mass
