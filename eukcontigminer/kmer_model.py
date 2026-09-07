"""Reverse-complement-invariant canonical k-mer encoder used by the frozen k10 gate."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


AUXILIARY_DIMENSION = 7


def canonical_vocabulary_size(kmer_length: int) -> int:
    if type(kmer_length) is not int or not 1 <= kmer_length <= 15:
        raise ValueError("kmer_length must be an integer in [1, 15]")
    return 4**kmer_length + 1


def sampled_kmer_count(length_bp: int, *, kmer_length: int, maximum_tokens: int) -> int:
    if type(length_bp) is not int or length_bp < kmer_length:
        raise ValueError("sequence is shorter than k-mer length")
    if type(maximum_tokens) is not int or maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    available = length_bp - kmer_length + 1
    if available > maximum_tokens and maximum_tokens % 2:
        raise ValueError("a capped maximum_tokens value must be even for RC invariance")
    return min(available, maximum_tokens)


def canonical_kmer_tokens(
    sequence: str | bytes, *, kmer_length: int, maximum_tokens: int
) -> tuple[int, ...]:
    raw = sequence.encode("ascii") if isinstance(sequence, str) else bytes(sequence)
    count = sampled_kmer_count(len(raw), kmer_length=kmer_length, maximum_tokens=maximum_tokens)
    available = len(raw) - kmer_length + 1
    if count == available:
        positions = tuple(range(available))
    else:
        maximum_position = available - 1
        left = tuple(index * maximum_position // (count - 1) for index in range(count // 2))
        positions = left + tuple(maximum_position - value for value in reversed(left))
    nucleotide = {65: 0, 67: 1, 71: 2, 84: 3}
    unknown = 4**kmer_length
    result = []
    upper = raw.upper()
    for position in positions:
        forward = 0
        reverse = 0
        valid = True
        for offset in range(kmer_length):
            value = nucleotide.get(upper[position + offset])
            if value is None:
                valid = False
                break
            forward = (forward << 2) | value
            reverse |= (3 - value) << (2 * offset)
        result.append(min(forward, reverse) if valid else unknown)
    return tuple(result)


class CanonicalKmerEncoder(nn.Module):
    """Mean-pool learned canonical k-mers, then combine sequence summaries."""

    def __init__(
        self,
        *,
        kmer_length: int = 10,
        embedding_dimension: int = 32,
        hidden_dimension: int = 96,
        dropout: float = 0.15,
        sparse_embeddings: bool = True,
    ) -> None:
        super().__init__()
        vocabulary_size = canonical_vocabulary_size(kmer_length)
        if embedding_dimension < 1 or hidden_dimension < 1:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.kmer_length = kmer_length
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.dropout = dropout
        self.sparse_embeddings = bool(sparse_embeddings)
        self.embedding = nn.EmbeddingBag(
            vocabulary_size,
            embedding_dimension,
            mode="mean",
            include_last_offset=True,
            sparse=self.sparse_embeddings,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(embedding_dimension + AUXILIARY_DIMENSION),
            nn.Linear(embedding_dimension + AUXILIARY_DIMENSION, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, tokens: Tensor, offsets: Tensor, auxiliary: Tensor) -> Tensor:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a one-dimensional torch.long tensor")
        if offsets.ndim != 1 or offsets.dtype != torch.long or len(offsets) < 2:
            raise ValueError("offsets must include one start and the final offset")
        if int(offsets[0]) != 0 or int(offsets[-1]) != len(tokens):
            raise ValueError("offsets do not span the token tensor")
        if torch.any(offsets[1:] <= offsets[:-1]):
            raise ValueError("every contig must contain at least one k-mer token")
        batch = len(offsets) - 1
        if auxiliary.shape != (batch, AUXILIARY_DIMENSION):
            raise ValueError("auxiliary feature matrix has the wrong shape")
        if not auxiliary.is_floating_point():
            raise ValueError("auxiliary features must be floating point")
        vocabulary_size = self.embedding.num_embeddings
        if torch.any(tokens < 0) or torch.any(tokens >= vocabulary_size):
            raise ValueError("canonical k-mer token lies outside the vocabulary")
        embedded = self.embedding(tokens, offsets)
        values = self.projection(torch.cat((embedded, auxiliary), dim=1)).squeeze(1)
        if not torch.isfinite(values).all():
            raise FloatingPointError("non-finite canonical k-mer logit")
        return values


def normalized_log_length(lengths: Tensor) -> Tensor:
    if torch.any(lengths < 1):
        raise ValueError("lengths must be positive")
    return torch.log(lengths.to(torch.float32)) / math.log(100_000.0)
