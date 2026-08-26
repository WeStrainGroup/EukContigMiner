"""Scientific constants and prediction semantics that models must obey."""

from __future__ import annotations

import math

BENCHMARK_LENGTHS_BP: tuple[int, ...] = (
    *range(1_000, 10_001, 500),
    50_000,
    100_000,
)
REPORT_PREVALENCES: tuple[float, ...] = (0.001, 0.01, 0.10)
DEPLOYMENT_FULL_F1_TARGET_AT_1PCT = 0.99
MINIMUM_ITERATION_F1_DELTA_AT_1PCT = 0.0005
THRESHOLDS: tuple[float, ...] = tuple(i / 100 for i in range(1, 100))
FORMAL_MIN_LENGTH_BP = 1_000
DATA_CUTOFF_EXCLUSIVE = "2026-01-01"
IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")
EUK_SOURCE_LABELS = frozenset(
    {"Fungi", "Metazoa", "Viridiplantae", "Other_Eukaryota", "Organelle"}
)


def normalize_sequence(sequence: str) -> str:
    """Return uppercase IUPAC DNA or raise for an illegal contig.

    FASTA line wrapping is handled by the FASTA reader. Embedded whitespace,
    gaps, and an empty sequence are rejected so every accepted record has an
    unambiguous length and prediction row.
    """

    normalized = sequence.upper()
    if not normalized:
        raise ValueError("contig sequence is empty")
    illegal = set(normalized) - IUPAC_DNA
    if illegal:
        symbols = "".join(sorted(illegal))
        raise ValueError(f"illegal contig symbols: {symbols!r}")
    return normalized


def validate_probability(value: float, *, name: str = "p_euk") -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def classify_score(p_euk: float, threshold: float) -> str:
    """Apply the production rule: equality belongs to Other."""

    score = validate_probability(p_euk)
    cutoff = validate_probability(threshold, name="threshold")
    return "Eukaryota" if score > cutoff else "Other"


def binary_truth_from_source_label(label: str) -> int:
    """Map source taxonomy to the locked binary target.

    Organelle is intentionally positive: mitochondria and plastids belong to
    eukaryotic organisms for this screening task despite their prokaryotic
    evolutionary origins. It is not an independent output class.
    """

    if label in EUK_SOURCE_LABELS:
        return 1
    if label in {"Bacteria", "Archaea", "Other"}:
        return 0
    raise ValueError(f"unknown source label: {label}")
