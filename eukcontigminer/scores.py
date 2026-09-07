"""Adapters into one complete, model-independent score-table contract."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import validate_probability

STANDARD_HEADER = ("contig_id", "p_euk")
TIARA_REQUIRED_COLUMNS = frozenset(
    {
        "sequence_id",
        "class_fst_stage",
        "class_snd_stage",
        "org",
        "bac",
        "arc",
        "euk",
        "unk1",
        "pla",
        "unk2",
        "mit",
    }
)
EUKCONTIGMINER_REQUIRED_COLUMNS = (
    "contig",
    "length",
    "p_euk",
    "label",
)


@dataclass(frozen=True, slots=True)
class ScoreRow:
    contig_id: str
    p_euk: float


def fuse_probabilities(
    left: float,
    right: float,
    *,
    left_weight: float,
    space: str = "probability",
) -> float:
    """Fuse two probabilities in probability or log-odds coordinates."""

    left_value = validate_probability(left, name="left probability")
    right_value = validate_probability(right, name="right probability")
    weight = float(left_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("left weight must be finite and in [0, 1]")
    if space == "probability":
        return weight * left_value + (1.0 - weight) * right_value
    if space != "logit":
        raise ValueError("fusion space must be probability or logit")
    if weight == 1.0:
        return left_value
    if weight == 0.0:
        return right_value

    epsilon = 1e-15
    left_clipped = min(max(left_value, epsilon), 1.0 - epsilon)
    right_clipped = min(max(right_value, epsilon), 1.0 - epsilon)
    fused_logit = weight * (
        math.log(left_clipped) - math.log1p(-left_clipped)
    ) + (1.0 - weight) * (
        math.log(right_clipped) - math.log1p(-right_clipped)
    )
    if fused_logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-fused_logit))
    exponential = math.exp(fused_logit)
    return exponential / (1.0 + exponential)


def shift_log_odds_probability(
    probability: float, *, center: float = 0.99, temperature: float = 1.0
) -> float:
    """Spread saturated scores without changing their ordering.

    ``center`` maps to 0.5. Exact zero and one remain exact boundaries.
    This is a deterministic score-coordinate transform, not fitted calibration.
    """

    value = validate_probability(probability)
    center = validate_probability(center, name="log-odds center")
    if center in (0.0, 1.0):
        raise ValueError("log-odds center must be strictly between zero and one")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if value in (0.0, 1.0):
        return value
    shifted = (
        math.log(value)
        - math.log1p(-value)
        - math.log(center)
        + math.log1p(-center)
    ) / temperature
    if shifted >= 0.0:
        return 1.0 / (1.0 + math.exp(-shifted))
    exponential = math.exp(shifted)
    return exponential / (1.0 + exponential)


def _validate_unique(rows: Iterable[ScoreRow]) -> list[ScoreRow]:
    result = list(rows)
    if not result:
        raise ValueError("score table is empty")
    seen: set[str] = set()
    for row in result:
        if not row.contig_id:
            raise ValueError("score table contains an empty contig_id")
        if row.contig_id in seen:
            raise ValueError(f"duplicate score for contig_id: {row.contig_id}")
        seen.add(row.contig_id)
        validate_probability(row.p_euk)
    return result


def read_standard_scores(path: str | Path) -> list[ScoreRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(STANDARD_HEADER):
            raise ValueError("standard score header must be: contig_id\\tp_euk")
        rows = [
            ScoreRow(row["contig_id"], validate_probability(float(row["p_euk"])))
            for row in reader
        ]
    return _validate_unique(rows)


def _read_tiara_scores(
    path: str | Path, *, nonfinite_fallback: float | None
) -> tuple[list[ScoreRow], tuple[str, ...]]:
    """Aggregate first-stage Euk+org scores and expose any explicit fallback."""

    if nonfinite_fallback is not None:
        nonfinite_fallback = validate_probability(
            nonfinite_fallback, name="Tiara non-finite fallback"
        )

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(TIARA_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "Tiara output must be generated with --probabilities; "
                f"missing columns: {missing}"
            )
        rows = []
        fallback_ids: list[str] = []
        for row in reader:
            euk_raw = float(row["euk"])
            organelle_raw = float(row["org"])
            if not math.isfinite(euk_raw) or not math.isfinite(organelle_raw):
                if nonfinite_fallback is None:
                    validate_probability(euk_raw, name="Tiara euk")
                    validate_probability(organelle_raw, name="Tiara org")
                fallback_ids.append(row["sequence_id"])
                rows.append(ScoreRow(row["sequence_id"], nonfinite_fallback))
                continue
            euk = validate_probability(euk_raw, name="Tiara euk")
            organelle = validate_probability(organelle_raw, name="Tiara org")
            combined = euk + organelle
            # Tiara prints components independently with six decimal places.
            # Their rounded sum can be 1.000001 although the model distribution
            # is bounded by one. Larger excesses and non-finite values fail.
            if 1.0 < combined <= 1.0 + 1.1e-6:
                combined = 1.0
            rows.append(
                ScoreRow(row["sequence_id"], validate_probability(combined))
            )
    return _validate_unique(rows), tuple(fallback_ids)


def read_tiara_scores(path: str | Path) -> list[ScoreRow]:
    """Read Tiara strictly; reject every non-finite component probability."""

    rows, fallback_ids = _read_tiara_scores(path, nonfinite_fallback=None)
    if fallback_ids:
        raise AssertionError("strict Tiara parser unexpectedly used a fallback")
    return rows


def read_tiara_scores_with_fallback(
    path: str | Path, *, nonfinite_fallback: float
) -> tuple[list[ScoreRow], tuple[str, ...]]:
    """Read Tiara with an explicit score for rows where it emitted NaN/Inf."""

    return _read_tiara_scores(path, nonfinite_fallback=nonfinite_fallback)


def read_eukcontigminer_scores(path: str | Path) -> list[ScoreRow]:
    """Read the public previous EukContigMiner CLI output."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(EUKCONTIGMINER_REQUIRED_COLUMNS):
            raise ValueError(
                "EukContigMiner score header must be: contig\\tlength\\tp_euk\\tlabel"
            )
        rows = [
            ScoreRow(
                row["contig"],
                validate_probability(float(row["p_euk"])),
            )
            for row in reader
        ]
    return _validate_unique(rows)


def align_scores(expected_contig_ids: Sequence[str], rows: Iterable[ScoreRow]) -> list[float]:
    """Return scores in frozen panel order or fail closed on any ID difference."""

    if not expected_contig_ids:
        raise ValueError("expected contig IDs are empty")
    if len(set(expected_contig_ids)) != len(expected_contig_ids):
        raise ValueError("expected contig IDs are not unique")
    score_by_id = {row.contig_id: row.p_euk for row in _validate_unique(rows)}
    expected = set(expected_contig_ids)
    observed = set(score_by_id)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"score IDs differ from frozen panel; missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    return [score_by_id[contig_id] for contig_id in expected_contig_ids]


def write_standard_scores(path: str | Path, rows: Iterable[ScoreRow]) -> None:
    rows = _validate_unique(rows)
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(STANDARD_HEADER)
        for row in rows:
            writer.writerow((row.contig_id, f"{row.p_euk:.17g}"))
    temporary.replace(destination)
