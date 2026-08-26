"""Deterministic RC-invariant ORF selection for frozen protein embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .protein_model import (
    AA_ORDER,
    STOP_ID,
    UNKNOWN_ID,
    _CODON_LOOKUP,
    _NT_COMPLEMENT,
    _NT_LOOKUP,
)


_TOKEN_TO_AA = np.asarray(["?"] + list(AA_ORDER) + ["*", "X"])


def _segments(frame: np.ndarray) -> list[np.ndarray]:
    if frame.ndim != 1 or frame.dtype != np.uint8:
        raise ValueError("protein frame must be a one-dimensional uint8 array")
    boundaries = np.flatnonzero(frame == STOP_ID)
    result: list[np.ndarray] = []
    start = 0
    for stop in boundaries:
        if int(stop) > start:
            result.append(frame[start : int(stop)])
        start = int(stop) + 1
    if start < len(frame):
        result.append(frame[start:])
    return result


def select_long_orfs(
    frames: Sequence[np.ndarray],
    *,
    maximum_orfs: int = 2,
    minimum_length: int = 20,
    maximum_length: int = 1_000,
) -> tuple[str, ...]:
    """Select a deterministic top-ORF set from all six complete translations.

    The six translated frames form the same multiset after reverse
    complementation. Ranking only by length and amino-acid content therefore
    makes this selection independent of strand and frame order.
    """

    if len(frames) != 6:
        raise ValueError("exactly six translated frames are required")
    if maximum_orfs < 1 or minimum_length < 1 or maximum_length < minimum_length:
        raise ValueError("invalid ORF selection bounds")
    candidates: list[tuple[int, str]] = []
    fallback: list[tuple[int, str]] = []
    for frame in frames:
        for segment in _segments(frame):
            if not len(segment):
                continue
            if np.any((segment < 1) | (segment > UNKNOWN_ID)):
                raise ValueError("protein token lies outside the translated vocabulary")
            sequence = "".join(_TOKEN_TO_AA[segment].tolist())
            row = (-len(sequence), sequence)
            fallback.append(row)
            if len(sequence) >= minimum_length:
                candidates.append(row)
    if not candidates:
        candidates = fallback
    if not candidates:
        raise ValueError("six-frame translation contains no amino-acid token")
    unique = sorted(set(candidates))
    selected = []
    for _negative_length, sequence in unique[:maximum_orfs]:
        if len(sequence) > maximum_length:
            left = maximum_length // 2
            right = maximum_length - left
            sequence = sequence[:left] + sequence[-right:]
        selected.append(sequence)
    return tuple(selected)


def _translated_frames_one_at_a_time(
    sequence: str | bytes,
):
    if isinstance(sequence, str):
        raw = sequence.encode("ascii")
    elif isinstance(sequence, bytes):
        raw = sequence
    else:
        raise TypeError("sequence must be str or ASCII bytes")
    if not raw:
        raise ValueError("sequence must be non-empty")
    nucleotide = _NT_LOOKUP[np.frombuffer(raw, dtype=np.uint8)]
    for orientation in range(2):
        strand = (
            nucleotide
            if orientation == 0
            else _NT_COMPLEMENT[nucleotide][::-1]
        )
        for offset in range(3):
            codons = (len(strand) - offset) // 3
            if codons <= 0:
                yield np.empty(0, dtype=np.uint8)
                continue
            stop = offset + 3 * codons
            code = strand[offset:stop:3].astype(np.int16, copy=True)
            code *= 25
            code += strand[offset + 1 : stop : 3] * 5
            code += strand[offset + 2 : stop : 3]
            yield _CODON_LOOKUP[code]


def _segments_bounded(
    frame: np.ndarray, *, scan_chunk_tokens: int
):
    start = 0
    for chunk_start in range(0, len(frame), scan_chunk_tokens):
        chunk_stop = min(chunk_start + scan_chunk_tokens, len(frame))
        stops = np.flatnonzero(
            frame[chunk_start:chunk_stop] == STOP_ID
        )
        for local_stop in stops:
            stop = chunk_start + int(local_stop)
            if stop > start:
                yield frame[start:stop]
            start = stop + 1
    if start < len(frame):
        yield frame[start:]


def _could_enter(
    ranked: list[tuple[int, str]], length: int, maximum_orfs: int
) -> bool:
    return len(ranked) < maximum_orfs or -length <= ranked[-1][0]


def _insert_ranked(
    ranked: list[tuple[int, str]],
    row: tuple[int, str],
    maximum_orfs: int,
) -> None:
    if row in ranked:
        return
    ranked.append(row)
    ranked.sort()
    del ranked[maximum_orfs:]


def select_long_orfs_from_sequence(
    sequence: str | bytes,
    *,
    maximum_orfs: int = 2,
    minimum_length: int = 20,
    maximum_length: int = 1_000,
    scan_chunk_tokens: int = 1_000_000,
) -> tuple[str, ...]:
    """Select frozen top ORFs without materializing six frames or all segments.

    The ranking and truncation are identical to :func:`select_long_orfs`.
    Only one translated frame, one bounded stop-index chunk, and the current
    best rows are retained, which prevents stop-rich long contigs from creating
    an unbounded Python object list during deployment inference.
    """

    if (
        maximum_orfs < 1
        or minimum_length < 1
        or maximum_length < minimum_length
        or scan_chunk_tokens < 1
    ):
        raise ValueError("invalid ORF selection bounds")
    candidates: list[tuple[int, str]] = []
    fallback: list[tuple[int, str]] = []
    for frame in _translated_frames_one_at_a_time(sequence):
        for segment in _segments_bounded(
            frame, scan_chunk_tokens=scan_chunk_tokens
        ):
            length = len(segment)
            candidate_eligible = length >= minimum_length
            keep_fallback = _could_enter(fallback, length, maximum_orfs)
            keep_candidate = candidate_eligible and _could_enter(
                candidates, length, maximum_orfs
            )
            if not keep_fallback and not keep_candidate:
                continue
            amino_acid = "".join(_TOKEN_TO_AA[segment].tolist())
            row = (-length, amino_acid)
            if keep_fallback:
                _insert_ranked(fallback, row, maximum_orfs)
            if keep_candidate:
                _insert_ranked(candidates, row, maximum_orfs)
    ranked = candidates if candidates else fallback
    if not ranked:
        raise ValueError("six-frame translation contains no amino-acid token")
    selected = []
    for _negative_length, amino_acid in ranked:
        if len(amino_acid) > maximum_length:
            left = maximum_length // 2
            right = maximum_length - left
            amino_acid = amino_acid[:left] + amino_acid[-right:]
        selected.append(amino_acid)
    return tuple(selected)


__all__ = ["select_long_orfs", "select_long_orfs_from_sequence"]
