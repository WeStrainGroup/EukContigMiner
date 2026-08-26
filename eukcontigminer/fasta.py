"""Streaming FASTA input with the locked legal-contig contract."""

from __future__ import annotations

import gzip
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, TextIO

from .contracts import normalize_sequence


def _open_text(path: str | Path):
    raw = str(path)
    if raw == "-":
        return nullcontext(sys.stdin)
    if raw.endswith(".gz"):
        return gzip.open(raw, "rt", encoding="utf-8")
    return Path(raw).open("rt", encoding="utf-8")


def _validated_record(
    identifier: str | None,
    parts: list[str],
) -> tuple[str, str]:
    if identifier is None:
        raise ValueError("FASTA contains no record")
    try:
        sequence = normalize_sequence("".join(parts))
    except ValueError as error:
        raise ValueError(f"invalid FASTA record {identifier!r}: {error}") from error
    return identifier, sequence


def fasta_records(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield unique IDs and normalized legal IUPAC-DNA whole contigs.

    Blank physical lines and header descriptions are ignored. Empty records,
    embedded whitespace, gaps, non-IUPAC symbols, duplicate IDs, and sequence
    text before the first header fail closed before a score can be emitted.
    """

    seen: set[str] = set()
    identifier: str | None = None
    parts: list[str] = []
    handle_context = _open_text(path)
    with handle_context as handle:
        input_handle: TextIO = handle
        for raw_line in input_handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield _validated_record(identifier, parts)
                header = line[1:].split(maxsplit=1)
                if not header:
                    raise ValueError("empty FASTA identifier")
                identifier = header[0]
                if identifier in seen:
                    raise ValueError(f"duplicate FASTA identifier: {identifier}")
                seen.add(identifier)
                parts = []
            elif identifier is None:
                raise ValueError("sequence appears before the first FASTA header")
            else:
                parts.append(line)
    if identifier is None:
        raise ValueError("input FASTA has no records")
    yield _validated_record(identifier, parts)


__all__ = ["fasta_records"]
