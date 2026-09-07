"""Deterministic, species-balanced sampling from frozen development genomes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .contracts import (
    BENCHMARK_LENGTHS_BP,
    DATA_CUTOFF_EXCLUSIVE,
    binary_truth_from_source_label,
    normalize_sequence,
)

DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
NUCLEAR_LABELS = (
    "Archaea",
    "Bacteria",
    "Fungi",
    "Metazoa",
    "Other_Eukaryota",
    "Viridiplantae",
)
CANARY_LABELS = (*NUCLEAR_LABELS, "Organelle")
COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVN",
    "TGCAYRSWMKVHDBN",
)


@dataclass(frozen=True, slots=True)
class AssemblySource:
    assembly_accession: str
    species_id: str
    split: str
    source_label: str
    fasta_path: str
    n50: int
    output_bases: int
    source_kind: str = "nuclear"


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    assembly_accession: str
    species_id: str
    split: str
    source_label: str
    source_kind: str
    fasta_path: str
    record_id: str
    record_length: int
    host_subclass: str = ""
    organelle_type: str = ""

    @property
    def source_key(self) -> tuple[str, str]:
        return self.fasta_path, self.record_id


@dataclass(frozen=True, slots=True)
class PlannedFragment:
    fragment_id: str
    split: str
    source_label: str
    truth: int
    source_kind: str
    species_id: str
    assembly_accession: str
    record_id: str
    fasta_path: str
    start: int
    end: int
    length_bp: int
    host_subclass: str = ""
    organelle_type: str = ""
    sequence_sha256: str = ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=16 * 1024 * 1024) as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sequence_sha256(sequence: str) -> str:
    """Hash a sequence identically in forward and reverse-complement orientation."""

    normalized = normalize_sequence(sequence)
    reverse = normalized.translate(COMPLEMENT)[::-1]
    canonical = normalized if normalized <= reverse else reverse
    return hashlib.sha256(canonical.encode()).hexdigest()


def acgt_fraction(sequence: str) -> float:
    """Return the fraction of unambiguous A/C/G/T bases in a legal sequence."""

    normalized = normalize_sequence(sequence)
    return sum(base in "ACGT" for base in normalized) / len(normalized)


def _stable_digest(*values: object, seed: int) -> str:
    payload = "\x1f".join(map(str, (*values, seed)))
    return hashlib.sha256(payload.encode()).hexdigest()


def _open_text(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, encoding="utf-8")


def fasta_record_lengths(path: str | Path) -> dict[str, int]:
    """Read only FASTA headers and line lengths; reject duplicate/empty records."""

    lengths: dict[str, int] = {}
    identifier: str | None = None
    length = 0
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if length == 0:
                        raise ValueError(f"empty FASTA record: {identifier}")
                    lengths[identifier] = length
                identifier = line[1:].split(maxsplit=1)[0]
                if not identifier or identifier in lengths:
                    raise ValueError(f"empty or duplicate FASTA identifier: {identifier!r}")
                length = 0
            else:
                if identifier is None:
                    raise ValueError("sequence appears before FASTA header")
                length += len(line)
    if identifier is not None:
        if length == 0:
            raise ValueError(f"empty FASTA record: {identifier}")
        lengths[identifier] = length
    if not lengths:
        raise ValueError(f"FASTA has no records: {path}")
    return lengths


def read_development_assemblies(
    genome_manifest: str | Path,
    split_manifests: Sequence[str | Path],
) -> list[AssemblySource]:
    """Join clean materialized FASTA views to Train/Validation taxonomy rows."""

    genome_by_accession: dict[str, dict[str, str]] = {}
    with Path(genome_manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["split"] not in DEVELOPMENT_SPLITS:
                continue
            accession = row["accession"]
            if accession in genome_by_accession:
                raise ValueError(f"duplicate development genome: {accession}")
            genome_by_accession[accession] = row

    sources: list[AssemblySource] = []
    species_split: dict[str, str] = {}
    seen_accessions: set[str] = set()
    for manifest in split_manifests:
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                split = row["split"]
                if split not in DEVELOPMENT_SPLITS:
                    raise ValueError(f"non-development split manifest supplied: {split}")
                accession = row["accession"]
                if accession in seen_accessions:
                    raise ValueError(f"duplicate split accession: {accession}")
                seen_accessions.add(accession)
                genome = genome_by_accession.get(accession)
                if genome is None or genome["split"] != split:
                    raise ValueError(f"genome/split manifest mismatch: {accession}")
                if genome["subclass"] != row["subclass"]:
                    raise ValueError(f"subclass mismatch: {accession}")
                species = row["species_taxid"]
                previous = species_split.setdefault(species, split)
                if previous != split:
                    raise ValueError(f"species leakage: {species}")
                path = genome["view_path"]
                if not Path(path).is_absolute():
                    raise ValueError(f"non-absolute clean FASTA path: {accession}")
                sources.append(
                    AssemblySource(
                        assembly_accession=accession,
                        species_id=species,
                        split=split,
                        source_label=row["subclass"],
                        fasta_path=path,
                        n50=int(row["n50"]),
                        output_bases=int(genome["output_bases"]),
                    )
                )
    if seen_accessions != set(genome_by_accession):
        raise ValueError("development split manifests do not cover clean genome manifest")
    return sources


def read_train_assemblies_from_accession_summaries(
    train_manifest: str | Path,
    accession_summary_root: str | Path,
) -> list[AssemblySource]:
    """Load only explicit Train rows and their per-accession clean FASTA summaries.

    This avoids opening a combined Train/Validation/Final genome manifest.  The
    accession list comes exclusively from ``train_manifest``; every referenced
    materialization summary must independently attest that it is a Train
    nuclear-candidate view and agree with the frozen taxonomy/source metadata.
    """

    root = Path(accession_summary_root)
    sources: list[AssemblySource] = []
    seen_accessions: set[str] = set()
    seen_species: set[str] = set()
    with Path(train_manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = row["accession"]
            species = row["species_taxid"]
            label = row["subclass"]
            if row["split"] != "train":
                raise ValueError(f"non-Train row supplied: {accession}")
            if row["release_date"] >= DATA_CUTOFF_EXCLUSIVE:
                raise ValueError(f"post-cutoff Train genome: {accession}")
            if label not in NUCLEAR_LABELS:
                raise ValueError(f"unsupported Train subclass: {label}")
            if accession in seen_accessions:
                raise ValueError(f"duplicate Train accession: {accession}")
            if species in seen_species:
                raise ValueError(f"duplicate Train species: {species}")
            seen_accessions.add(accession)
            seen_species.add(species)

            summary_path = root / accession / "summary.json"
            with summary_path.open(encoding="utf-8") as summary_handle:
                summary = json.load(summary_handle)
            if (
                summary.get("status") != "materialized_clean_fasta"
                or summary.get("accession") != accession
                or summary.get("split") != "train"
                or str(summary.get("species_taxid")) != species
                or summary.get("subclass") != label
            ):
                raise ValueError(f"Train materialization summary mismatch: {accession}")
            source = summary.get("source")
            clean_object = summary.get("object")
            counts = summary.get("counts")
            if not all(isinstance(value, dict) for value in (source, clean_object, counts)):
                raise ValueError(f"incomplete Train materialization summary: {accession}")
            assert isinstance(source, dict)
            assert isinstance(clean_object, dict)
            assert isinstance(counts, dict)
            if (
                source.get("path") != row["source_fasta"]
                or source.get("sha256") != row["source_sha256"]
            ):
                raise ValueError(f"Train source provenance mismatch: {accession}")
            clean_path = Path(str(clean_object.get("path", "")))
            if not clean_path.is_absolute() or not clean_path.is_file():
                raise ValueError(f"missing clean Train FASTA: {accession}")
            output_bases = int(counts.get("output_bases", 0))
            if output_bases <= 0:
                raise ValueError(f"empty clean Train FASTA: {accession}")
            sources.append(
                AssemblySource(
                    assembly_accession=accession,
                    species_id=species,
                    split="train",
                    source_label=label,
                    fasta_path=str(clean_path),
                    n50=int(row["n50"]),
                    output_bases=output_bases,
                )
            )
    if not sources:
        raise ValueError("Train manifest is empty")
    return sources


def select_train_nuclear_sources(
    sources: Iterable[AssemblySource],
    *,
    species_caps: Mapping[str, int],
    seed: int,
    minimum_n50: int = 100_000,
    minimum_output_bases: int = 1_500_000,
) -> list[AssemblySource]:
    """Select a deterministic, label-balanced Train-only nuclear cohort."""

    if set(species_caps) != set(NUCLEAR_LABELS) or any(
        int(value) < 1 for value in species_caps.values()
    ):
        raise ValueError("species caps must provide every nuclear label")
    grouped: defaultdict[str, list[AssemblySource]] = defaultdict(list)
    for source in sources:
        if source.split != "train" or source.source_label not in NUCLEAR_LABELS:
            raise ValueError("source is outside the Train nuclear contract")
        if source.n50 >= minimum_n50 and source.output_bases >= minimum_output_bases:
            grouped[source.source_label].append(source)
    selected: list[AssemblySource] = []
    for label in NUCLEAR_LABELS:
        ordered = sorted(
            grouped[label],
            key=lambda row: _stable_digest(
                label, row.species_id, row.assembly_accession, seed=seed
            ),
        )
        cap = int(species_caps[label])
        if len(ordered) < cap:
            raise ValueError(f"insufficient eligible Train species for {label}: {len(ordered)}")
        selected.extend(ordered[:cap])
    return sorted(
        selected,
        key=lambda row: (row.source_label, row.species_id, row.assembly_accession),
    )


def read_organelle_sequence_records(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = row["split"]
            if split not in DEVELOPMENT_SPLITS:
                raise ValueError(f"organelle row is not development data: {split}")
            records.append(
                SequenceRecord(
                    assembly_accession=row["assembly_accession"],
                    species_id=row["host_species_taxid"],
                    split=split,
                    source_label="Organelle",
                    source_kind="organelle",
                    fasta_path=row["materialized_archive"],
                    record_id=row["record_accession"],
                    record_length=int(row["record_length"]),
                    host_subclass=row["host_subclass"],
                    organelle_type=row["organelle_type"],
                )
            )
    if not records:
        raise ValueError("organelle manifest is empty")
    return records


def select_nuclear_canary_sources(
    sources: Iterable[AssemblySource],
    *,
    species_per_label: int,
    seed: int,
    minimum_n50: int = 100_000,
) -> list[AssemblySource]:
    if species_per_label < 1:
        raise ValueError("species_per_label must be positive")
    grouped: defaultdict[tuple[str, str], list[AssemblySource]] = defaultdict(list)
    for source in sources:
        if source.n50 >= minimum_n50 and source.output_bases >= sum(BENCHMARK_LENGTHS_BP):
            grouped[source.split, source.source_label].append(source)
    selected: list[AssemblySource] = []
    for split in sorted(DEVELOPMENT_SPLITS):
        for label in NUCLEAR_LABELS:
            values = sorted(
                grouped[split, label],
                key=lambda row: _stable_digest(
                    split, label, row.species_id, row.assembly_accession, seed=seed
                ),
            )
            distinct: list[AssemblySource] = []
            used_species: set[str] = set()
            for value in values:
                if value.species_id in used_species:
                    continue
                distinct.append(value)
                used_species.add(value.species_id)
                if len(distinct) == species_per_label:
                    break
            if len(distinct) != species_per_label:
                raise ValueError(
                    f"insufficient nuclear canary species for {split}/{label}: {len(distinct)}"
                )
            selected.extend(distinct)
    return selected


def scan_nuclear_sources(sources: Iterable[AssemblySource]) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for source in sources:
        for record_id, length in fasta_record_lengths(source.fasta_path).items():
            records.append(
                SequenceRecord(
                    assembly_accession=source.assembly_accession,
                    species_id=source.species_id,
                    split=source.split,
                    source_label=source.source_label,
                    source_kind=source.source_kind,
                    fasta_path=source.fasta_path,
                    record_id=record_id,
                    record_length=length,
                )
            )
    return records


def scan_nuclear_sources_parallel(
    sources: Iterable[AssemblySource], *, workers: int
) -> list[SequenceRecord]:
    """Scan FASTA record lengths concurrently, returning deterministic order."""

    source_rows = sorted(
        sources,
        key=lambda row: (row.split, row.source_label, row.species_id, row.assembly_accession),
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return scan_nuclear_sources(source_rows)
    records_by_accession: dict[str, list[SequenceRecord]] = {}
    # FASTA gzip decoding and Python line parsing are CPU-heavy. Processes give
    # real parallelism; each task is one assembly so returned record inventories
    # remain bounded.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_nuclear_sources, [source]): source.assembly_accession
            for source in source_rows
        }
        for future in as_completed(futures):
            accession = futures[future]
            records_by_accession[accession] = future.result()
    return [
        record
        for source in source_rows
        for record in records_by_accession[source.assembly_accession]
    ]


def select_validation_panel_sources(
    sources: Iterable[AssemblySource], *, minimum_n50: int = 100_000
) -> list[AssemblySource]:
    """Select the species-disjoint Validation sources able to expose 100 kb records."""

    selected = [
        source
        for source in sources
        if source.split == "validation"
        and source.source_label in NUCLEAR_LABELS
        and source.n50 >= minimum_n50
    ]
    if not selected:
        raise ValueError("no Validation panel sources pass the N50 requirement")
    species = [(source.source_label, source.species_id) for source in selected]
    if len(species) != len(set(species)):
        raise ValueError("Validation panel requires one assembly per source-label/species")
    return sorted(
        selected,
        key=lambda row: (row.source_label, row.species_id, row.assembly_accession),
    )


def _planned_fragment(
    record: SequenceRecord,
    *,
    start: int,
    end: int,
    seed: int,
    phase: str,
) -> PlannedFragment:
    digest = _stable_digest(
        phase,
        record.split,
        record.source_label,
        record.species_id,
        record.assembly_accession,
        record.record_id,
        start,
        end,
        seed=seed,
    )[:20]
    return PlannedFragment(
        fragment_id=(
            f"ecm_{record.split}_{record.source_label}_{end - start}_{digest}"
        ),
        split=record.split,
        source_label=record.source_label,
        truth=binary_truth_from_source_label(record.source_label),
        source_kind=record.source_kind,
        species_id=record.species_id,
        assembly_accession=record.assembly_accession,
        record_id=record.record_id,
        fasta_path=record.fasta_path,
        start=start,
        end=end,
        length_bp=end - start,
        host_subclass=record.host_subclass,
        organelle_type=record.organelle_type,
    )


def plan_common_species_validation_nuclear(
    records: Sequence[SequenceRecord],
    *,
    positive_replicate_cap: int,
    negative_replicate_cap: int,
    seed: int,
) -> list[PlannedFragment]:
    """Plan complete 21-length replicates while holding species mix fixed by length.

    A replicate contributes one non-overlapping fragment at every benchmark
    length. A species is retained only if at least one complete replicate fits.
    Additional replicates are capped by binary truth so rare-prevalence FPR can
    be measured more precisely without allowing a few large genomes to dominate.
    """

    if positive_replicate_cap < 1 or negative_replicate_cap < 1:
        raise ValueError("replicate caps must be positive")
    grouped: defaultdict[tuple[str, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        if (
            record.split != "validation"
            or record.source_kind != "nuclear"
            or record.source_label not in NUCLEAR_LABELS
        ):
            raise ValueError("nuclear Validation planner received an ineligible record")
        grouped[record.source_label, record.species_id].append(record)

    planned: list[PlannedFragment] = []
    for (label, species_id), species_records in sorted(grouped.items()):
        cap = (
            positive_replicate_cap
            if binary_truth_from_source_label(label)
            else negative_replicate_cap
        )
        ordered_records = sorted(
            species_records,
            key=lambda record: _stable_digest(
                label, species_id, record.record_id, seed=seed
            ),
        )
        free: dict[tuple[str, str], list[tuple[int, int]]] = {}
        species_plan: list[PlannedFragment] = []
        for replicate in range(cap):
            snapshot = {key: list(value) for key, value in free.items()}
            replicate_plan: list[PlannedFragment] = []
            for length_index, length in enumerate(
                sorted(BENCHMARK_LENGTHS_BP, reverse=True)
            ):
                offset = (replicate * len(BENCHMARK_LENGTHS_BP) + length_index) % len(
                    ordered_records
                )
                allocation: tuple[SequenceRecord, int, int] | None = None
                for index in range(len(ordered_records)):
                    record = ordered_records[(offset + index) % len(ordered_records)]
                    if record.record_length < length:
                        continue
                    interval = _allocate_interval(
                        free, record, length, seed=seed + replicate
                    )
                    if interval is not None:
                        allocation = record, *interval
                        break
                if allocation is None:
                    free = snapshot
                    replicate_plan = []
                    break
                record, start, end = allocation
                replicate_plan.append(
                    _planned_fragment(
                        record,
                        start=start,
                        end=end,
                        seed=seed,
                        phase=f"nuclear_rep{replicate}",
                    )
                )
            if not replicate_plan:
                break
            species_plan.extend(replicate_plan)
        if species_plan:
            planned.extend(species_plan)
    if not planned:
        raise ValueError("no complete nuclear Validation replicate could be planned")
    return sorted(
        planned,
        key=lambda row: (row.length_bp, row.source_label, row.species_id, row.fragment_id),
    )


def plan_validation_organelle_fragments(
    records: Sequence[SequenceRecord], *, seed: int
) -> list[PlannedFragment]:
    """Use every eligible Validation organelle record without concatenation.

    Each record of at least 1 kb contributes as many distinct requested lengths
    as fit, longest first. Coordinates never overlap within a record.
    """

    eligible = [
        record
        for record in records
        if record.split == "validation"
        and record.source_kind == "organelle"
        and record.source_label == "Organelle"
        and record.record_length >= min(BENCHMARK_LENGTHS_BP)
    ]
    if not eligible:
        raise ValueError("no eligible Validation organelle record")
    free: dict[tuple[str, str], list[tuple[int, int]]] = {}
    planned: list[PlannedFragment] = []
    used_records: set[tuple[str, str]] = set()
    for record in sorted(
        eligible,
        key=lambda row: _stable_digest(
            row.host_subclass,
            row.species_id,
            row.assembly_accession,
            row.record_id,
            seed=seed,
        ),
    ):
        for length in sorted(BENCHMARK_LENGTHS_BP, reverse=True):
            interval = _allocate_interval(free, record, length, seed=seed)
            if interval is None:
                continue
            start, end = interval
            planned.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase="organelle_all_eligible",
                )
            )
            used_records.add(record.source_key)
    expected_records = {record.source_key for record in eligible}
    if used_records != expected_records:
        raise RuntimeError("an eligible organelle record did not yield a fragment")
    return sorted(
        planned,
        key=lambda row: (row.length_bp, row.host_subclass, row.species_id, row.fragment_id),
    )


def _select_organelle_species(
    records: Sequence[SequenceRecord],
    *,
    species_count: int,
    seed: int,
    fungal_hosts: bool,
) -> set[tuple[str, str]]:
    totals: Counter[tuple[str, str]] = Counter()
    maxima: Counter[tuple[str, str]] = Counter()
    host_is_fungal: dict[tuple[str, str], bool] = {}
    for record in records:
        key = record.split, record.species_id
        totals[key] += record.record_length
        maxima[key] = max(maxima[key], record.record_length)
        observed = record.host_subclass == "Fungi"
        previous = host_is_fungal.setdefault(key, observed)
        if previous != observed:
            raise ValueError(f"organelle host subclass conflict: {key}")
    selected: set[tuple[str, str]] = set()
    for split in sorted(DEVELOPMENT_SPLITS):
        candidates = [
            key
            for key in totals
            if key[0] == split
            and maxima[key] >= max(BENCHMARK_LENGTHS_BP)
            and host_is_fungal[key] == fungal_hosts
        ]
        candidates.sort(key=lambda key: _stable_digest(*key, seed=seed))
        if len(candidates) < species_count:
            raise ValueError(f"insufficient organelle canary species for {split}")
        selected.update(candidates[:species_count])
    return selected


def select_organelle_canary_records(
    records: Sequence[SequenceRecord], *, species_per_label: int, seed: int
) -> list[SequenceRecord]:
    pool_size = max(species_per_label * 2, species_per_label)
    selected_species = _select_organelle_species(
        records,
        species_count=pool_size,
        seed=seed,
        fungal_hosts=True,
    ) | _select_organelle_species(
        records,
        species_count=pool_size,
        seed=seed + 1,
        fungal_hosts=False,
    )
    return [record for record in records if (record.split, record.species_id) in selected_species]


def _allocate_interval(
    free: dict[tuple[str, str], list[tuple[int, int]]],
    record: SequenceRecord,
    length: int,
    *,
    seed: int,
) -> tuple[int, int] | None:
    intervals = free.setdefault(record.source_key, [(0, record.record_length)])
    eligible = [(index, value) for index, value in enumerate(intervals) if value[1] - value[0] >= length]
    if not eligible:
        return None
    index, (left, right) = max(eligible, key=lambda item: item[1][1] - item[1][0])
    take_right = int(_stable_digest(record.record_id, length, left, right, seed=seed)[-1], 16) % 2
    if take_right:
        start, end = right - length, right
        replacement = [(left, start)] if left < start else []
    else:
        start, end = left, left + length
        replacement = [(end, right)] if end < right else []
    intervals[index : index + 1] = replacement
    return start, end


def _stable_integer(
    lower: int,
    upper: int,
    *values: object,
    seed: int,
) -> int:
    if lower > upper:
        raise ValueError("invalid deterministic integer interval")
    width = upper - lower + 1
    return lower + int(_stable_digest(*values, seed=seed)[:16], 16) % width


def _free_intervals_excluding_fragments(
    records: Sequence[SequenceRecord],
    reserved_fragments: Sequence[PlannedFragment],
    *,
    source_kind: str,
    split: str = "validation",
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Return record intervals left after subtracting a frozen fragment panel."""

    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"unsupported development split: {split}")

    record_by_key: dict[tuple[str, str], SequenceRecord] = {}
    for record in records:
        if record.source_kind != source_kind or record.split != split:
            raise ValueError(f"continuous {split} planner received an ineligible record")
        previous = record_by_key.setdefault(record.source_key, record)
        if previous != record:
            raise ValueError(f"conflicting {split} record metadata: {record.source_key}")
    if not record_by_key:
        raise ValueError(f"continuous {split} records are empty")

    reserved_by_key: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in reserved_fragments:
        if row.source_kind != source_kind:
            continue
        if row.split != split:
            raise ValueError(f"reserved fragment is not from {split}")
        record = record_by_key.get((row.fasta_path, row.record_id))
        if record is None:
            raise ValueError(f"reserved fragment source is absent: {row.fragment_id}")
        if row.end > record.record_length:
            raise ValueError(f"reserved fragment exceeds its source: {row.fragment_id}")
        reserved_by_key[record.source_key].append((row.start, row.end))

    free: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for key, record in record_by_key.items():
        cursor = 0
        intervals: list[tuple[int, int]] = []
        for start, end in sorted(reserved_by_key.get(key, ())):
            if start < cursor:
                raise ValueError(f"reserved fragments overlap for source: {key}")
            if cursor < start:
                intervals.append((cursor, start))
            cursor = end
        if cursor < record.record_length:
            intervals.append((cursor, record.record_length))
        free[key] = intervals
    return free


def _continuous_validation_length(
    label: str,
    species_id: str,
    ordinal: int,
    *,
    seed: int,
) -> tuple[int, str]:
    """Draw an off-grid length from a fixed 60/30/10 short/middle/long profile."""

    profile_index = ordinal % 10
    if profile_index < 6:
        lower, upper, stratum = 1_000, 3_000, "short"
    elif profile_index < 9:
        lower, upper, stratum = 3_001, 10_000, "middle"
    else:
        lower, upper, stratum = 10_001, 100_000, "long"
    length = _stable_integer(
        lower,
        upper,
        label,
        species_id,
        "continuous_validation",
        ordinal,
        seed=seed,
    )
    while length in BENCHMARK_LENGTHS_BP:
        length = lower + (length - lower + 1) % (upper - lower + 1)
    return length, stratum


def plan_continuous_validation_nuclear(
    records: Sequence[SequenceRecord],
    reserved_fragments: Sequence[PlannedFragment],
    *,
    negative_fragments_per_species: int,
    positive_fragments_per_species: int,
    seed: int,
) -> list[PlannedFragment]:
    """Plan off-grid Validation fragments disjoint from the frozen 21-point panel.

    Every retained species receives a fixed binary-class quota. Lengths follow
    a deterministic 60/30/10 profile over 1-3 kb, 3-10 kb, and 10-100 kb.
    Exact benchmark lengths are excluded so this panel diagnoses interpolation
    rather than reusing the formal benchmark grid.
    """

    if negative_fragments_per_species < 1 or positive_fragments_per_species < 1:
        raise ValueError("fragments per species must be positive")
    grouped: defaultdict[tuple[str, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        if (
            record.split != "validation"
            or record.source_kind != "nuclear"
            or record.source_label not in NUCLEAR_LABELS
        ):
            raise ValueError("nuclear continuous Validation planner received an ineligible record")
        grouped[record.source_label, record.species_id].append(record)
    if not grouped:
        raise ValueError("nuclear continuous Validation records are empty")

    free = _free_intervals_excluding_fragments(
        records, reserved_fragments, source_kind="nuclear"
    )
    planned: list[PlannedFragment] = []
    for (label, species_id), species_records in sorted(grouped.items()):
        total = (
            positive_fragments_per_species
            if binary_truth_from_source_label(label)
            else negative_fragments_per_species
        )
        requested = [
            (*_continuous_validation_length(label, species_id, ordinal, seed=seed), ordinal)
            for ordinal in range(total)
        ]
        requested.sort(key=lambda item: (-item[0], item[2]))
        ordered_records = sorted(
            species_records,
            key=lambda record: _stable_digest(
                label, species_id, record.record_id, seed=seed
            ),
        )
        species_plan: list[PlannedFragment] = []
        snapshot = {record.source_key: list(free[record.source_key]) for record in species_records}
        for request_index, (length, stratum, ordinal) in enumerate(requested):
            offset = _stable_integer(
                0,
                len(ordered_records) - 1,
                label,
                species_id,
                stratum,
                ordinal,
                seed=seed,
            )
            allocation: tuple[SequenceRecord, int, int] | None = None
            for index in range(len(ordered_records)):
                record = ordered_records[(offset + index) % len(ordered_records)]
                interval = _allocate_interval(
                    free, record, length, seed=seed + request_index
                )
                if interval is not None:
                    allocation = record, *interval
                    break
            if allocation is None:
                for key, intervals in snapshot.items():
                    free[key] = intervals
                species_plan = []
                break
            record, start, end = allocation
            species_plan.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase=f"validation_continuous_{stratum}_{ordinal}",
                )
            )
        planned.extend(species_plan)
    if not planned:
        raise ValueError("no complete continuous nuclear Validation quota could be planned")
    return sorted(
        planned,
        key=lambda row: (row.source_label, row.species_id, row.length_bp, row.fragment_id),
    )


def plan_continuous_validation_organelle(
    records: Sequence[SequenceRecord],
    reserved_fragments: Sequence[PlannedFragment],
    *,
    maximum_fragments_per_record: int,
    seed: int,
) -> list[PlannedFragment]:
    """Sample unused organelle intervals without forcing overlap with the frozen panel."""

    if maximum_fragments_per_record < 1:
        raise ValueError("organelle fragment cap must be positive")
    eligible = [
        record
        for record in records
        if record.split == "validation"
        and record.source_kind == "organelle"
        and record.source_label == "Organelle"
        and record.record_length >= 1_000
    ]
    if not eligible:
        raise ValueError("no eligible continuous Validation organelle record")
    free = _free_intervals_excluding_fragments(
        eligible, reserved_fragments, source_kind="organelle"
    )
    planned: list[PlannedFragment] = []
    for record in sorted(
        eligible,
        key=lambda row: _stable_digest(
            row.host_subclass,
            row.species_id,
            row.assembly_accession,
            row.record_id,
            seed=seed,
        ),
    ):
        for ordinal in range(maximum_fragments_per_record):
            maximum_free = max((right - left for left, right in free[record.source_key]), default=0)
            # A 1,000 bp remainder has no legal off-grid length.
            if maximum_free < 1_001:
                break
            upper = min(10_000, maximum_free)
            length = _stable_integer(
                1_000,
                upper,
                record.species_id,
                record.record_id,
                "continuous_validation_organelle",
                ordinal,
                seed=seed,
            )
            while length in BENCHMARK_LENGTHS_BP:
                length = 1_000 + (length - 1_000 + 1) % (upper - 1_000 + 1)
            interval = _allocate_interval(free, record, length, seed=seed + ordinal)
            if interval is None:
                raise RuntimeError(f"failed to allocate Validation organelle record: {record.source_key}")
            start, end = interval
            planned.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase=f"validation_continuous_organelle_{ordinal}",
                )
            )
    return sorted(
        planned,
        key=lambda row: (row.host_subclass, row.species_id, row.length_bp, row.fragment_id),
    )


def plan_continuous_train_nuclear(
    records: Sequence[SequenceRecord],
    *,
    negative_fragments_per_species: int,
    positive_fragments_per_species: int,
    short_fraction: float,
    seed: int,
    long_fraction: float = 0.0,
    long_lengths: Sequence[int] = (15_000, 30_000, 50_000, 100_000),
    skip_unallocatable_requests: bool = False,
    reserved_fragments: Sequence[PlannedFragment] = (),
) -> list[PlannedFragment]:
    """Plan short-heavy continuous Train fragments with global no-overlap.

    Nuclear lengths are deterministic continuous draws in the requested
    short/middle/long proportions.  A small set of longer anchors prevents a
    short-only specialist from silently degrading complete-contig behavior.
    """

    if negative_fragments_per_species < 1 or positive_fragments_per_species < 1:
        raise ValueError("fragments per species must be positive")
    if not 0.0 < short_fraction < 1.0:
        raise ValueError("short fraction must be in (0, 1)")
    if not 0.0 <= long_fraction < 1.0 or short_fraction + long_fraction >= 1.0:
        raise ValueError("short and long fractions must leave a positive middle fraction")
    if not long_lengths or any(length < 10_001 or length > 100_000 for length in long_lengths):
        raise ValueError("long Train lengths must be in [10001, 100000]")

    grouped: defaultdict[tuple[str, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        if (
            record.split != "train"
            or record.source_kind != "nuclear"
            or record.source_label not in NUCLEAR_LABELS
        ):
            raise ValueError("nuclear Train planner received an ineligible record")
        grouped[record.source_label, record.species_id].append(record)
    if not grouped:
        raise ValueError("nuclear Train records are empty")

    reserved_nuclear_accessions = {
        row.assembly_accession
        for row in reserved_fragments
        if row.source_kind == "nuclear" and row.split == "train"
    }
    free = _free_intervals_excluding_fragments(
        records,
        reserved_fragments,
        source_kind="nuclear",
        split="train",
    )
    planned: list[PlannedFragment] = []
    for (label, species_id), species_records in sorted(grouped.items()):
        group_start = len(planned)
        truth = binary_truth_from_source_label(label)
        total = (
            positive_fragments_per_species
            if truth
            else negative_fragments_per_species
        )
        short_count = round(total * short_fraction)
        long_count = round(total * long_fraction)
        middle_count = total - short_count - long_count
        requested: list[tuple[int, str]] = []
        for ordinal in range(short_count):
            requested.append(
                (
                    _stable_integer(
                        1_000,
                        3_000,
                        label,
                        species_id,
                        "short",
                        ordinal,
                        seed=seed,
                    ),
                    f"short_{ordinal}",
                )
            )
        for ordinal in range(middle_count):
            requested.append(
                (
                    _stable_integer(
                        3_001,
                        10_000,
                        label,
                        species_id,
                        "middle",
                        ordinal,
                        seed=seed,
                    ),
                    f"middle_{ordinal}",
                )
            )
        for ordinal in range(long_count):
            requested.append(
                (
                    _stable_integer(
                        10_001,
                        100_000,
                        label,
                        species_id,
                        "long_continuous",
                        ordinal,
                        seed=seed,
                    ),
                    f"long_continuous_{ordinal}",
                )
            )
        requested.extend((int(length), f"long_{index}") for index, length in enumerate(long_lengths))
        requested.sort(key=lambda item: (-item[0], item[1]))
        ordered_records = sorted(
            species_records,
            key=lambda record: _stable_digest(
                label, species_id, record.record_id, seed=seed
            ),
        )
        for request_index, (length, phase) in enumerate(requested):
            offset = _stable_integer(
                0,
                len(ordered_records) - 1,
                label,
                species_id,
                phase,
                request_index,
                seed=seed,
            )
            allocation: tuple[SequenceRecord, int, int] | None = None
            for index in range(len(ordered_records)):
                record = ordered_records[(offset + index) % len(ordered_records)]
                if record.record_length < length:
                    continue
                interval = _allocate_interval(free, record, length, seed=seed + request_index)
                if interval is not None:
                    allocation = record, *interval
                    break
            if allocation is None:
                if skip_unallocatable_requests:
                    continue
                raise ValueError(
                    f"insufficient non-overlapping Train sequence for {label}/{species_id}/{length}"
                )
            record, start, end = allocation
            planned.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase=f"train_continuous_{phase}",
                )
            )
        if len(planned) == group_start and skip_unallocatable_requests:
            fallback = next(
                (
                    record
                    for record in ordered_records
                    if max(
                        (right - left for left, right in free[record.source_key]),
                        default=0,
                    )
                    >= 1_000
                ),
                None,
            )
            if fallback is None:
                if (
                    species_records[0].assembly_accession
                    in reserved_nuclear_accessions
                ):
                    continue
                raise ValueError(
                    "Train genome has neither a legal unused >=1000 bp interval "
                    f"nor a reserved fragment: {label}/{species_id}"
                )
            interval = _allocate_interval(free, fallback, 1_000, seed=seed)
            if interval is None:
                raise RuntimeError(
                    f"failed to allocate Train fallback: {label}/{species_id}"
                )
            start, end = interval
            planned.append(
                _planned_fragment(
                    fallback,
                    start=start,
                    end=end,
                    seed=seed,
                    phase="train_continuous_fallback_1000",
                )
            )
    return sorted(
        planned,
        key=lambda row: (row.source_label, row.species_id, row.length_bp, row.fragment_id),
    )


def plan_continuous_train_organelle(
    records: Sequence[SequenceRecord],
    *,
    maximum_continuous_fragments_per_record: int,
    seed: int,
    reserved_fragments: Sequence[PlannedFragment] = (),
) -> list[PlannedFragment]:
    """Use every eligible Train organelle record as a positive without joining records."""

    if maximum_continuous_fragments_per_record < 1:
        raise ValueError("organelle fragment cap must be positive")
    eligible = [
        record
        for record in records
        if record.split == "train"
        and record.source_kind == "organelle"
        and record.source_label == "Organelle"
        and record.record_length >= 1_000
    ]
    if not eligible:
        raise ValueError("no eligible Train organelle record")
    free = _free_intervals_excluding_fragments(
        eligible,
        reserved_fragments,
        source_kind="organelle",
        split="train",
    )
    planned: list[PlannedFragment] = []
    used = {
        (row.fasta_path, row.record_id)
        for row in reserved_fragments
        if row.split == "train" and row.source_kind == "organelle"
    }
    for record in sorted(
        eligible,
        key=lambda row: _stable_digest(
            row.host_subclass,
            row.species_id,
            row.assembly_accession,
            row.record_id,
            seed=seed,
        ),
    ):
        maximum_free = max(
            (right - left for left, right in free[record.source_key]),
            default=0,
        )
        anchor_request: tuple[int, str] | None = None
        for anchor in (100_000, 50_000, 10_000):
            if maximum_free >= anchor:
                anchor_request = anchor, f"anchor_{anchor}"
                break
        if anchor_request is not None:
            length, phase = anchor_request
            interval = _allocate_interval(free, record, length, seed=seed)
            if interval is None:
                raise RuntimeError(
                    f"failed to allocate Train organelle record: {record.source_key}"
                )
            start, end = interval
            planned.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase=f"train_organelle_{phase}",
                )
            )
            used.add(record.source_key)
        for ordinal in range(maximum_continuous_fragments_per_record):
            maximum_free = max(
                (right - left for left, right in free[record.source_key]),
                default=0,
            )
            if maximum_free < 1_000:
                break
            upper = min(10_000, maximum_free)
            length = _stable_integer(
                1_000,
                upper,
                record.species_id,
                record.record_id,
                ordinal,
                seed=seed,
            )
            interval = _allocate_interval(
                free,
                record,
                length,
                seed=seed + ordinal + 1,
            )
            if interval is None:
                raise RuntimeError(
                    f"failed to allocate Train organelle record: {record.source_key}"
                )
            start, end = interval
            planned.append(
                _planned_fragment(
                    record,
                    start=start,
                    end=end,
                    seed=seed,
                    phase=f"train_organelle_continuous_{ordinal}",
                )
            )
            used.add(record.source_key)
    if used != {record.source_key for record in eligible}:
        raise RuntimeError("an eligible Train organelle record did not yield a fragment")
    return sorted(
        planned,
        key=lambda row: (row.host_subclass, row.species_id, row.length_bp, row.fragment_id),
    )


def plan_balanced_canary(
    records: Sequence[SequenceRecord],
    *,
    fragments_per_label_length: int,
    seed: int,
    fungal_organelle_per_length: int = 0,
) -> list[PlannedFragment]:
    """Plan exact 21-length strata with distinct species and global no-overlap."""

    if fragments_per_label_length < 1:
        raise ValueError("fragments_per_label_length must be positive")
    if not 0 <= fungal_organelle_per_length <= fragments_per_label_length:
        raise ValueError("invalid fungal organelle quota")
    grouped: defaultdict[tuple[str, str, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        if record.split not in DEVELOPMENT_SPLITS or record.source_label not in CANARY_LABELS:
            raise ValueError("record is outside canary contract")
        grouped[record.split, record.source_label, record.species_id].append(record)

    free: dict[tuple[str, str], list[tuple[int, int]]] = {}
    planned: list[PlannedFragment] = []
    for split in sorted(DEVELOPMENT_SPLITS):
        for label in CANARY_LABELS:
            species = sorted(
                {key[2] for key in grouped if key[:2] == (split, label)},
                key=lambda value: _stable_digest(split, label, value, seed=seed),
            )
            for length in sorted(BENCHMARK_LENGTHS_BP, reverse=True):
                ordered_species = sorted(
                    species,
                    key=lambda value: _stable_digest(split, label, length, value, seed=seed),
                )
                if label == "Organelle" and fungal_organelle_per_length:
                    fungal_species = [
                        species_id
                        for species_id in ordered_species
                        if any(
                            record.host_subclass == "Fungi"
                            for record in grouped[split, label, species_id]
                        )
                    ]
                    other_species = [
                        species_id
                        for species_id in ordered_species
                        if species_id not in set(fungal_species)
                    ]
                    ordered_species = [
                        *fungal_species,
                        *other_species,
                    ]
                    quotas = (
                        (set(fungal_species), fungal_organelle_per_length),
                        (
                            set(other_species),
                            fragments_per_label_length - fungal_organelle_per_length,
                        ),
                    )
                else:
                    quotas = ((set(ordered_species), fragments_per_label_length),)
                chosen = 0
                used_species: set[str] = set()
                for allowed_species, quota in quotas:
                    group_chosen = 0
                    for species_id in ordered_species:
                        if species_id not in allowed_species or species_id in used_species:
                            continue
                        candidate_records = sorted(
                            grouped[split, label, species_id],
                            key=lambda record: _stable_digest(
                                split, label, length, species_id, record.record_id, seed=seed
                            ),
                        )
                        allocation: tuple[SequenceRecord, int, int] | None = None
                        for record in candidate_records:
                            interval = _allocate_interval(free, record, length, seed=seed)
                            if interval is not None:
                                allocation = record, *interval
                                break
                        if allocation is None:
                            continue
                        record, start, end = allocation
                        digest = _stable_digest(
                            split,
                            label,
                            species_id,
                            record.assembly_accession,
                            record.record_id,
                            start,
                            end,
                            seed=seed,
                        )[:20]
                        planned.append(
                            PlannedFragment(
                                fragment_id=f"ecm_{split}_{label}_{length}_{digest}",
                                split=split,
                                source_label=label,
                                truth=binary_truth_from_source_label(label),
                                source_kind=record.source_kind,
                                species_id=species_id,
                                assembly_accession=record.assembly_accession,
                                record_id=record.record_id,
                                fasta_path=record.fasta_path,
                                start=start,
                                end=end,
                                length_bp=length,
                                host_subclass=record.host_subclass,
                                organelle_type=record.organelle_type,
                            )
                        )
                        used_species.add(species_id)
                        chosen += 1
                        group_chosen += 1
                        if group_chosen == quota:
                            break
                    if group_chosen != quota:
                        raise ValueError(
                            f"insufficient organelle host-group fragments for "
                            f"{split}/{label}/{length}: {group_chosen}/{quota}"
                        )
                if chosen != fragments_per_label_length:
                    raise ValueError(
                        f"insufficient fragments for {split}/{label}/{length}: {chosen}"
                    )
    validate_fragment_plan(planned, expected_per_stratum=fragments_per_label_length)
    return sorted(planned, key=lambda row: (row.split, row.length_bp, row.source_label, row.fragment_id))


def validate_fragment_plan(
    rows: Sequence[PlannedFragment], *, expected_per_stratum: int | None = None
) -> None:
    if not rows or len({row.fragment_id for row in rows}) != len(rows):
        raise ValueError("fragment plan is empty or has duplicate IDs")
    species_split: dict[str, str] = {}
    intervals: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    counts: Counter[tuple[str, str, int]] = Counter()
    species_by_stratum: defaultdict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.end - row.start != row.length_bp or row.start < 0:
            raise ValueError(f"invalid fragment coordinates: {row.fragment_id}")
        if row.truth != binary_truth_from_source_label(row.source_label):
            raise ValueError(f"invalid truth: {row.fragment_id}")
        previous = species_split.setdefault(row.species_id, row.split)
        if previous != row.split:
            raise ValueError(f"fragment species leakage: {row.species_id}")
        intervals[row.fasta_path, row.record_id].append((row.start, row.end))
        key = row.split, row.source_label, row.length_bp
        counts[key] += 1
        species_by_stratum[key].add(row.species_id)
    for key, values in intervals.items():
        values.sort()
        for previous, current in zip(values, values[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"fragment overlap in {key}: {previous}, {current}")
    if expected_per_stratum is not None:
        expected_keys = {
            (split, label, length)
            for split in DEVELOPMENT_SPLITS
            for label in CANARY_LABELS
            for length in BENCHMARK_LENGTHS_BP
        }
        if set(counts) != expected_keys:
            raise ValueError("fragment plan lacks required strata")
        for key in expected_keys:
            if counts[key] != expected_per_stratum:
                raise ValueError(f"fragment count mismatch: {key}")
            if len(species_by_stratum[key]) != expected_per_stratum:
                raise ValueError(f"species diversity mismatch: {key}")


def _extract_requested_fragments(
    path: str,
    requests: Mapping[str, Sequence[PlannedFragment]],
) -> dict[str, str]:
    collected: dict[str, list[str]] = {
        row.fragment_id: [] for rows in requests.values() for row in rows
    }
    identifier: str | None = None
    position = 0
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                identifier = line[1:].split(maxsplit=1)[0]
                position = 0
                continue
            if identifier is None:
                raise ValueError("sequence appears before FASTA header")
            sequence = line.upper()
            line_start, line_end = position, position + len(sequence)
            for row in requests.get(identifier, ()):
                overlap_start = max(line_start, row.start)
                overlap_end = min(line_end, row.end)
                if overlap_start < overlap_end:
                    collected[row.fragment_id].append(
                        sequence[overlap_start - line_start : overlap_end - line_start]
                    )
            position = line_end
    result = {identifier: "".join(parts) for identifier, parts in collected.items()}
    for rows in requests.values():
        for row in rows:
            sequence = result[row.fragment_id]
            if len(sequence) != row.length_bp:
                raise RuntimeError(f"failed to extract complete fragment: {row.fragment_id}")
            normalize_sequence(sequence)
    return result


def _atomic_write_tsv(path: Path, rows: Sequence[PlannedFragment]) -> None:
    temporary = path.with_name(path.name + ".part")
    fields = tuple(PlannedFragment.__dataclass_fields__)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: getattr(row, field) for field in fields} for row in rows)
    os.replace(temporary, path)


def write_fragment_plan(path: str | Path, rows: Sequence[PlannedFragment]) -> dict[str, object]:
    validate_fragment_plan(rows)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_tsv(destination, rows)
    return {
        "rows": len(rows),
        "bases": sum(row.length_bp for row in rows),
        "path": str(destination),
        "sha256": sha256_file(destination),
    }


def read_fragment_plan(path: str | Path) -> list[PlannedFragment]:
    integer_fields = {"truth", "start", "end", "length_bp"}
    rows: list[PlannedFragment] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = set(PlannedFragment.__dataclass_fields__)
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("fragment plan columns differ from the locked schema")
        for raw in reader:
            values: dict[str, str | int] = {
                key: (int(value) if key in integer_fields else value)
                for key, value in raw.items()
            }
            rows.append(PlannedFragment(**values))
    validate_fragment_plan(rows)
    return rows


def materialize_fragment_plan(
    rows: Sequence[PlannedFragment],
    output_root: str | Path,
    *,
    workers: int = 1,
    minimum_acgt_fraction: float | None = None,
    drop_canonical_duplicates: bool = False,
    input_plan: str | Path | None = None,
) -> dict[str, object]:
    validate_fragment_plan(rows)
    if workers < 1:
        raise ValueError("workers must be positive")
    if minimum_acgt_fraction is not None and not 0.0 < minimum_acgt_fraction <= 1.0:
        raise ValueError("minimum_acgt_fraction must be in (0, 1]")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    requests_by_path: defaultdict[str, defaultdict[str, list[PlannedFragment]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        requests_by_path[row.fasta_path][row.record_id].append(row)
    sequences: dict[str, str] = {}
    paths = sorted(requests_by_path)
    if workers == 1:
        for path in paths:
            sequences.update(_extract_requested_fragments(path, requests_by_path[path]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _extract_requested_fragments, path, requests_by_path[path]
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                extracted = future.result()
                duplicate_ids = set(sequences).intersection(extracted)
                if duplicate_ids:
                    raise RuntimeError(
                        f"fragment IDs occur in multiple FASTA sources: {sorted(duplicate_ids)[:5]}"
                    )
                sequences.update(extracted)
    sequence_sha = {
        row.fragment_id: hashlib.sha256(sequences[row.fragment_id].encode()).hexdigest()
        for row in rows
    }
    fractions = {
        row.fragment_id: acgt_fraction(sequences[row.fragment_id]) for row in rows
    }
    quality_ids = {
        row.fragment_id
        for row in rows
        if minimum_acgt_fraction is None
        or fractions[row.fragment_id] >= minimum_acgt_fraction
    }
    canonical_by_id = {
        row.fragment_id: canonical_sequence_sha256(sequences[row.fragment_id])
        for row in rows
        if row.fragment_id in quality_ids
    }
    canonical_counts = Counter(canonical_by_id.values())
    duplicate_canonical = {
        digest for digest, count in canonical_counts.items() if count > 1
    }
    low_quality = [row for row in rows if row.fragment_id not in quality_ids]
    duplicate_rows = [
        row
        for row in rows
        if row.fragment_id in quality_ids
        and canonical_by_id[row.fragment_id] in duplicate_canonical
    ]
    excluded_duplicate_ids = (
        {row.fragment_id for row in duplicate_rows}
        if drop_canonical_duplicates
        else set()
    )
    retained_rows = [
        row
        for row in rows
        if row.fragment_id in quality_ids
        and row.fragment_id not in excluded_duplicate_ids
    ]
    if not retained_rows:
        raise ValueError("fragment QC excluded every planned fragment")
    materialized = [
        replace(row, sequence_sha256=sequence_sha[row.fragment_id])
        for row in retained_rows
    ]
    plan_path = output / "fragments.tsv"
    fasta_path = output / "fragments.fna.gz"
    _atomic_write_tsv(plan_path, materialized)
    temporary = fasta_path.with_name(fasta_path.name + ".part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as zipped:
            for row in materialized:
                sequence = sequences[row.fragment_id]
                # One legal FASTA sequence line avoids tens of millions of
                # tiny gzip writes on large frozen panels.
                zipped.write(f">{row.fragment_id}\n{sequence}\n".encode())
    os.replace(temporary, fasta_path)
    counts = Counter((row.split, row.source_label, row.length_bp) for row in materialized)
    duplicate_truths: defaultdict[str, set[int]] = defaultdict(set)
    for row in duplicate_rows:
        duplicate_truths[canonical_by_id[row.fragment_id]].add(row.truth)
    summary = {
        "schema": "eukcontigminer.fragment_panel.v2",
        "status": "complete",
        "planned_fragments": len(rows),
        "fragments": len(materialized),
        "bases": sum(row.length_bp for row in materialized),
        "species": len({(row.split, row.species_id) for row in materialized}),
        "sequence_sha256_unique": len({row.sequence_sha256 for row in materialized}),
        "counts_by_split": dict(sorted(Counter(row.split for row in materialized).items())),
        "counts_by_label": dict(sorted(Counter(row.source_label for row in materialized).items())),
        "counts_by_truth": {
            str(key): value
            for key, value in sorted(Counter(row.truth for row in materialized).items())
        },
        "counts_by_length_and_truth": {
            f"{length}|{truth}": value
            for (length, truth), value in sorted(
                Counter((row.length_bp, row.truth) for row in materialized).items()
            )
        },
        "organelle_host_subclasses": dict(
            sorted(
                Counter(
                    row.host_subclass
                    for row in materialized
                    if row.source_label == "Organelle"
                ).items()
            )
        ),
        "fragment_qc": {
            "single_extraction_pass": True,
            "minimum_acgt_fraction": minimum_acgt_fraction,
            "canonical_exact_duplicate_key": (
                "minimum_forward_reverse_complement_sequence_sha256"
            ),
            "drop_every_member_of_canonical_duplicate_group": (
                drop_canonical_duplicates
            ),
            "low_acgt_rows_excluded": len(low_quality),
            "canonical_duplicate_groups_observed_after_acgt_filter": len(
                duplicate_canonical
            ),
            "canonical_duplicate_rows_excluded": len(excluded_duplicate_ids),
            "canonical_duplicate_groups_with_truth_conflict": sum(
                len(truths) > 1 for truths in duplicate_truths.values()
            ),
            "labels_changed": False,
        },
        "strata": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "artifacts": {
            "fragments_tsv": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "fragments_fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
        },
        "inputs": (
            {
                "fragment_plan": {
                    "path": str(Path(input_plan)),
                    "sha256": sha256_file(input_plan),
                    "rows": len(rows),
                }
            }
            if input_plan is not None
            else {}
        ),
        "gates": {
            "all_21_lengths": {row.length_bp for row in materialized} == set(BENCHMARK_LENGTHS_BP),
            "includes_all_21_formal_lengths": set(BENCHMARK_LENGTHS_BP).issubset(
                {row.length_bp for row in materialized}
            ),
            "species_disjoint": True,
            "coordinates_non_overlapping": True,
            "all_sequences_legal_iupac": True,
            "all_retained_acgt_fraction_at_least_threshold": all(
                minimum_acgt_fraction is None
                or fractions[row.fragment_id] >= minimum_acgt_fraction
                for row in materialized
            ),
            "all_retained_canonical_sequences_unique": (
                len({canonical_by_id[row.fragment_id] for row in materialized})
                == len(materialized)
            ),
            "organelle_truth_is_eukaryota": True,
            "parallel_source_workers": workers,
            "final_test_rows_or_sequences_read": sum(
                row.split == "final_test" for row in materialized
            ),
        },
    }
    summary_path = output / "summary.json"
    temporary_summary = summary_path.with_name(summary_path.name + ".part")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)
    return summary


def subset_fasta_by_ids(
    input_fasta: str | Path,
    output_fasta: str | Path,
    identifiers: Iterable[str],
) -> dict[str, object]:
    """Write a deterministic FASTA subset in source order and fail on ID drift."""

    wanted = set(identifiers)
    if not wanted:
        raise ValueError("FASTA subset IDs are empty")
    output = Path(output_fasta)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".part")
    observed: set[str] = set()
    selected = False
    raw = temporary.open("wb")
    try:
        context = (
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
            )
            if str(output).endswith(".gz")
            else raw
        )
        with context as destination, _open_text(input_fasta) as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    identifier = line[1:].split(maxsplit=1)[0]
                    selected = identifier in wanted
                    if selected:
                        if identifier in observed:
                            raise ValueError(f"duplicate selected FASTA ID: {identifier}")
                        observed.add(identifier)
                        destination.write(f">{identifier}\n".encode())
                elif selected:
                    normalize_sequence(line)
                    destination.write(line.upper().encode() + b"\n")
    finally:
        if not raw.closed:
            raw.close()
    if observed != wanted:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"FASTA subset ID mismatch; missing={sorted(wanted - observed)[:10]}"
        )
    os.replace(temporary, output)
    return {
        "records": len(observed),
        "path": str(output),
        "sha256": sha256_file(output),
    }
