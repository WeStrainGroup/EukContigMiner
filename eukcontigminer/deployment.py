"""Hash-bound one-GPU FASTA inference for the frozen DNA plus ESM model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from .contracts import classify_score, validate_probability
from .dna_ensemble import ENSEMBLE_WEIGHTS, TrainableDNAEnsemble, load_frozen_head
from .esm_inference import (
    ESM2ORFInferenceConfig,
    esm2_features_from_orfs,
    optimize_esm2_feature_inference,
    select_orfs_from_contig,
)
from .fasta import fasta_records
from .joint_inference import (
    direct_joint_probability,
    dual_probe_piecewise_probability,
)


PREDICTION_HEADER = ("contig_id", "length_bp", "p_euk", "label")
DEFAULT_CONFIG_RESOURCE = "model.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb", buffering=16 * 1024 * 1024) as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeploymentParameters:
    model_id: str
    threshold: float
    early_exit_other_max_score: float
    positive_alpha: float
    negative_alpha: float
    probe_center: float
    probe_scale: float
    secondary_probe_center: float
    secondary_probe_scale: float
    secondary_source_alpha: float
    short_alpha: float
    long_alpha: float
    piecewise_boundary_bp: int
    config: dict[str, Any]


DUAL_PROBE_FORMULA = (
    "sigmoid(reference_logit + alpha(length) * secondary_source_alpha * "
    "secondary_probe_z)"
)


class ESM2Probe(nn.Module):
    def __init__(self, feature_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def _bound_asset(
    binding: object, name: str, resources: ExitStack
) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{name} binding is missing")
    asset = str(binding.get("asset", ""))
    if not asset or Path(asset).name != asset:
        raise ValueError(f"{name} asset name is invalid")
    path = resources.enter_context(
        as_file(files("eukcontigminer.model_data").joinpath(asset))
    )
    digest = str(binding.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{name} artifact differs")
    return path


def _read_config(path: str | Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        config_path = Path(path)
        return json.loads(config_path.read_text()), str(config_path)
    resource = files("eukcontigminer.model_data").joinpath(DEFAULT_CONFIG_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8")), "bundled:model.json"


def load_deployment_parameters(
    path: str | Path | None = None,
) -> DeploymentParameters:
    config, _source = _read_config(path)
    prediction = config.get("prediction_rule")
    model = config.get("model")
    dna = model.get("dna") if isinstance(model, dict) else None
    esm2 = model.get("esm2") if isinstance(model, dict) else None
    if (
        config.get("schema") != "eukcontigminer.release_model.v2"
        or config.get("status") != "released"
        or not isinstance(prediction, dict)
        or prediction.get("comparison") != "strict_greater_than"
        or prediction.get("equal_threshold_label") != "Other"
        or not isinstance(model, dict)
        or not isinstance(dna, dict)
        or not isinstance(esm2, dict)
        or model.get("formula") != DUAL_PROBE_FORMULA
        or dna.get("ensemble_weights") != list(ENSEMBLE_WEIGHTS)
        or dna.get("unfreeze_scope") != "heads-only"
        or len(dna.get("heads", [])) != 2
        or model.get("feature_definition", {}).get("reverse_complement_invariant")
        is not True
        or model.get("feature_definition", {}).get("orfs_per_contig") != 2
        or model.get("feature_definition", {}).get("feature_dimension") != 2560
        or config.get("binary_target", {}).get("unknown_class") is not False
        or config.get("benchmark", {}).get("final_test_used_for_model_selection")
        is not False
        or not isinstance(model.get("secondary_probe"), dict)
        or model.get("piecewise_secondary_fusion", {}).get("comparison")
        != "length_less_than_or_equal"
    ):
        raise ValueError("deployment config violates the frozen model contract")
    values = {
        "threshold": float(prediction.get("threshold", math.nan)),
        "early_exit_other_max_score": float(
            model.get("dna_other_early_exit", {}).get(
                "maximum_dna_p_euk", math.nan
            )
        ),
        "positive_alpha": float(model.get("positive_alpha", math.nan)),
        "negative_alpha": float(model.get("negative_alpha", math.nan)),
        "probe_center": float(model.get("probe_logit_center", math.nan)),
        "probe_scale": float(model.get("probe_logit_scale", math.nan)),
        "secondary_probe_center": float(
            model.get("secondary_probe_logit_center", math.nan)
        ),
        "secondary_probe_scale": float(
            model.get("secondary_probe_logit_scale", math.nan)
        ),
        "secondary_source_alpha": float(
            model.get("secondary_source_alpha", math.nan)
        ),
        "short_alpha": float(
            model.get("piecewise_secondary_fusion", {}).get(
                "short_alpha", math.nan
            )
        ),
        "long_alpha": float(
            model.get("piecewise_secondary_fusion", {}).get(
                "long_alpha", math.nan
            )
        ),
        "piecewise_boundary_bp": int(
            model.get("piecewise_secondary_fusion", {}).get("boundary_bp", -1)
        ),
    }
    if (
        not all(math.isfinite(value) for value in values.values())
        or not 0.0 <= values["threshold"] <= 1.0
        or not 0.0
        <= values["early_exit_other_max_score"]
        < values["threshold"]
        or min(values["positive_alpha"], values["negative_alpha"]) < 0.0
        or values["probe_scale"] <= 0.0
        or values["secondary_probe_scale"] <= 0.0
        or values["secondary_source_alpha"] <= 0.0
        or min(values["short_alpha"], values["long_alpha"]) <= 0.0
        or values["piecewise_boundary_bp"] < 1_000
    ):
        raise ValueError("deployment probabilities or fusion parameters are invalid")
    return DeploymentParameters(
        model_id=str(config["model_id"]),
        config=config,
        **values,
    )


def _collate(
    sequences: list[str | bytearray],
    indices: np.ndarray,
    token_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [sequences[int(index)] for index in indices]
    if selected and all(isinstance(sequence, bytearray) for sequence in selected):
        lengths_array = np.fromiter(
            (len(sequence) for sequence in selected),
            dtype=np.int64,
            count=len(selected),
        )
        tokens_array = np.full(
            (len(selected), int(lengths_array.max())), 5, dtype=np.uint8
        )
        for row_index, sequence in enumerate(selected):
            tokens_array[row_index, : len(sequence)] = np.frombuffer(
                sequence, dtype=np.uint8
            )
        return torch.from_numpy(tokens_array).long(), torch.from_numpy(lengths_array)
    encoded = []
    for sequence in selected:
        if isinstance(sequence, str):
            raw = bytearray(sequence.encode("ascii", "replace"))
            values = torch.frombuffer(raw, dtype=torch.uint8).long()
            encoded.append(token_table[values])
        elif isinstance(sequence, bytearray):
            encoded.append(torch.frombuffer(sequence, dtype=torch.uint8).long())
        else:
            raise TypeError("sequence storage must be str or token bytearray")
    lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    tokens = torch.full((len(encoded), int(lengths.max())), 5, dtype=torch.long)
    for row_index, row in enumerate(encoded):
        tokens[row_index, : len(row)] = row
    return tokens, lengths


def _pretokenize_sequences(
    sequences: list[str], token_table: torch.Tensor
) -> list[bytearray]:
    table = token_table.detach().cpu().long()
    if table.shape != (256,) or torch.any(table < 0) or torch.any(table > 255):
        raise ValueError("token table cannot be represented as uint8")
    translation = bytes(int(value) for value in table.tolist())
    return [
        bytearray(sequence.encode("ascii", "replace")).translate(translation)
        for sequence in sequences
    ]


def _bounded_indices(
    lengths: np.ndarray, *, batch_size: int, max_padded_bases: int
) -> list[np.ndarray]:
    order = np.argsort(lengths, kind="stable")
    batches = []
    start = 0
    while start < len(order):
        first_length = int(lengths[order[start]])
        if first_length > max_padded_bases:
            batches.append(order[start : start + 1])
            start += 1
            continue
        stop = start + 1
        while stop < len(order) and stop - start < batch_size:
            longest = int(lengths[order[stop]])
            if (stop - start + 1) * longest > max_padded_bases:
                break
            stop += 1
        batches.append(order[start:stop])
        start = stop
    return batches


def _buffered(
    rows: Iterator[tuple[str, str]], size: int
) -> Iterator[list[tuple[str, str]]]:
    while True:
        selected = list(islice(rows, size))
        if not selected:
            return
        yield selected


def _load_probe_asset(
    binding: object,
    *,
    name: str,
    expected_schema: str,
    device: torch.device,
    feature_definition: dict[str, Any] | None = None,
    esm_sha256: str | None = None,
) -> tuple[ESM2Probe, torch.Tensor, torch.Tensor]:
    with ExitStack() as resources:
        checkpoint = _bound_asset(binding, name, resources)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{name} checkpoint schema differs")
    if feature_definition is not None:
        checkpoint_definition = payload.get("feature_definition")
        selection = (
            checkpoint_definition.get("selection")
            if isinstance(checkpoint_definition, dict)
            else None
        )
        checkpoint_model = (
            checkpoint_definition.get("model")
            if isinstance(checkpoint_definition, dict)
            else None
        )
        if (
            not isinstance(selection, dict)
            or not isinstance(checkpoint_model, dict)
            or checkpoint_definition.get("feature_dimension") != 2560
            or selection.get("maximum_orfs")
            != feature_definition.get("orfs_per_contig")
            or selection.get("minimum_orf_length")
            != feature_definition.get("minimum_orf_aa")
            or selection.get("maximum_orf_length")
            != feature_definition.get("maximum_orf_aa")
            or selection.get("reverse_complement_invariant") is not True
            or checkpoint_model.get("checkpoint_sha256") != esm_sha256
        ):
            raise ValueError(f"{name} feature definition differs")
    model_config = payload.get("model_config")
    if (
        not isinstance(model_config, dict)
        or int(model_config.get("feature_dimension", -1)) != 2560
    ):
        raise ValueError(f"{name} model configuration differs")
    probe = ESM2Probe(**model_config)
    probe.load_state_dict(payload["model_state_dict"], strict=True)
    probe.requires_grad_(False).eval().to(device)
    feature_mean = payload["feature_mean"].float().cpu()
    feature_std = payload["feature_std"].float().cpu()
    if (
        feature_mean.shape != (2560,)
        or feature_std.shape != (2560,)
        or not torch.isfinite(feature_mean).all()
        or not torch.isfinite(feature_std).all()
        or torch.any(feature_std <= 0.0)
    ):
        raise ValueError(f"{name} feature standardization differs")
    return probe, feature_mean, feature_std


def _load_models(
    parameters: DeploymentParameters, device: torch.device
) -> tuple[
    TrainableDNAEnsemble,
    torch.Tensor,
    nn.Module,
    Any,
    ESM2Probe,
    torch.Tensor,
    torch.Tensor,
    ESM2Probe,
    torch.Tensor,
    torch.Tensor,
    ESM2ORFInferenceConfig,
]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("frozen DNA+ESM prediction requires CUDA")
    torch.cuda.set_device(device)
    model = parameters.config["model"]
    dna = model["dna"]
    from eukcontigminer._model.motif_v20 import motif_length_gate
    from eukcontigminer._model.sequence import reverse_complement_batch
    from eukcontigminer.predictor import _TOKEN, WEIGHTS_SHA256, Predictor

    if WEIGHTS_SHA256 != dna.get("base_weights_sha256"):
        raise ValueError("public DNA base weights differ")
    head_rows = dna["heads"]
    with ExitStack() as resources:
        head_paths = [
            _bound_asset(row, f"DNA head {index}", resources)
            for index, row in enumerate(head_rows)
        ]
        heads = tuple(
            load_frozen_head(path, int(row["hidden_dimension"]))
            for path, row in zip(head_paths, head_rows, strict=True)
        )
    if len(heads) != 2:
        raise RuntimeError("DNA ensemble requires two heads")
    dna_model = TrainableDNAEnsemble(
        Predictor(device).model,
        (heads[0][0], heads[1][0]),
        (heads[0][1], heads[1][1]),
        reverse_complement_batch,
        motif_length_gate,
        "heads-only",
    ).to(device)
    dna_model.requires_grad_(False).eval()

    import esm

    esm_binding = model["esm2"]
    if not isinstance(esm_binding, dict):
        raise ValueError("ESM-2 binding is missing")
    cache_name = str(esm_binding.get("cache_filename", ""))
    if Path(cache_name).name != cache_name or not cache_name.endswith(".pt"):
        raise ValueError("ESM-2 cache filename is invalid")
    esm_checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / cache_name
    expected_esm_sha = str(esm_binding.get("sha256", ""))
    with torch.serialization.safe_globals([argparse.Namespace]):
        if esm_checkpoint.is_file():
            if sha256_file(esm_checkpoint) != expected_esm_sha:
                raise ValueError("cached ESM-2 checkpoint differs")
            esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(
                esm_checkpoint
            )
        else:
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            if (
                not esm_checkpoint.is_file()
                or sha256_file(esm_checkpoint) != expected_esm_sha
            ):
                raise ValueError("downloaded ESM-2 checkpoint differs")
    definition = model["feature_definition"]
    if not isinstance(getattr(esm_model, "lm_head", None), nn.Module):
        raise ValueError("ESM-2 language-model head is missing")
    esm_model.lm_head = nn.Identity()
    optimize_esm2_feature_inference(esm_model)
    esm_model.requires_grad_(False).eval().to(device)
    probe, feature_mean, feature_std = _load_probe_asset(
        model["probe"],
        name="primary ESM probe",
        expected_schema="eukcontigminer.esm2_probe.v1",
        device=device,
    )
    secondary_probe, secondary_mean, secondary_std = _load_probe_asset(
        model["secondary_probe"],
        name="secondary ESM probe",
        expected_schema="eukcontigminer.esm2_probe_full.v1",
        device=device,
        feature_definition=definition,
        esm_sha256=expected_esm_sha,
    )
    orf_config = ESM2ORFInferenceConfig(
        maximum_orfs=int(definition["orfs_per_contig"]),
        minimum_orf_length=int(definition["minimum_orf_aa"]),
        maximum_orf_length=int(definition["maximum_orf_aa"]),
        aggregation="mean_max",
    )
    return (
        dna_model,
        _TOKEN.cpu(),
        esm_model,
        alphabet.get_batch_converter(truncation_seq_length=None),
        probe,
        feature_mean,
        feature_std,
        secondary_probe,
        secondary_mean,
        secondary_std,
        orf_config,
    )


def predict_fasta(
    fasta: str | Path,
    output: str | Path,
    summary: str | Path,
    *,
    config: str | Path | None = None,
    device_name: str = "cuda:0",
    buffer_records: int = 4_096,
    dna_batch_size: int = 32,
    dna_max_padded_bases: int = 800_000,
    use_dna_early_exit: bool = True,
    minimum_length: int = 1_000,
) -> dict[str, Any]:
    if min(
        buffer_records,
        dna_batch_size,
        dna_max_padded_bases,
        minimum_length,
    ) < 1:
        raise ValueError("prediction batch bounds must be positive")
    fasta_path = Path(fasta)
    output_path = Path(output)
    summary_path = Path(summary)
    part = output_path.with_name(output_path.name + ".part")
    summary_part = summary_path.with_name(summary_path.name + ".part")
    if (
        output_path.resolve()
        in ({fasta_path.resolve()} | ({Path(config).resolve()} if config else set()))
        or summary_path.resolve()
        in (
            {fasta_path.resolve(), output_path.resolve()}
            | ({Path(config).resolve()} if config else set())
        )
        or any(
            path.exists()
            for path in (output_path, summary_path, part, summary_part)
        )
    ):
        raise FileExistsError("prediction outputs overlap an input or already exist")
    parameters = load_deployment_parameters(config)
    device = torch.device(device_name)
    (
        dna_model,
        token_table,
        esm_model,
        batch_converter,
        probe,
        feature_mean,
        feature_std,
        secondary_probe,
        secondary_mean,
        secondary_std,
        orf_config,
    ) = _load_models(parameters, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    records = bases = early_exit_other_records = esm_records = 0
    input_records = input_bases = skipped_records = skipped_bases = 0
    seen: set[str] = set()
    try:
        with part.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(PREDICTION_HEADER)
            for buffer in _buffered(iter(fasta_records(fasta_path)), buffer_records):
                identifiers = [row[0] for row in buffer]
                counts = Counter(identifiers)
                duplicates = seen.intersection(identifiers) | {
                    identifier for identifier, count in counts.items() if count > 1
                }
                if duplicates:
                    raise ValueError(
                        f"input FASTA identifiers are duplicated: {sorted(duplicates)[:5]}"
                    )
                seen.update(identifiers)
                input_records += len(buffer)
                input_bases += sum(len(row[1]) for row in buffer)
                retained = [
                    row for row in buffer if len(row[1]) >= minimum_length
                ]
                skipped_records += len(buffer) - len(retained)
                skipped_bases += sum(
                    len(row[1]) for row in buffer if len(row[1]) < minimum_length
                )
                if not retained:
                    continue
                buffer = retained
                identifiers = [row[0] for row in buffer]
                sequences = [row[1] for row in buffer]
                lengths = np.asarray([len(sequence) for sequence in sequences])
                dna_sequences = _pretokenize_sequences(sequences, token_table)
                dna_scores = np.full(len(buffer), np.nan, dtype=np.float64)
                with torch.inference_mode():
                    for indices in _bounded_indices(
                        lengths,
                        batch_size=dna_batch_size,
                        max_padded_bases=dna_max_padded_bases,
                    ):
                        tokens, selected_lengths = _collate(
                            dna_sequences, indices, token_table
                        )
                        logits = dna_model(
                            tokens.to(device), selected_lengths.to(device)
                        )
                        dna_scores[indices] = (
                            torch.sigmoid(logits.double()).cpu().numpy()
                        )
                early_exit_other = (
                    dna_scores <= parameters.early_exit_other_max_score
                    if use_dna_early_exit
                    else np.zeros(len(buffer), dtype=bool)
                )
                protein_indices = np.flatnonzero(
                    (lengths >= 3) & ~early_exit_other
                )
                probe_logit = np.full(
                    len(buffer), parameters.probe_center, dtype=np.float64
                )
                secondary_probe_logit = np.full(
                    len(buffer), parameters.secondary_probe_center, dtype=np.float64
                )
                if len(protein_indices):
                    orfs = [
                        select_orfs_from_contig(sequences[int(index)], orf_config)
                        for index in protein_indices
                    ]
                    features = esm2_features_from_orfs(
                        esm_model,
                        batch_converter,
                        orfs,
                        representation_layer=int(esm_model.num_layers),
                        device=device,
                        config=orf_config,
                    )
                    standardized = (features - feature_mean) / feature_std
                    with torch.inference_mode():
                        selected_probe_logit = (
                            probe(standardized.to(device)).double().cpu().numpy()
                        )
                    probe_logit[protein_indices] = selected_probe_logit
                    secondary_standardized = (
                        features - secondary_mean
                    ) / secondary_std
                    with torch.inference_mode():
                        selected_secondary_logit = (
                            secondary_probe(secondary_standardized.to(device))
                            .double()
                            .cpu()
                            .numpy()
                        )
                    secondary_probe_logit[
                        protein_indices
                    ] = selected_secondary_logit
                scores = direct_joint_probability(
                    dna_scores,
                    probe_logit,
                    probe_center=parameters.probe_center,
                    probe_scale=parameters.probe_scale,
                    positive_alpha=parameters.positive_alpha,
                    negative_alpha=parameters.negative_alpha,
                )
                scores = dual_probe_piecewise_probability(
                    scores,
                    secondary_probe_logit,
                    lengths,
                    secondary_probe_center=parameters.secondary_probe_center,
                    secondary_probe_scale=parameters.secondary_probe_scale,
                    secondary_source_alpha=parameters.secondary_source_alpha,
                    short_alpha=parameters.short_alpha,
                    long_alpha=parameters.long_alpha,
                    boundary_bp=parameters.piecewise_boundary_bp,
                )
                scores[early_exit_other] = dna_scores[early_exit_other]
                for identifier, sequence, score in zip(
                    identifiers, sequences, scores, strict=True
                ):
                    value = validate_probability(float(score))
                    writer.writerow(
                        (
                            identifier,
                            len(sequence),
                            format(value, ".17g"),
                            classify_score(value, parameters.threshold),
                        )
                    )
                records += len(buffer)
                bases += int(lengths.sum())
                early_exit_other_records += int(early_exit_other.sum())
                esm_records += len(protein_indices)
        if not input_records:
            raise ValueError("input FASTA has no records")
        os.replace(part, output_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "eukcontigminer.deployment_prediction.v1",
        "status": "complete",
        "model_id": parameters.model_id,
        "input_records": input_records,
        "input_bases": input_bases,
        "records": records,
        "bases": bases,
        "minimum_length_bp": minimum_length,
        "skipped_below_minimum_length_records": skipped_records,
        "skipped_below_minimum_length_bases": skipped_bases,
        "elapsed_seconds": elapsed,
        "records_per_second": records / elapsed if records else 0.0,
        "inference_mode": (
            "dna_early_exit" if use_dna_early_exit else "full_esm"
        ),
        "dna_early_exit_enabled": use_dna_early_exit,
        "dna_other_early_exit_max_score": parameters.early_exit_other_max_score,
        "dna_other_early_exit_records": early_exit_other_records,
        "dna_other_early_exit_fraction": (
            early_exit_other_records / records if records else 0.0
        ),
        "esm_records": esm_records,
        "esm_fraction": esm_records / records if records else 0.0,
        "probe_heads": 2,
        "device": str(device),
        "threshold": parameters.threshold,
        "comparison": "strict_score_greater_than_threshold",
        "classes": ["Eukaryota", "Other"],
        "inputs": {
            "fasta": {"path": str(fasta_path), "sha256": sha256_file(fasta_path)},
            "config": (
                {"path": str(config), "sha256": sha256_file(config)}
                if config
                else {"path": "bundled:model.json"}
            ),
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gates": {
            "all_eligible_fasta_records_scored_once": True,
            "records_below_minimum_length_omitted": True,
            "default_minimum_length_bp": 1_000,
            "no_eligible_records_is_success": True,
            "whole_contigs_scored_without_chopping": True,
            "dna_then_esm_on_one_gpu": True,
            "one_esm_forward_shared_by_all_probe_heads": True,
            "strict_score_greater_than_threshold": True,
            "unknown_class": False,
            "model_artifacts_hash_bound": True,
            "dna_other_early_exit_is_below_deployment_threshold": (
                parameters.early_exit_other_max_score < parameters.threshold
            ),
            "final_test_rows_read": 0,
        },
    }
    summary_part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(summary_part, summary_path)
    return payload


__all__ = ["DeploymentParameters", "load_deployment_parameters", "predict_fasta"]
