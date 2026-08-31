"""Hash-bound CPU or one-GPU inference for the released DNA + ESM-C model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from dataclasses import dataclass, replace
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
    _esmc_hardware_policy,
    esm2_features_from_orfs,
    esmc_features_from_orfs,
    optimize_esm2_feature_inference,
    optimize_esm2_sdpa_feature_inference,
    optimize_esmc_feature_inference,
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
    protein_family: str
    early_exit_parity_validated: bool
    threshold: float
    early_exit_other_max_score: float
    positive_alpha: float
    negative_alpha: float
    probe_center: float
    probe_scale: float
    secondary_probe_center: float | None
    secondary_probe_scale: float | None
    secondary_source_alpha: float | None
    short_alpha: float | None
    long_alpha: float | None
    piecewise_boundary_bp: int | None
    config: dict[str, Any]


SINGLE_PROBE_FORMULA = (
    "sigmoid(base_logit + positive_alpha * max(probe_z, 0) + "
    "negative_alpha * min(probe_z, 0))"
)
DUAL_PROBE_FORMULA = (
    "sigmoid(reference_logit + alpha(length) * secondary_source_alpha * "
    "secondary_probe_z)"
)
ESMC_PIECEWISE_FORMULA = (
    "sigmoid(reference_logit + alpha(length) * protein_probe_z)"
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


def _resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto`` and fail clearly for unavailable accelerator requests."""

    if device_name == "auto":
        return torch.device(
            "cuda:0"
            if torch.cuda.is_available() and torch.cuda.device_count()
            else "cpu"
        )
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid inference device: {device_name}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("inference device must be auto, cpu, or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device_name!r} was requested but CUDA is unavailable"
            )
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)"
            )
        return torch.device("cuda", index)
    return torch.device("cpu")


def _cuda_memory_summary(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _bound_path(binding: object, name: str) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{name} binding is missing")
    asset = binding.get("asset")
    if asset is not None:
        asset_name = str(asset)
        if not asset_name or Path(asset_name).name != asset_name:
            raise ValueError(f"{name} asset name is invalid")
        path = Path(str(files("eukcontigminer.model_data").joinpath(asset_name)))
    else:
        path = Path(str(binding.get("path", "")))
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
    schema = config.get("schema")
    dual_probe = schema == "eukcontigminer.validation_candidate.v3"
    esmc_piecewise = schema in {
        "eukcontigminer.validation_candidate.v4",
        "eukcontigminer.release_model.v3",
    }
    protein_family = "esmc" if esmc_piecewise else "esm2"
    encoder = (
        model.get("esmc") if esmc_piecewise else model.get("esm2")
    ) if isinstance(model, dict) else None
    expected_formula = (
        ESMC_PIECEWISE_FORMULA
        if esmc_piecewise
        else DUAL_PROBE_FORMULA if dual_probe else SINGLE_PROBE_FORMULA
    )
    definition = model.get("feature_definition", {}) if isinstance(model, dict) else {}
    selection = (
        definition.get("selection")
        if esmc_piecewise and isinstance(definition, dict)
        else definition
    )
    expected_feature_dimension = 1920 if esmc_piecewise else 2560
    if (
        schema not in {
            "eukcontigminer.validation_candidate.v2",
            "eukcontigminer.validation_candidate.v3",
            "eukcontigminer.validation_candidate.v4",
            "eukcontigminer.release_model.v3",
        }
        or config.get("status")
        not in {"validation_only_final_test_unchanged", "released"}
        or not isinstance(prediction, dict)
        or prediction.get("comparison") != "strict_greater_than"
        or prediction.get("equal_threshold_label") != "Other"
        or not isinstance(model, dict)
        or not isinstance(dna, dict)
        or not isinstance(encoder, dict)
        or model.get("formula") != expected_formula
        or dna.get("ensemble_weights") != list(ENSEMBLE_WEIGHTS)
        or dna.get("unfreeze_scope") != "heads-only"
        or len(dna.get("heads", [])) != 2
        or not isinstance(selection, dict)
        or selection.get("reverse_complement_invariant") is not True
        or (
            selection.get("maximum_orfs")
            if esmc_piecewise
            else selection.get("orfs_per_contig")
        ) != 2
        or definition.get("feature_dimension") != expected_feature_dimension
        or config.get("binary_target", {}).get("unknown_class") is not False
        or config.get("final_test", {}).get("read_or_changed_for_this_candidate")
        is not False
        or (
            dual_probe
            and (
                not isinstance(model.get("secondary_probe"), dict)
                or model.get("piecewise_secondary_fusion", {}).get("comparison")
                != "length_less_than_or_equal"
            )
        )
        or (
            esmc_piecewise
            and (
                encoder.get("name") != "esmc_300m"
                or encoder.get("release_package") != "esm 3.2.1"
                or encoder.get("use_flash_attention") is not False
                or encoder.get("layers") != 30
                or encoder.get("embedding_dimension") != 960
                or not isinstance(model.get("probe"), dict)
                or model.get("piecewise_protein_fusion", {}).get("comparison")
                != "length_less_than_or_equal"
            )
        )
    ):
        raise ValueError("deployment config violates the frozen model contract")
    values = {
        "threshold": float(prediction.get("threshold", math.nan)),
        "early_exit_other_max_score": float(
            model.get("dna_other_early_exit", {}).get(
                "maximum_dna_p_euk", math.nan
            )
        ),
        "positive_alpha": float(
            model.get("positive_alpha", 0.0 if esmc_piecewise else math.nan)
        ),
        "negative_alpha": float(
            model.get("negative_alpha", 0.0 if esmc_piecewise else math.nan)
        ),
        "probe_center": float(model.get("probe_logit_center", math.nan)),
        "probe_scale": float(model.get("probe_logit_scale", math.nan)),
    }
    secondary_values: dict[str, float | int | None] = {
        "secondary_probe_center": None,
        "secondary_probe_scale": None,
        "secondary_source_alpha": None,
        "short_alpha": None,
        "long_alpha": None,
        "piecewise_boundary_bp": None,
    }
    if dual_probe or esmc_piecewise:
        piecewise = model[
            "piecewise_protein_fusion"
            if esmc_piecewise
            else "piecewise_secondary_fusion"
        ]
        secondary_values = {
            "secondary_probe_center": float(
                model.get(
                    "probe_logit_center"
                    if esmc_piecewise
                    else "secondary_probe_logit_center",
                    math.nan,
                )
            ),
            "secondary_probe_scale": float(
                model.get(
                    "probe_logit_scale"
                    if esmc_piecewise
                    else "secondary_probe_logit_scale",
                    math.nan,
                )
            ),
            "secondary_source_alpha": float(
                1.0
                if esmc_piecewise
                else model.get("secondary_source_alpha", math.nan)
            ),
            "short_alpha": float(piecewise.get("short_alpha", math.nan)),
            "long_alpha": float(piecewise.get("long_alpha", math.nan)),
            "piecewise_boundary_bp": int(piecewise.get("boundary_bp", -1)),
        }
    if (
        not all(math.isfinite(value) for value in values.values())
        or not 0.0 <= values["threshold"] <= 1.0
        or not 0.0
        <= values["early_exit_other_max_score"]
        < values["threshold"]
        or min(values["positive_alpha"], values["negative_alpha"]) < 0.0
        or values["probe_scale"] <= 0.0
        or (
            (dual_probe or esmc_piecewise)
            and (
                not all(
                    math.isfinite(float(secondary_values[name]))
                    and float(secondary_values[name]) > 0.0
                    for name in (
                        "secondary_probe_scale",
                        "secondary_source_alpha",
                        "short_alpha",
                        "long_alpha",
                    )
                )
                or not math.isfinite(
                    float(secondary_values["secondary_probe_center"])
                )
                or int(secondary_values["piecewise_boundary_bp"]) < 1_000
            )
        )
    ):
        raise ValueError("deployment probabilities or fusion parameters are invalid")
    early_exit_parity_validated = protein_family == "esm2"
    if esmc_piecewise and values["early_exit_other_max_score"] > 0.0:
        early_exit = model.get("dna_other_early_exit", {})
        gate_path = _bound_path(
            early_exit.get("independent_gate"), "ESM-C early-exit gate"
        )
        gate = json.loads(gate_path.read_text())
        gate_values = gate.get("gates", {})
        if (
            early_exit.get("status") != "accepted_for_runtime_canary"
            or gate.get("schema")
            != "eukcontigminer.global_early_exit_gate.v1"
            or gate.get("status") != "accepted_for_runtime_canary"
            or gate.get("candidate_cutoff")
            != values["early_exit_other_max_score"]
            or gate.get("deployment_threshold") != values["threshold"]
            or not isinstance(gate_values, dict)
            or gate_values.get("continuous_zero_deployment_label_changes")
            is not True
            or gate_values.get("formal_zero_deployment_label_changes") is not True
            or gate_values.get("zero_organelle_label_changes_both_panels")
            is not True
            or gate_values.get("all_reported_prevalence_f1_values_unchanged")
            is not True
            or gate_values.get("runtime_canary_eligible") is not True
            or gate_values.get("final_test_rows_read") != 0
        ):
            raise ValueError("ESM-C early-exit gate differs or is ineligible")
        early_exit_parity_validated = True
    return DeploymentParameters(
        model_id=str(config["model_id"]),
        protein_family=protein_family,
        early_exit_parity_validated=early_exit_parity_validated,
        config=config,
        **secondary_values,
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
    """Encode a FASTA buffer once while retaining strings for ORF inference."""
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


def _load_probe(
    binding: object,
    *,
    name: str,
    expected_schema: str,
    device: torch.device,
    feature_definition: dict[str, Any] | None = None,
    esm_sha256: str | None = None,
    feature_dimension: int = 2560,
) -> tuple[ESM2Probe, torch.Tensor, torch.Tensor]:
    checkpoint = _bound_path(binding, name)
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
        requested_selection = feature_definition.get("selection")
        exact_nested_definition = isinstance(requested_selection, dict)
        if exact_nested_definition:
            if not isinstance(checkpoint_model, dict) or not isinstance(
                feature_definition.get("model"), dict
            ):
                definition_matches = False
            else:
                checkpoint_comparable = dict(checkpoint_definition)
                requested_comparable = dict(feature_definition)
                checkpoint_model_comparable = dict(checkpoint_model)
                requested_model_comparable = dict(requested_comparable["model"])
                # The training checkpoint records its historical local path. A
                # release binds the immutable backbone by hash and official
                # identity, so only that non-portable path is excluded.
                checkpoint_model_comparable.pop("checkpoint", None)
                requested_model_comparable.pop("checkpoint", None)
                checkpoint_comparable["model"] = checkpoint_model_comparable
                requested_comparable["model"] = requested_model_comparable
                definition_matches = checkpoint_comparable == requested_comparable
        else:
            definition_matches = (
                isinstance(selection, dict)
                and isinstance(checkpoint_model, dict)
                and selection.get("maximum_orfs")
                == feature_definition.get("orfs_per_contig")
                and selection.get("minimum_orf_length")
                == feature_definition.get("minimum_orf_aa")
                and selection.get("maximum_orf_length")
                == feature_definition.get("maximum_orf_aa")
                and selection.get("reverse_complement_invariant") is True
            )
        if (
            not definition_matches
            or not isinstance(checkpoint_definition, dict)
            or not isinstance(checkpoint_model, dict)
            or checkpoint_definition.get("feature_dimension")
            != feature_dimension
            or checkpoint_model.get("checkpoint_sha256") != esm_sha256
        ):
            raise ValueError(f"{name} feature definition differs")
    model_config = payload.get("model_config")
    if (
        not isinstance(model_config, dict)
        or int(model_config.get("feature_dimension", -1)) != feature_dimension
    ):
        raise ValueError(f"{name} model configuration differs")
    probe = ESM2Probe(**model_config)
    probe.load_state_dict(payload["model_state_dict"], strict=True)
    probe.requires_grad_(False).eval().to(device)
    feature_mean = payload["feature_mean"].float().cpu()
    feature_std = payload["feature_std"].float().cpu()
    if (
        feature_mean.shape != (feature_dimension,)
        or feature_std.shape != (feature_dimension,)
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
    ESM2Probe | None,
    torch.Tensor | None,
    torch.Tensor | None,
    ESM2ORFInferenceConfig,
]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("frozen DNA+ESM prediction supports only CPU or CUDA")
    model = parameters.config["model"]
    dna = model["dna"]
    from eukcontigminer._model.motif_v20 import motif_length_gate
    from eukcontigminer._model.sequence import reverse_complement_batch
    from eukcontigminer.predictor import _TOKEN, WEIGHTS_SHA256, Predictor

    if WEIGHTS_SHA256 != dna.get("base_weights_sha256"):
        raise ValueError("public DNA base weights differ")
    head_rows = dna["heads"]
    head_paths = [
        _bound_path(row, f"DNA head {index}")
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

    definition = model["feature_definition"]
    secondary_probe = secondary_mean = secondary_std = None
    if parameters.protein_family == "esmc":
        esm_binding = model["esmc"]
        try:
            esm_version = importlib.metadata.version("esm")
        except importlib.metadata.PackageNotFoundError:
            import esm

            esm_version = getattr(esm, "__version__", None)
        if esm_version != "3.2.1":
            raise ValueError("ESM-C deployment requires the frozen esm 3.2.1 package")
        from esm.models.esmc import ESMC
        from esm.pretrained import get_esmc_model_tokenizers
        from esm.utils.constants.esm3 import data_root

        weight_path = (
            data_root(str(esm_binding["data_root_key"]))
            / str(esm_binding["weight_relative_path"])
        )
        if (
            not weight_path.is_file()
            or weight_path.stat().st_size != int(esm_binding["checkpoint_bytes"])
            or sha256_file(weight_path) != str(esm_binding["sha256"])
        ):
            raise ValueError("ESM-C 300M backbone differs")
        with torch.device(device):
            esm_model = ESMC(
                d_model=960,
                n_heads=15,
                n_layers=30,
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            ).eval()
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        esm_model.load_state_dict(state_dict, strict=True)
        esm_model.requires_grad_(False).eval().to(device)
        if (
            len(esm_model.transformer.blocks) != 30
            or int(esm_model.embed.embedding_dim) != 960
            or esm_model._use_flash_attn is not False
            or optimize_esmc_feature_inference(esm_model) != 1
        ):
            raise ValueError("loaded ESM-C architecture differs")
        probe, feature_mean, feature_std = _load_probe(
            model["probe"],
            name="ESM-C probe",
            expected_schema="eukcontigminer.esmc_probe_full.v1",
            device=device,
            feature_definition=definition,
            esm_sha256=str(esm_binding.get("sha256", "")),
            feature_dimension=1920,
        )
        batch_converter = esm_model._tokenize
        selection = definition["selection"]
        orf_config = ESM2ORFInferenceConfig(
            maximum_orfs=int(selection["maximum_orfs"]),
            minimum_orf_length=int(selection["minimum_orf_length"]),
            maximum_orf_length=int(selection["maximum_orf_length"]),
            aggregation=str(selection["aggregation"]),
        )
    else:
        esm_binding = model["esm2"]
        if not isinstance(esm_binding, dict):
            raise ValueError("ESM-2 binding is missing")
        esm_checkpoint = Path(str(esm_binding.get("path", "")))
        import esm

        # Hash and load the immutable 2.5 GB checkpoint concurrently.
        with ThreadPoolExecutor(max_workers=1) as executor:
            verified_esm = executor.submit(_bound_path, esm_binding, "ESM-2")
            with torch.serialization.safe_globals([argparse.Namespace]):
                esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(
                    esm_checkpoint
                )
            if verified_esm.result() != esm_checkpoint:
                raise ValueError("ESM-2 artifact binding path differs")
        if not isinstance(getattr(esm_model, "lm_head", None), nn.Module):
            raise ValueError("ESM-2 language-model head is missing")
        esm_model.lm_head = nn.Identity()
        optimize_esm2_feature_inference(esm_model)
        esm_model.requires_grad_(False).eval().to(device)
        probe, feature_mean, feature_std = _load_probe(
            model["probe"],
            name="primary ESM probe",
            expected_schema="eukcontigminer.esm2_probe.v1",
            device=device,
        )
        if parameters.secondary_probe_center is not None:
            secondary_probe, secondary_mean, secondary_std = _load_probe(
                model["secondary_probe"],
                name="secondary ESM probe",
                expected_schema="eukcontigminer.esm2_probe_full.v1",
                device=device,
                feature_definition=definition,
                esm_sha256=str(esm_binding.get("sha256", "")),
            )
        batch_converter = alphabet.get_batch_converter(truncation_seq_length=None)
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
        batch_converter,
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
    device_name: str = "auto",
    cpu_threads: int | None = None,
    buffer_records: int = 4_096,
    dna_batch_size: int = 32,
    dna_max_padded_bases: int = 800_000,
    esm_token_budget: int = 16_384,
    esm_attention_budget: int = 2_000_000,
    use_dna_early_exit: bool = True,
    minimum_length: int = 1_000,
    use_esm_sdpa: bool = False,
) -> dict[str, Any]:
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be positive when specified")
    if min(
        buffer_records,
        dna_batch_size,
        dna_max_padded_bases,
        esm_token_budget,
        esm_attention_budget,
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
    device = _resolve_device(device_name)
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    effective_cpu_threads = int(torch.get_num_threads())
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
    orf_config = replace(
        orf_config,
        token_budget=esm_token_budget,
        attention_budget=esm_attention_budget,
    )
    orf_config.validate()
    if use_esm_sdpa and parameters.protein_family != "esm2":
        raise ValueError("--esm-sdpa is available only for the ESM-2 encoder")
    sdpa_layers = (
        optimize_esm2_sdpa_feature_inference(esm_model)
        if use_esm_sdpa
        else 0
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
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
                secondary_probe_logit = (
                    np.full(
                        len(buffer),
                        parameters.secondary_probe_center,
                        dtype=np.float64,
                    )
                    if parameters.secondary_probe_center is not None
                    else None
                )
                if len(protein_indices):
                    orfs = [
                        select_orfs_from_contig(sequences[int(index)], orf_config)
                        for index in protein_indices
                    ]
                    if parameters.protein_family == "esmc":
                        features = esmc_features_from_orfs(
                            esm_model,
                            batch_converter,
                            orfs,
                            device=device,
                            config=orf_config,
                        )
                    else:
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
                    if secondary_probe is not None:
                        if secondary_mean is None or secondary_std is None:
                            raise RuntimeError(
                                "secondary probe standardization is missing"
                            )
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
                        assert secondary_probe_logit is not None
                        secondary_probe_logit[
                            protein_indices
                        ] = selected_secondary_logit
                if parameters.protein_family == "esmc":
                    if any(
                        value is None
                        for value in (
                            parameters.secondary_probe_center,
                            parameters.secondary_probe_scale,
                            parameters.short_alpha,
                            parameters.long_alpha,
                            parameters.piecewise_boundary_bp,
                        )
                    ):
                        raise RuntimeError(
                            "ESM-C piecewise fusion parameters are missing"
                        )
                    scores = dual_probe_piecewise_probability(
                        dna_scores,
                        probe_logit,
                        lengths,
                        secondary_probe_center=float(parameters.probe_center),
                        secondary_probe_scale=float(parameters.probe_scale),
                        secondary_source_alpha=1.0,
                        short_alpha=float(parameters.short_alpha),
                        long_alpha=float(parameters.long_alpha),
                        boundary_bp=int(parameters.piecewise_boundary_bp),
                    )
                else:
                    scores = direct_joint_probability(
                        dna_scores,
                        probe_logit,
                        probe_center=parameters.probe_center,
                        probe_scale=parameters.probe_scale,
                        positive_alpha=parameters.positive_alpha,
                        negative_alpha=parameters.negative_alpha,
                    )
                if (
                    parameters.protein_family == "esm2"
                    and secondary_probe_logit is not None
                ):
                    if any(
                        value is None
                        for value in (
                            parameters.secondary_probe_center,
                            parameters.secondary_probe_scale,
                            parameters.secondary_source_alpha,
                            parameters.short_alpha,
                            parameters.long_alpha,
                            parameters.piecewise_boundary_bp,
                        )
                    ):
                        raise RuntimeError("dual-probe fusion parameters are missing")
                    scores = dual_probe_piecewise_probability(
                        scores,
                        secondary_probe_logit,
                        lengths,
                        secondary_probe_center=float(
                            parameters.secondary_probe_center
                        ),
                        secondary_probe_scale=float(parameters.secondary_probe_scale),
                        secondary_source_alpha=float(
                            parameters.secondary_source_alpha
                        ),
                        short_alpha=float(parameters.short_alpha),
                        long_alpha=float(parameters.long_alpha),
                        boundary_bp=int(parameters.piecewise_boundary_bp),
                    )
                # The Train-frozen cutoff lies strictly below the deployment
                # threshold.  Validation gates established that these easy
                # negatives never cross the final decision boundary, so their
                # calibrated DNA probability is a valid Other score and the
                # expensive ESM encoder can be skipped entirely.
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
    peak_allocated, peak_reserved = _cuda_memory_summary(device)
    esmc_hardware = (
        _esmc_hardware_policy(device)
        if parameters.protein_family == "esmc"
        else None
    )
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
        "dna_other_early_exit_max_score": (
            parameters.early_exit_other_max_score
        ),
        "dna_other_early_exit_records": early_exit_other_records,
        "dna_other_early_exit_fraction": (
            early_exit_other_records / records if records else 0.0
        ),
        "esm_records": esm_records,
        "esm_fraction": esm_records / records if records else 0.0,
        "esm_token_budget": orf_config.token_budget,
        "esm_attention_budget": orf_config.attention_budget,
        "protein_encoder_family": parameters.protein_family,
        "esm_attention_backend": (
            "esmc_released_no_flash"
            if parameters.protein_family == "esmc"
            else "pytorch_sdpa" if use_esm_sdpa else "fair_esm"
        ),
        "esm_sdpa_optimized_layers": sdpa_layers,
        "probe_heads": (
            1
            if parameters.protein_family == "esmc"
            else 2 if parameters.secondary_probe_center is not None else 1
        ),
        "device_requested": device_name,
        "device": str(device),
        "device_type": device.type,
        "cpu_threads_requested": cpu_threads,
        "cpu_threads_effective": effective_cpu_threads,
        "esm_compute_dtype": (
            str(esmc_hardware["compute_dtype"]).removeprefix("torch.")
            if esmc_hardware is not None and device.type == "cuda"
            else "float16" if device.type == "cuda" else "float32"
        ),
        "hardware_compatibility": (
            {
                "class": esmc_hardware["compatibility_class"],
                "cuda_compute_capability": (
                    list(esmc_hardware["compute_capability"])
                    if esmc_hardware["compute_capability"] is not None
                    else None
                ),
                "minimum_cuda_compute_capability": esmc_hardware[
                    "minimum_cuda_compute_capability"
                ],
                "compiled_cuda_architectures": esmc_hardware[
                    "compiled_cuda_architectures"
                ],
                "compiled_architecture_compatible": esmc_hardware[
                    "compiled_architecture_compatible"
                ],
                "required_profiles": {
                    "cpu_amd_intel": "float32",
                    "nvidia_v100_volta": "float16",
                    "nvidia_rtx_2080ti_turing": "float16",
                    "nvidia_a40_ampere": "bfloat16_or_float16_fallback",
                    "nvidia_a100_ampere": "bfloat16_or_float16_fallback",
                },
            }
            if esmc_hardware is not None
            else None
        ),
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
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "gates": {
            "all_eligible_fasta_records_scored_once": True,
            "records_below_minimum_length_omitted": True,
            "default_minimum_length_bp": 1_000,
            "no_eligible_records_is_success": True,
            "whole_contigs_scored_without_chopping": True,
            "dna_then_esm_on_one_device": True,
            "dna_then_esm_on_one_gpu": device.type == "cuda",
            "one_gpu_deployment_supported": True,
            "cpu_only_deployment_supported": True,
            "one_esm_forward_shared_by_all_probe_heads": True,
            "esm_batch_budgets_positive": True,
            "esm_sdpa_is_explicit_opt_in": True,
            "strict_score_greater_than_threshold": True,
            "unknown_class": False,
            "model_artifacts_hash_bound": True,
            "dna_other_early_exit_is_below_deployment_threshold": (
                parameters.early_exit_other_max_score < parameters.threshold
            ),
            "dna_other_early_exit_label_parity_validated": (
                parameters.early_exit_parity_validated
            ),
            "dna_other_early_exit_efficiency_only_calibration": (
                parameters.early_exit_other_max_score > 0.0
            ),
            "final_test_rows_read": 0,
        },
    }
    summary_part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(summary_part, summary_path)
    return payload


__all__ = ["DeploymentParameters", "load_deployment_parameters", "predict_fasta"]
