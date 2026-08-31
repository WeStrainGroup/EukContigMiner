import hashlib
import json
from importlib.resources import as_file, files

import torch

from eukcontigminer import DEPLOYMENT_THRESHOLD, MODEL_ID, __version__
from eukcontigminer.contracts import classify_score
from eukcontigminer.deployment import (
    _cuda_memory_summary,
    _load_probe,
    _resolve_device,
    load_deployment_parameters,
)
from eukcontigminer.esm_inference import ESM2ORFInferenceConfig, select_orfs_from_contig


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_release_identity_and_strict_threshold():
    parameters = load_deployment_parameters()
    assert __version__ == "0.50"
    assert parameters.model_id == MODEL_ID == "esmc_300m_true_replacement_piecewise_v1"
    assert parameters.protein_family == "esmc"
    assert parameters.threshold == DEPLOYMENT_THRESHOLD == 0.9822634639258321
    assert classify_score(DEPLOYMENT_THRESHOLD, DEPLOYMENT_THRESHOLD) == "Other"
    assert parameters.config["binary_target"]["unknown_class"] is False
    assert parameters.config["final_test"]["rows_read"] == 0


def test_release_is_the_promoted_esmc_candidate():
    benchmark = load_deployment_parameters().config["benchmark"]
    assert benchmark["promotion_status"] == "iteration_promoted"
    assert benchmark["all_formal_guardrails_passed"] is True
    assert benchmark["continuous_full_delta_vs_reference"] > 0.0
    assert benchmark["formal_full_delta_vs_reference"] > 0.0
    assert benchmark["final_test_used_for_model_selection"] is False


def test_cpu_device_is_supported_without_cuda_calls():
    device = _resolve_device("cpu")
    assert str(device) == "cpu"
    assert _cuda_memory_summary(device) == (0, 0)


def test_all_bundled_assets_are_hash_bound():
    parameters = load_deployment_parameters()
    model = parameters.config["model"]
    bindings = [
        *model["dna"]["heads"],
        model["probe"],
        model["dna_other_early_exit"]["independent_gate"],
    ]
    for binding in bindings:
        with as_file(
            files("eukcontigminer.model_data").joinpath(binding["asset"])
        ) as path:
            assert _sha256(path) == binding["sha256"]


def test_esmc_probe_and_feature_definition_are_exact():
    parameters = load_deployment_parameters()
    model = parameters.config["model"]
    probe, mean, standard_deviation = _load_probe(
        model["probe"],
        name="ESM-C probe",
        expected_schema="eukcontigminer.esmc_probe_full.v1",
        device=torch.device("cpu"),
        feature_definition=model["feature_definition"],
        esm_sha256=model["esmc"]["sha256"],
        feature_dimension=1920,
    )
    assert probe.network[0].in_features == 1920
    assert tuple(mean.shape) == (1920,)
    assert tuple(standard_deviation.shape) == (1920,)


def test_early_exit_is_independently_label_parity_validated():
    parameters = load_deployment_parameters()
    assert parameters.early_exit_parity_validated is True
    binding = parameters.config["model"]["dna_other_early_exit"]["independent_gate"]
    with as_file(
        files("eukcontigminer.model_data").joinpath(binding["asset"])
    ) as path:
        gate = json.loads(path.read_text())
    assert gate["gates"]["continuous_zero_deployment_label_changes"] is True
    assert gate["gates"]["formal_zero_deployment_label_changes"] is True
    assert gate["gates"]["zero_organelle_label_changes_both_panels"] is True
    assert gate["gates"]["final_test_rows_read"] == 0


def test_piecewise_fusion_and_reverse_complement_contract_are_frozen():
    parameters = load_deployment_parameters()
    assert parameters.probe_center == 1.5310083413159512
    assert parameters.probe_scale == 8.262124854196705
    assert parameters.short_alpha == 6.300000000000011
    assert parameters.long_alpha == 10.949999999999985
    assert parameters.piecewise_boundary_bp == 2000

    sequence = "ATGGCC" * 200 + "TAACCC" * 20
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    config = ESM2ORFInferenceConfig()
    assert select_orfs_from_contig(sequence, config) == select_orfs_from_contig(
        reverse, config
    )
