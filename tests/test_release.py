import hashlib
import json
from importlib.resources import as_file, files

import numpy as np
import pytest
import torch

from eukcontigminer import DEPLOYMENT_THRESHOLD, MODEL_ID, __version__
from eukcontigminer.contracts import classify_score
from eukcontigminer.deployment import (
    _apply_tree_routing,
    _load_fusion_tree,
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
    assert __version__ == "0.54"
    assert parameters.model_id == MODEL_ID == "esmc_tree_nt500m_shortcoverage_lora_qv8_epoch3_v1"
    assert parameters.protein_family == "esmc"
    assert parameters.threshold == DEPLOYMENT_THRESHOLD == 0.9983936852318344
    assert classify_score(DEPLOYMENT_THRESHOLD, DEPLOYMENT_THRESHOLD) == "Other"
    assert parameters.config["binary_target"]["unknown_class"] is False
    assert parameters.config["final_test"]["read_or_changed_for_this_candidate"] is True
    assert parameters.config["dataset_role_revision"]["confirmation_used_for_training_or_selection"] is False


def test_release_has_frozen_independent_confirmation_evidence():
    config = load_deployment_parameters().config
    assert config["status"] == "released"
    science = config["scientific_validation"]
    evidence = science["evidence"]
    contents = {}
    for key, binding in evidence.items():
        with as_file(files("eukcontigminer.model_data").joinpath(binding["asset"])) as path:
            assert _sha256(path) == binding["sha256"]
            contents[key] = json.loads(path.read_text())
    policy=contents["confirmation_policy.json"];plan=contents["confirmation_plan.json"];report=contents["confirmation_report.json"]
    assert report["plan_sha256"] == evidence["confirmation_plan.json"]["sha256"]
    assert plan["cohort_policy"]["sha256"] == evidence["confirmation_policy.json"]["sha256"]
    assert not policy["training_on_confirmation"] and not policy["model_or_threshold_selection_on_confirmation"]
    assert not report["threshold_search_performed"] and not report["model_selected_on_confirmation"]
    assert policy["candidate"] == plan["candidate"]
    assert policy["candidate"]["checkpoint"]["sha256"] == config["model"]["nt_adapter"]["checkpoint"]["sha256"]
    assert policy["candidate"]["alpha"] == config["model"]["nt_adapter"]["alpha"] == .5
    assert policy["candidate"]["threshold"] == report["thresholds"]["candidate"] == config["prediction_rule"]["threshold"]
    assert plan["species_isolation"]["overlap"] == 0
    assert report["species"] == plan["species_isolation"]["species"] == 3474
    assert report["records"] == plan["records"] == 1090186
    assert report["decision"] == "confirmation_point_gate_passed_pending_runtime"
    for key in ["primary", "canonical_duplicate_sensitivity"]:
        section=report[key];assert section["point_gate_passed"] and not section["gate_failures"]
        new=section["results"]["candidate"];old=section["results"]["v053"]
        assert new["full"]["0.01"]["f1"] > old["full"]["0.01"]["f1"]
        for scope in ["short","fungi_bacteria","continuous","continuous_short"]:
            assert new[scope]["0.01"]["f1"] >= old[scope]["0.01"]["f1"]
        assert set(map(int,section["lengths"])) == set(range(1000,10001,500))|{50000,100000}
        for row in section["lengths"].values():assert row["candidate"]["0.01"]["f1"]-row["v053"]["0.01"]["f1"]>=-.005
    lo,hi=report["paired_species_bootstrap"]["full"]["delta_f1_95ci"]
    assert 0<lo<hi and science["paired_full_delta_ci95"]==[lo,hi]
    assert report["statistically_supported_full_improvement"] and science["statistically_established_superiority"]
    assert not science["goal_achieved"] and not report["all21_point_estimates_at_least_0p99"]
    duplicate=contents["confirmation_duplicate_audit.json"]
    assert duplicate["status"]=="complete_before_scoring" and duplicate["confirmation_scores_read"]==0
    assert duplicate["records"]==report["records"] and duplicate["duplicate_rows"]==13 and duplicate["mixed_truth_pairs"]==0
    assert report["duplicate_audit"]["sha256"]==evidence["confirmation_duplicate_audit.json"]["sha256"]
    independent=contents["confirmation_independent_verification.json"]
    assert independent["status"]=="passed" and independent["report"]["sha256"]==evidence["confirmation_report.json"]["sha256"]
    assert independent["all_five_raw_score_arrays_recounted"] and independent["paired_species_bootstrap_independently_reproduced"]
    inputs=contents["confirmation_input_verification.json"]
    assert inputs["status"]=="passed" and inputs["species_isolation_overlap"]==0 and inputs["all_eight_shards_match_original_manifest"]
    metadata=contents["confirmation_metadata_resolution.json"]
    assert metadata["confirmation_scores_read"]==0 and metadata["cohort_rows_removed"]==metadata["cohort_rows_replaced"]==0
    assert report["assembly_metadata_resolution"]["sha256"]==evidence["confirmation_metadata_resolution.json"]["sha256"]
    train=contents["training_data_summary.json"]
    assert train["records"]==1633859 and train["species"]==12197 and train["validation_species_overlap"]==0


def test_release_inference_is_explicitly_reference_free():
    contract = load_deployment_parameters().config["inference_contract"]
    assert contract == {
        "reference_free": True,
        "reference_database": None,
        "external_similarity_search": False,
        "runtime_inputs": [
            "contig_sequence",
            "DNA_model_weights",
            "ESM-C_300M_weights",
            "learned_head_weights",
            "fusion_tree_weights",
            "NT500M_weights",
            "NT_adapter_weights",
        ],
    }


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
        model["fusion_tree"],
        model["nt_adapter"]["checkpoint"],
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


def test_tree_routing_is_model_semantics_not_legacy_parity_claim():
    p = load_deployment_parameters()
    assert p.early_exit_parity_validated is False
    route = p.config["model"]["dna_other_early_exit"]
    assert route["status"] == "frozen_model_routing"
    assert route["full_esm_preserves_routing"] is True
    cutoff = p.early_exit_other_max_score
    dna = np.array([0., cutoff, np.nextafter(cutoff, 1), 0.8, 0.9])
    lengths = np.array([1000, 1500, 2000, 1, 2])
    scores = _apply_tree_routing(np.full(5, .999), dna, lengths, cutoff)
    np.testing.assert_array_equal(scores, [0., cutoff, .999, .8, .9])


def test_corrupted_tree_hash_is_rejected():
    p = load_deployment_parameters()
    p.config["model"]["fusion_tree"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fusion tree artifact differs"):
        _load_fusion_tree(p, torch.device("cpu"), 1)


@pytest.mark.parametrize("mutation", ["feature_order", "route", "schema"])
def test_tree_contract_changes_are_rejected(tmp_path, mutation):
    config = load_deployment_parameters().config
    if mutation == "feature_order":
        config["model"]["fusion_tree"]["feature_order"].reverse()
    elif mutation == "route":
        config["model"]["dna_other_early_exit"]["full_esm_preserves_routing"] = False
    else:
        config["schema"] = "eukcontigminer.release_model.v3"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_deployment_parameters(path)


def test_cpu_tree_loads_frozen_architecture():
    tree = _load_fusion_tree(load_deployment_parameters(), torch.device("cpu"), 1)
    assert tree.num_features() == 1923 and tree.num_boosted_rounds() == 400


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
