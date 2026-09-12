"""Preserve applicable v0.54 production inference contracts for NTv3 releases."""
import hashlib
import json
from importlib.resources import as_file, files
import numpy as np
import pytest
import torch
from eukcontigminer import DEPLOYMENT_THRESHOLD, MODEL_ID, __version__
from eukcontigminer.contracts import classify_score
from eukcontigminer.deployment import _apply_tree_routing, _load_fusion_tree, _cuda_memory_summary, _load_probe, _resolve_device, load_deployment_parameters
from eukcontigminer.esm_inference import ESM2ORFInferenceConfig, select_orfs_from_contig

def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def test_release_inference_is_explicitly_reference_free():
    contract = load_deployment_parameters().config['inference_contract']
    assert contract == {'reference_free': True, 'reference_database': None, 'external_similarity_search': False, 'runtime_inputs': ['contig_sequence', 'DNA_model_weights', 'ESM-C_300M_weights', 'learned_head_weights', 'fusion_tree_weights', 'NTv3_100M_weights', 'NT_adapter_weights']}

def test_cpu_device_is_supported_without_cuda_calls():
    device = _resolve_device('cpu')
    assert str(device) == 'cpu'
    assert _cuda_memory_summary(device) == (0, 0)

def test_esmc_probe_and_feature_definition_are_exact():
    parameters = load_deployment_parameters()
    model = parameters.config['model']
    probe, mean, standard_deviation = _load_probe(model['probe'], name='ESM-C probe', expected_schema='eukcontigminer.esmc_probe_full.v1', device=torch.device('cpu'), feature_definition=model['feature_definition'], esm_sha256=model['esmc']['sha256'], feature_dimension=1920)
    assert probe.network[0].in_features == 1920
    assert tuple(mean.shape) == (1920,)
    assert tuple(standard_deviation.shape) == (1920,)

def test_tree_routing_is_model_semantics_not_legacy_parity_claim():
    p = load_deployment_parameters()
    assert p.early_exit_parity_validated is False
    route = p.config['model']['dna_other_early_exit']
    assert route['status'] == 'frozen_model_routing'
    assert route['full_esm_preserves_routing'] is True
    cutoff = p.early_exit_other_max_score
    dna = np.array([0.0, cutoff, np.nextafter(cutoff, 1), 0.8, 0.9])
    lengths = np.array([1000, 1500, 2000, 1, 2])
    scores = _apply_tree_routing(np.full(5, 0.999), dna, lengths, cutoff)
    np.testing.assert_array_equal(scores, [0.0, cutoff, 0.999, 0.8, 0.9])

def test_corrupted_tree_hash_is_rejected():
    p = load_deployment_parameters()
    p.config['model']['fusion_tree']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='fusion tree artifact differs'):
        _load_fusion_tree(p, torch.device('cpu'), 1)

@pytest.mark.parametrize('mutation', ['feature_order', 'route', 'schema'])
def test_tree_contract_changes_are_rejected(tmp_path, mutation):
    config = load_deployment_parameters().config
    if mutation == 'feature_order':
        config['model']['fusion_tree']['feature_order'].reverse()
    elif mutation == 'route':
        config['model']['dna_other_early_exit']['full_esm_preserves_routing'] = False
    else:
        config['schema'] = 'eukcontigminer.release_model.v3'
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_deployment_parameters(path)

def test_cpu_tree_loads_frozen_architecture():
    tree = _load_fusion_tree(load_deployment_parameters(), torch.device('cpu'), 1)
    assert tree.num_features() == 1923 and tree.num_boosted_rounds() == 400

def test_piecewise_fusion_and_reverse_complement_contract_are_frozen():
    parameters = load_deployment_parameters()
    assert parameters.probe_center == 1.5310083413159512
    assert parameters.probe_scale == 8.262124854196705
    assert parameters.short_alpha == 6.300000000000011
    assert parameters.long_alpha == 10.949999999999985
    assert parameters.piecewise_boundary_bp == 2000
    sequence = 'ATGGCC' * 200 + 'TAACCC' * 20
    reverse = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    config = ESM2ORFInferenceConfig()
    assert select_orfs_from_contig(sequence, config) == select_orfs_from_contig(reverse, config)
