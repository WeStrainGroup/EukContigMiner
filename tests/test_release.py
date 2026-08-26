import hashlib
from importlib.resources import as_file, files

from eukcontigminer import DEPLOYMENT_THRESHOLD, MODEL_ID, __version__
from eukcontigminer.contracts import classify_score
from eukcontigminer.deployment import load_deployment_parameters
from eukcontigminer.esm_inference import ESM2ORFInferenceConfig, select_orfs_from_contig


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundled_release_config_and_strict_threshold():
    parameters = load_deployment_parameters()
    assert __version__ == "0.2.0"
    assert parameters.model_id == MODEL_ID
    assert parameters.threshold == DEPLOYMENT_THRESHOLD
    assert classify_score(DEPLOYMENT_THRESHOLD, DEPLOYMENT_THRESHOLD) == "Other"


def test_small_model_assets_are_hash_bound():
    parameters = load_deployment_parameters()
    bindings = [
        *parameters.config["model"]["dna"]["heads"],
        parameters.config["model"]["probe"],
    ]
    for binding in bindings:
        with as_file(
            files("eukcontigminer.model_data").joinpath(binding["asset"])
        ) as path:
            assert _sha256(path) == binding["sha256"]


def test_orf_selection_is_reverse_complement_invariant():
    sequence = "ATGGCC" * 200 + "TAACCC" * 20
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    config = ESM2ORFInferenceConfig()
    assert select_orfs_from_contig(sequence, config) == select_orfs_from_contig(
        reverse, config
    )
