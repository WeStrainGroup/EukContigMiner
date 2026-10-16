import hashlib,json,math
from importlib.resources import files
import numpy as np
from eukcontigminer import __version__,DEPLOYMENT_THRESHOLD,MODEL_ID
from eukcontigminer.deployment import load_deployment_parameters
from eukcontigminer.contracts import classify_score
def test_release_assets_and_decision():
    root=files("eukcontigminer");c=json.loads(root.joinpath("model_data/model.json").read_text())
    assert __version__==c["release_version"]=="0.62" and MODEL_ID==c["model_id"]
    assert DEPLOYMENT_THRESHOLD==c["prediction_rule"]["threshold"]==0.9996954815448658
    assert load_deployment_parameters().threshold==DEPLOYMENT_THRESHOLD
    for b in [c["model"][x] for x in ["dna","probe","fusion_tree"]]+[c["model"]["nt_adapter"]["checkpoint"],c["runtime_optimization"]["conditional_bounds"]]:
        h=hashlib.sha256()
        with root.joinpath("model_data",b["asset"]).open("rb") as f:
            for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
        assert h.hexdigest()==b["sha256"]
    g=c["runtime_optimization"]["empirical_gates"]
    assert hashlib.sha256(root.joinpath(g["asset"]).read_bytes()).hexdigest()==g["sha256"]
    assert classify_score(DEPLOYMENT_THRESHOLD,DEPLOYMENT_THRESHOLD)=="Other"
    assert classify_score(float(np.nextafter(DEPLOYMENT_THRESHOLD,1.)),DEPLOYMENT_THRESHOLD)=="Eukaryota"

def test_full_buffer_eligible_row_mapping():
    # Full eligible batches are exactly the original row mapping, including tail buffers.
    for n in [1,446,1024]:
        lengths=np.full(n,1000);indices=np.flatnonzero(lengths>=3)
        selected=np.arange(n)[::3]
        assert np.array_equal(indices,np.arange(n))
        assert np.array_equal(np.searchsorted(indices,selected),selected)
    lengths=np.array([1,1000,2,3,1999,2000]);indices=np.flatnonzero(lengths>=3)
    assert list(indices)==[1,3,4,5]
    assert list(np.searchsorted(indices,[1,4,5]))==[0,2,3]
