import json
import numpy as np
import pytest
from importlib.resources import files,as_file
from eukcontigminer import __version__,MODEL_ID,DEPLOYMENT_THRESHOLD
from eukcontigminer.deployment import load_deployment_parameters,sha256_file
from eukcontigminer.contracts import classify_score
from eukcontigminer.length_calibration import apply_length_calibration

def test_release_candidate_identity_and_assets():
    p=load_deployment_parameters();assert __version__=='0.55' and p.model_id==MODEL_ID
    assert p.threshold==DEPLOYMENT_THRESHOLD==0.9994805844224834 and classify_score(.5,.5)=='Other'
    assert p.config['status']=='validation_only_final_test_unchanged'
    assert not p.config['scientific_validation']['goal_achieved']
    m=p.config['model'];assert 'length_calibration' not in m and m['nt_adapter']['checkpoint_step']==32768;assert m['nt_adapter']['trainable_parameters']==106654897
    for b in [*m['dna']['heads'],m['probe'],m['fusion_tree'],m['nt_adapter']['checkpoint']]:
        with as_file(files('eukcontigminer.model_data').joinpath(b['asset'])) as q:assert sha256_file(q)==b['sha256']

def test_length_score_preserves_strict_ties_and_endpoints():
    c=dict(schema='ecm.log_length_decision.v1',knots_bp=[1000,2000,5000,10000,100000],coefficients=[0.]*5)
    raw=np.array([0.,np.nextafter(.5,0.),.5,np.nextafter(.5,1.),1.]);length=np.array([1,999,1501,100001,200000])
    out=apply_length_calibration(raw,length,c)
    assert np.array_equal(out>.5,raw>.5) and out[2]==.5 and np.isfinite(out).all()
    c['coefficients']=[float('nan')]*5
    with pytest.raises(ValueError):apply_length_calibration(raw,length,c)
