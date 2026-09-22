"""Installed v0.61 packaging and routing regression tests."""
import ast
import hashlib
from importlib.resources import files
import numpy as np
import pytest
from eukcontigminer import __version__
from eukcontigminer.deployment import load_deployment_parameters, _bound_path
from eukcontigminer._staged_exit import FrozenGates
from eukcontigminer._certified_exit import BoundRouter

def test_bundled_gates_bounds_and_finite_all_length_bounds():
    assert __version__ == '0.61'
    p=load_deployment_parameters(); opt=p.config['runtime_optimization']
    gate=files('eukcontigminer').joinpath('_staged_gates.json')
    assert hashlib.sha256(gate.read_bytes()).hexdigest()==opt['empirical_gates']['sha256']
    b=BoundRouter(_bound_path(opt['conditional_bounds'],'bounds'),_bound_path(p.config['model']['fusion_tree'],'tree'),p.config)
    lengths=np.array([1,2,3,999,1000,1501,1999,2000,9999,10000,50000,100000,100001,200000])
    for z in [-100.,0.,100.]:
        lo=b.lower_probability(np.full(len(lengths),z),np.full(len(lengths),z),lengths)
        hi=b.upper_probability(np.full(len(lengths),z),np.full(len(lengths),z),lengths)
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert ((0<=lo)&(lo<=hi)&(hi<=1)).all()

@pytest.mark.parametrize('branch,side',[('dna','euk_above'),('nt','other_below'),('nt','euk_above')])
def test_empirical_gate_boundaries_and_scope(branch,side):
    g=FrozenGates(files('eukcontigminer').joinpath('_staged_gates.json'))
    lengths=np.array([1000,1999,2000,9999,10000,100000])
    bins=np.digitize(lengths,[2000,10000]); cuts=np.array([g.cuts[branch][str(i)][side] for i in bins])
    assert not g.apply(cuts,lengths,branch,side).any()
    direction=np.inf if side=='euk_above' else -np.inf
    assert g.apply(np.nextafter(cuts,direction),lengths,branch,side).all()
    assert not g.apply(np.full(4,1000 if side=='euk_above' else -1000),np.array([1,999,100001,200000]),branch,side).any()
    assert not g.apply(np.array([np.nan,np.inf,-np.inf]),np.array([1000]*3),branch,side).any()

def test_no_whole_buffer_retry_and_frozen_runtime_contract():
    p=load_deployment_parameters();r=p.config['model']['dna_other_early_exit']
    assert r['fallback']=='disabled' and p.config['runtime_optimization']['whole_buffer_fallback'] is False
    text=files('eukcontigminer').joinpath('deployment.py').read_text();tree=ast.parse(text)
    assert 'for guarded_attempt' not in text
    assert not any(isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id.startswith('guarded_fallback') for n in ast.walk(tree))
