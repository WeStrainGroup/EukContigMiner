"""Release-facing regression coverage for the simplified whole-contig route."""
import copy
import json
from importlib.resources import files,as_file
import numpy as np
import pytest
import torch
from eukcontigminer import __version__,MODEL_ID,DEPLOYMENT_THRESHOLD
from eukcontigminer.deployment import load_deployment_parameters,_load_fusion_tree,sha256_file,_collate
from eukcontigminer.simplified_dna import load_model,load_parameters,transform
from eukcontigminer.predictor import _TOKEN
from eukcontigminer.contracts import classify_score
from eukcontigminer.fasta import fasta_records

def test_release_assets_and_actual_simplified_architecture():
 p=load_deployment_parameters();assert __version__.startswith('0.6') and p.model_id==MODEL_ID and p.threshold==DEPLOYMENT_THRESHOLD
 m=p.config['model'];assert p.config['schema']=='eukcontigminer.release_model.v6'
 for b in [m['dna'],m['probe'],m['fusion_tree'],m['nt_adapter']['checkpoint']]:
  with as_file(files('eukcontigminer.model_data').joinpath(b['asset'])) as f:assert sha256_file(f)==b['sha256']
 model=load_model(m['dna'],'cpu');assert sum(p.numel() for p in model.parameters())==2409641
 assert len(model.encoder.blocks)==6
 assert all('motif' not in n and 'short_' not in n for n,_ in model.named_modules())
 assert not hasattr(model.encoder,'shared')
 tree=_load_fusion_tree(p,torch.device('cpu'),2);assert tree.num_features()==1923 and tree.num_boosted_rounds()==m['fusion_tree']['rounds']

def test_all_length_scores_rc_and_padding():
 torch.set_num_threads(2);p=load_deployment_parameters();model=load_model(p.config['model']['dna'],'cpu')
 seqs=['A','NN','NNN','N'*1001,'ACGT'*249+'ACG','ACGT'*250,'ATGCCC'*250,'ACGTRYSWKMBDHVN']
 rc=str.maketrans('ACGTRYMKBDHVSWN','TGCAYRKMVHDBSWN')
 t,l=_collate(seqs,np.arange(len(seqs)),_TOKEN);u,ul=_collate([s.translate(rc)[::-1] for s in seqs],np.arange(len(seqs)),_TOKEN)
 with torch.inference_mode():
  a=model(t,l);b=model(u,ul)
 assert torch.isfinite(a).all() and torch.equal(a,b)
 # Batching may introduce small floating differences, but padding must not change
 # the mathematical result or treat a short contig as if it were the padded length.
 with torch.inference_mode():
  for i,s in enumerate(seqs):
   v,vl=_collate([s],np.array([0]),_TOKEN);assert torch.allclose(model(v,vl),a[i:i+1],atol=2e-5,rtol=2e-5)

def test_strict_global_threshold_and_finite_length_transform():
 p=load_deployment_parameters();t=p.threshold
 assert classify_score(t,t)=='Other' and classify_score(np.nextafter(t,1),t)=='Eukaryota'
 scores=np.array([0.,1e-12,.5,1-1e-12,1.]);lengths=np.array([1,999,1500,100000,1000000]);s=transform(scores,lengths,p.config['model']['slim_length_calibration'])
 assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 assert np.array_equal(transform(scores,lengths,{'short':0.,'long':0.}),scores)
 for invalid in [np.nan,np.inf,-.01,1.01]:
  with pytest.raises(ValueError):classify_score(invalid,t)

@pytest.mark.parametrize('path,value',[(('prediction_rule','comparison'),'greater_equal'),(('prediction_rule','threshold'),float('nan')),(('model','dna_other_early_exit','maximum_dna_p_euk'),.001),(('model','fusion_tree','feature_dimension'),1922),(('model','slim_length_calibration','short'),float('inf'))])
def test_unvalidated_contract_mutations_rejected(path,value):
 c=copy.deepcopy(load_deployment_parameters().config);d=c
 for k in path[:-1]:d=d[k]
 d[path[-1]]=value
 with pytest.raises(ValueError):load_parameters(c)

def test_legal_fasta_and_invalid_records_fail_closed(tmp_path):
 f=tmp_path/'input.fna';f.write_text('>first description\nacgtn\n>second\nRYSWKMBDHV\n')
 assert list(fasta_records(f))==[('first','ACGTN'),('second','RYSWKMBDHV')]
 for text in ['>a\nAC-G\n','>a\nACGT\n>a\nA\n','>empty\n','ACGT\n','>a\nAC GT\n']:
  f.write_text(text)
  with pytest.raises(ValueError):list(fasta_records(f))


def test_unbounded_global_features_preserve_learned_length_reference():
 from eukcontigminer.dna_inference import _unbounded_global_features
 model=load_model(load_deployment_parameters().config['model']['dna'],'cpu');torch.set_num_threads(2)
 seq='ACGT'*25000+'ACGTACG';t,l=_collate([seq],np.array([0]),_TOKEN)
 with torch.inference_mode():
  z=model(t,l);assert torch.isfinite(z).all()
  t0,l0=_collate(['ATGCC'*200],np.array([0]),_TOKEN)
  assert torch.equal(model.encoder.forward_features(t0,l0),_unbounded_global_features(model.encoder,t0,l0))
 assert model.encoder.config.maximum_length==100000
