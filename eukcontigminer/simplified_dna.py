"""Six-block whole-contig DNA model and frozen release score transform."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from ._model.hierarchical import HierarchicalKmerTCN, HierarchicalKmerTCNConfig
from ._model.sequence import reverse_complement_batch
from .dna_inference import whole_contig_base_features

class Head(nn.Module):
    def __init__(self,d):
        super().__init__();self.skip=nn.Linear(d,1)
        self.net=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return (self.skip(x)+self.net(x)).squeeze(-1)

class SimplifiedDNA(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if config['channels']!=192 or tuple(config['residual_dilations'])!=(1,2,8,32,128,512) or tuple(config['kmer_orders'])!=(3,4,5,6):
            raise ValueError('Unsupported simplified DNA architecture')
        self.encoder=HierarchicalKmerTCN(HierarchicalKmerTCNConfig(**config))
        for name in ['shared','binary_head','detail_head']:delattr(self.encoder,name)
        d=config['channels']*4+sum(config['kmer_dimensions'])+7
        self.head=Head(d);self.register_buffer('mu',torch.zeros(d));self.register_buffer('sd',torch.ones(d))
    def forward(self,tokens,lengths):
        rc=reverse_complement_batch(tokens,lengths,pad_token=5)
        f=.5*(whole_contig_base_features(self.encoder,tokens,lengths)+whole_contig_base_features(self.encoder,rc,lengths))
        a=torch.cat([f,torch.log10(lengths.float())[:,None]],1)
        return self.head((a-self.mu)/self.sd)

def load_model(binding,device):
    from .deployment import _bound_path
    cp=torch.load(_bound_path(binding,'simplified DNA'),map_location='cpu',weights_only=True)
    model=SimplifiedDNA(cp['config']);model.load_state_dict(cp['state_dict'],strict=True)
    if sum(p.numel() for p in model.parameters())!=binding['parameters'] or not torch.isfinite(model.mu).all() or not torch.isfinite(model.sd).all() or not (model.sd>0).all():
        raise ValueError('Invalid DNA weights or normalization')
    return model.to(device).eval().requires_grad_(False)

def transform(scores,lengths,config):
    from scipy.special import expit,logit
    q=np.asarray(scores,dtype=np.float64);lengths=np.asarray(lengths)
    if q.shape!=lengths.shape or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or (lengths<1).any():raise ValueError('Invalid decision-score input')
    a=float(config['short']);b=float(config['long'])
    if not math.isfinite(a) or not math.isfinite(b):raise ValueError('Nonfinite length coefficients')
    if a==b==0:return q.copy()
    x=np.log2(np.clip(lengths,1000,10000)/2000.)
    return expit(logit(np.clip(q,1e-15,1-1e-15))-a*np.minimum(x,0)-b*np.maximum(x,0))

def load_parameters(config):
    from .deployment import DeploymentParameters,TREE_FEATURE_ORDER
    from .nt_runtime import validate_binding
    m=config['model'];p=config['prediction_rule'];d=m['dna'];tree=m['fusion_tree'];selection=m['feature_definition']['selection'];route=m['dna_other_early_exit']
    if config['schema']!='eukcontigminer.release_model.v6' or config['status'] not in ['frozen_confirmation_candidate','released']:
        raise ValueError('Invalid simplified release identity')
    if p['comparison']!='strict_greater_than' or p['equal_threshold_label']!='Other' or config['binary_target']['unknown_class'] is not False:
        raise ValueError('Invalid binary decision contract')
    if not math.isfinite(p['threshold']) or not 0<=p['threshold']<=1:raise ValueError('Invalid threshold')
    if d['schema']!='ecm.dna.192x6.kmer.v1' or d['parameters']!=2409641:raise ValueError('Invalid DNA binding')
    if tree['feature_dimension']!=1923 or tree['feature_order']!=TREE_FEATURE_ORDER or tree['objective']!='binary:logistic' or tree['xgboost_version']!='3.2.0' or tree['rounds'] not in [400,600]:raise ValueError('Invalid fusion binding')
    if selection['maximum_orfs']!=2 or selection['aggregation']!='mean_max' or selection['reverse_complement_invariant'] is not True or m['feature_definition']['feature_dimension']!=1920:raise ValueError('Invalid protein representation')
    if route['status'] not in ['validation_frozen_guarded','validated_guarded'] or route['maximum_dna_p_euk']!=.01 or route['fallback_logit_margin']!=2. or d['maximum_inference_length']!=100000:raise ValueError('Unexpected guarded inference contract')
    if config['inference_contract']['reference_free'] is not True or config['inference_contract']['reference_database'] is not None or config['inference_contract']['external_similarity_search'] is not False:raise ValueError('Reference-free inference required')
    esmc=m['esmc']
    if esmc['name']!='esmc_300m' or esmc['layers']!=30 or esmc['embedding_dimension']!=960 or esmc['release_package']!='esm 3.2.1' or esmc['use_flash_attention'] is not False:raise ValueError('Invalid protein backbone')
    validate_binding(m['nt_adapter']);transform(np.array([.5]),np.array([1000]),m['slim_length_calibration'])
    return DeploymentParameters(model_id=config['model_id'],protein_family='esmc',early_exit_parity_validated=False,threshold=float(p['threshold']),early_exit_other_max_score=.01,positive_alpha=0.,negative_alpha=0.,probe_center=float(m['probe_logit_center']),probe_scale=float(m['probe_logit_scale']),secondary_probe_center=None,secondary_probe_scale=None,secondary_source_alpha=None,short_alpha=None,long_alpha=None,piecewise_boundary_bp=None,config=config)
